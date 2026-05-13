"""Run replicate SUMO experiments across scenarios x controllers x seeds.

For each run, logs the full per-timestep trace (ttime, NS/EW CO2, true counts,
observed counts, bus counts, phase, NS/EW idle) and derives four per-run
summary metrics:

- cum_idle   total NS+EW vehicle-waiting-seconds summed over the run
- cum_co2    total NS+EW CO2 emissions summed over the run (mg)
- bus_pct    fraction of ticks with at least one bus on an approach where
             the green phase matched the bus's direction (paper's bus
             service rate)
- switches   number of phase transitions over the run

DQN is trained from scratch per scenario for `--dqn-episodes` episodes
under that scenario's toggles, then evaluated in inference mode against
each seed; the rule-based and active-inference controllers learn no
parameters online and are evaluated directly.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random as rnd
import time as wall_time
from collections import deque
from typing import Dict, List, Tuple

import attr
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

import torch
import torch.nn as nn
import torch.optim as optim

import traci

REPO = os.path.dirname(os.path.realpath(__file__))
SUMO_BIN = os.environ.get("SUMO_BIN", "sumo")
SUMO_CFG = os.path.join(REPO, "intersection2.sumocfg")
TL_ID = "center"

PHASE_NS_GREEN = 0
PHASE_EW_GREEN = 2

# Scenario toggles, set globally per scenario before each batch
WEATHER_ON = False
ACCIDENT_ON = False
OCCLUSION_ON = False

ACCIDENT_TRIGGER_TIME = 2000
ACCIDENT_DURATION = 300
ACCIDENT_ZONE_M = 10
WEATHER_START = 1500
WEATHER_END = 2000
EVAL_EVERY = 10


# ============================================================================ #
# SUMO lifecycle                                                               #
# ============================================================================ #
def start_sumo(seed: int, label: str | None = None):
    args = [
        SUMO_BIN, "-c", SUMO_CFG,
        "--no-step-log", "--no-warnings",
        "--seed", str(seed), "--start",
    ]
    if label is None:
        traci.start(args)
    else:
        traci.start(args, label=label)


# ============================================================================ #
# Per-step machinery shared by all controllers                                 #
# ============================================================================ #
@attr.s(auto_attribs=True)
class EpisodeState:
    base_weather: int = 0
    accident_pending: bool = False
    accident_active: bool = False
    accident_done: bool = False
    victim: str | None = None
    accident_stop_time: float | None = None
    east_lane_length: float | None = None


def update_weather(t: float, st: EpisodeState):
    if not WEATHER_ON:
        return
    if t == WEATHER_START:
        st.base_weather = 1
    elif t == WEATHER_END:
        st.base_weather = 0


def update_accident(t: float, st: EpisodeState):
    if not ACCIDENT_ON:
        return
    if st.east_lane_length is None:
        try:
            st.east_lane_length = traci.lane.getLength("east_in_0")
        except Exception:
            st.east_lane_length = 100.0

    if (t >= ACCIDENT_TRIGGER_TIME
            and not st.accident_pending and not st.accident_active
            and not st.accident_done and st.victim is None):
        st.accident_pending = True

    if st.accident_pending:
        near_thr = st.east_lane_length - ACCIDENT_ZONE_M
        best_vid, best_pos = None, -1.0
        for vid in traci.vehicle.getIDList():
            if traci.vehicle.getRoadID(vid) == "east_in":
                pos = traci.vehicle.getLanePosition(vid)
                if pos >= near_thr and pos > best_pos:
                    best_pos = pos
                    best_vid = vid
        if best_vid is not None:
            st.victim = best_vid
            st.accident_pending = False
            st.accident_active = True
            st.accident_stop_time = t
            traci.vehicle.setSpeed(best_vid, 0)
            traci.vehicle.setSpeedMode(best_vid, 0)

    if st.accident_active and st.victim is not None:
        try:
            traci.vehicle.setSpeed(st.victim, 0)
        except traci.TraCIException:
            st.accident_active = False
            st.accident_done = True
            st.victim = None
        if (st.accident_active and st.victim is not None
                and t >= st.accident_stop_time + ACCIDENT_DURATION):
            try:
                traci.vehicle.remove(st.victim)
            except traci.TraCIException:
                pass
            st.accident_active = False
            st.victim = None
            st.accident_done = True


def classify_vehicles():
    NS_cars, NS_bus, NS_truck = [], [], []
    EW_cars, EW_bus, EW_truck = [], [], []
    for i in traci.vehicle.getIDList():
        try:
            j = traci.vehicle.getRoadID(i)
        except Exception:
            continue
        if j in ("north_in", "south_in", "east_in", "west_in"):
            if any(x in i for x in ("car_s", "car_n")): NS_cars.append(i)
            if any(x in i for x in ("bus_s", "bus_n")): NS_bus.append(i)
            if any(x in i for x in ("truck_s", "truck_n")): NS_truck.append(i)
            if any(x in i for x in ("car_e", "car_w")): EW_cars.append(i)
            if any(x in i for x in ("bus_e", "bus_w")): EW_bus.append(i)
            if any(x in i for x in ("truck_e", "truck_w")): EW_truck.append(i)
    return NS_cars, NS_bus, NS_truck, EW_cars, EW_bus, EW_truck


def safe_sum(fn, vids):
    total = 0.0
    for v in vids:
        try:
            total += fn(v)
        except Exception:
            pass
    return total


def noisy_observe(NS_true, EW_true, NS_truck, NS_bus, EW_truck, EW_bus, base_weather):
    alpha = 1 / 200
    weather_alpha = 0.8
    p_NS = (weather_alpha ** base_weather) * (1 - max(1, NS_true - 1) * alpha)
    p_EW = (weather_alpha ** base_weather) * (1 - max(1, EW_true - 1) * alpha)
    p_NS = float(np.clip(p_NS, 0.05, 1.0))
    p_EW = float(np.clip(p_EW, 0.05, 1.0))
    NS_obs = int(np.random.binomial(NS_true, p_NS))
    EW_obs = int(np.random.binomial(EW_true, p_EW))

    if OCCLUSION_ON:
        occ_NS = 0
        nbig = len(NS_truck) + len(NS_bus)
        if NS_true >= 3 * nbig and nbig > 0:
            occ_NS = int(np.sum(np.random.choice([0, 1, 2, 3], nbig,
                                                 p=[0.5, 0.3, 0.15, 0.05])))
        occ_EW = 0
        ebig = len(EW_truck) + len(EW_bus)
        if EW_true >= 3 * ebig and ebig > 0:
            occ_EW = int(np.sum(np.random.choice([0, 1, 2, 3], ebig,
                                                 p=[0.5, 0.3, 0.15, 0.05])))
        NS_obs = max(0, NS_obs - occ_NS)
        EW_obs = max(0, EW_obs - occ_EW)
    return NS_obs, EW_obs


# ============================================================================ #
# Per-run trace + summary                                                      #
# ============================================================================ #
TRACE_COLUMNS = [
    "ttime", "North_CO2", "East_CO2",
    "vehicles_NS", "vehicles_EW", "observed_NS", "observed_EW",
    "bus_count_NS", "bus_count_EW", "phase",
    "idle_time_NS", "idle_time_EW",
]


def empty_trace() -> Dict[str, list]:
    return {c: [] for c in TRACE_COLUMNS}


def trace_append(trace, t, NS_CO2, EW_CO2, NS_true, EW_true,
                 NS_obs, EW_obs, n_NS_bus, n_EW_bus, phase,
                 NS_idle, EW_idle):
    trace["ttime"].append(t)
    trace["North_CO2"].append(NS_CO2)
    trace["East_CO2"].append(EW_CO2)
    trace["vehicles_NS"].append(NS_true)
    trace["vehicles_EW"].append(EW_true)
    trace["observed_NS"].append(NS_obs)
    trace["observed_EW"].append(EW_obs)
    trace["bus_count_NS"].append(n_NS_bus)
    trace["bus_count_EW"].append(n_EW_bus)
    trace["phase"].append(phase)
    trace["idle_time_NS"].append(NS_idle)
    trace["idle_time_EW"].append(EW_idle)


def summarise(trace: Dict[str, list]) -> Dict[str, float]:
    df = pd.DataFrame(trace)
    cum_idle = float((df["idle_time_NS"] + df["idle_time_EW"]).sum())
    cum_co2 = float((df["North_CO2"] + df["East_CO2"]).sum())
    # phase switches (per paper: any change in 'phase' column)
    switches = int((df["phase"].diff().fillna(0) != 0).astype(int).sum())
    # bus service rate per paper: among ticks with bus on either approach,
    # fraction where the green phase serves the bus's direction
    bus_served = []
    for _, r in df.iterrows():
        total_bus = r["bus_count_NS"] + r["bus_count_EW"]
        if total_bus == 0:
            continue
        ns_ok = r["bus_count_NS"] > 0 and r["phase"] == PHASE_NS_GREEN
        ew_ok = r["bus_count_EW"] > 0 and r["phase"] == PHASE_EW_GREEN
        bus_served.append(1.0 if (ns_ok or ew_ok) else 0.0)
    bus_pct = 100.0 * float(np.mean(bus_served)) if bus_served else float("nan")
    return {"cum_idle": cum_idle, "cum_co2": cum_co2,
            "bus_pct": bus_pct, "switches": switches,
            "n_ticks": len(df)}


# ============================================================================ #
# Rule-based controller                                                        #
# ============================================================================ #
RB_MIN_EVALS = 2
RB_MAX_EVALS = 6
RB_BUS_BONUS = 5
RB_MAIN_BONUS = 2


def rule_based_decision(NS_obs, EW_obs, NS_bus, EW_bus, rb_phase, evals_in_phase):
    if evals_in_phase < RB_MIN_EVALS:
        return rb_phase
    if evals_in_phase >= RB_MAX_EVALS:
        return PHASE_EW_GREEN if rb_phase == PHASE_NS_GREEN else PHASE_NS_GREEN
    NS_score = NS_obs + RB_BUS_BONUS * len(NS_bus)
    EW_score = EW_obs + RB_BUS_BONUS * len(EW_bus) + RB_MAIN_BONUS
    if rb_phase == PHASE_NS_GREEN and EW_score > NS_score:
        return PHASE_EW_GREEN
    if rb_phase == PHASE_EW_GREEN and NS_score > EW_score:
        return PHASE_NS_GREEN
    return rb_phase


def run_rule_based(seed: int) -> Dict:
    np.random.seed(seed)
    rnd.seed(seed)
    start_sumo(seed, label=f"rb_{seed}_{wall_time.time()}")
    st = EpisodeState()
    rb_phase = PHASE_EW_GREEN
    evals_in_phase = RB_MIN_EVALS
    traci.trafficlight.setPhase(TL_ID, rb_phase)
    traci.trafficlight.setPhaseDuration(TL_ID, EVAL_EVERY)
    trace = empty_trace()
    eval_time = 10
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            t = traci.simulation.getTime()
            update_weather(t, st)
            update_accident(t, st)
            if t == eval_time:
                eval_time += EVAL_EVERY
                NS_cars, NS_bus, NS_truck, EW_cars, EW_bus, EW_truck = classify_vehicles()
                NS_all = NS_cars + NS_bus + NS_truck
                EW_all = EW_cars + EW_bus + EW_truck
                NS_idle = safe_sum(traci.vehicle.getWaitingTime, NS_all)
                EW_idle = safe_sum(traci.vehicle.getWaitingTime, EW_all)
                NS_CO2 = safe_sum(traci.vehicle.getCO2Emission, NS_all)
                EW_CO2 = safe_sum(traci.vehicle.getCO2Emission, EW_all)
                NS_true = len(NS_all); EW_true = len(EW_all)
                NS_obs, EW_obs = noisy_observe(NS_true, EW_true,
                                               NS_truck, NS_bus, EW_truck, EW_bus,
                                               st.base_weather)
                evals_in_phase += 1
                desired = rule_based_decision(NS_obs, EW_obs, NS_bus, EW_bus,
                                              rb_phase, evals_in_phase)
                if desired != rb_phase:
                    traci.trafficlight.setPhase(TL_ID, desired)
                    rb_phase = desired
                    evals_in_phase = 0
                trace_append(trace, t, NS_CO2, EW_CO2, NS_true, EW_true,
                             NS_obs, EW_obs, len(NS_bus), len(EW_bus), rb_phase,
                             NS_idle, EW_idle)
    finally:
        traci.close()
    return trace


# ============================================================================ #
# Active-inference controller                                                  #
# ============================================================================ #
AI_STATES = 6
AI_PRIOR = np.array(2 * [[0.5, 0.4, 0.08, 0.01, 0.01, 0]])

AI_TRANS_RED_NS = np.array([
    [0.4, 0.55, 0.04, 0.01, 0.00, 0.00],
    [0.05, 0.30, 0.55, 0.09, 0.01, 0.00],
    [0.00, 0.00, 0.20, 0.70, 0.08, 0.02],
    [0.00, 0.00, 0.00, 0.20, 0.60, 0.20],
    [0.00, 0.00, 0.00, 0.00, 0.70, 0.30],
    [0.00, 0.00, 0.00, 0.00, 0.10, 0.90],
])
AI_TRANS_GREEN_NS = np.array([
    [0.80, 0.18, 0.02, 0.00, 0.00, 0.00],
    [0.15, 0.60, 0.22, 0.03, 0.00, 0.00],
    [0.08, 0.32, 0.44, 0.14, 0.02, 0.00],
    [0.02, 0.08, 0.33, 0.42, 0.13, 0.02],
    [0.01, 0.03, 0.33, 0.43, 0.15, 0.05],
    [0.05, 0.10, 0.25, 0.30, 0.20, 0.10],
])
AI_TRANS_RED_EW = np.array([
    [0.4, 0.55, 0.04, 0.01, 0.00, 0.00],
    [0.05, 0.30, 0.55, 0.09, 0.01, 0.00],
    [0.00, 0.00, 0.40, 0.50, 0.08, 0.02],
    [0.00, 0.00, 0.00, 0.20, 0.60, 0.20],
    [0.00, 0.00, 0.00, 0.00, 0.70, 0.30],
    [0.00, 0.00, 0.00, 0.00, 0.10, 0.90],
])
AI_TRANS_GREEN_EW = np.array([
    [0.80, 0.18, 0.02, 0.00, 0.00, 0.00],
    [0.15, 0.60, 0.22, 0.03, 0.00, 0.00],
    [0.08, 0.32, 0.44, 0.14, 0.02, 0.00],
    [0.02, 0.08, 0.33, 0.42, 0.13, 0.02],
    [0.01, 0.03, 0.33, 0.43, 0.15, 0.05],
    [0.00, 0.00, 0.00, 0.30, 0.50, 0.20],
])

AI_PARAMS = np.load(os.path.join(REPO, "frozen_params.npz"))
AI_MEANS = AI_PARAMS["MEANS"]
AI_SIGMAS = AI_PARAMS["SIGMAS"]
AI_PREFERRED_MEANS = [5, 14, 0]
AI_PREFERRED_COV = AI_SIGMAS[0]


def ai_decide(NS_obs, EW_obs, NS_CO2, EW_CO2, NS_bus, EW_bus, prior_states):
    obs = np.array([
        [NS_obs, NS_CO2 / 1000, len(NS_bus)],
        [EW_obs, EW_CO2 / 1000, len(EW_bus)],
    ])
    lights = 2
    post = np.zeros((lights, AI_STATES))
    for i in range(lights):
        for j in range(AI_STATES):
            mvn = multivariate_normal(mean=AI_MEANS[j], cov=AI_SIGMAS[j])
            post[i, j] = mvn.logpdf(obs[i]) + np.log(prior_states[i, j] + 1e-300)
    post -= post.max(axis=1, keepdims=True)
    post = np.exp(post)
    post /= post.sum(axis=1, keepdims=True)

    gn_next = np.array([post[0] @ AI_TRANS_GREEN_NS, post[1] @ AI_TRANS_RED_EW])
    ge_next = np.array([post[0] @ AI_TRANS_RED_NS, post[1] @ AI_TRANS_GREEN_EW])

    log_pref = np.zeros(AI_STATES)
    Sigma_inv = np.linalg.inv(AI_PREFERRED_COV)
    log_det = np.log(np.linalg.det(AI_PREFERRED_COV))
    d = len(AI_PREFERRED_MEANS)
    for j in range(AI_STATES):
        diff = np.array(AI_PREFERRED_MEANS) - AI_MEANS[j]
        log_pref[j] = -0.5 * (
            d * np.log(2 * np.pi) + log_det
            + np.trace(Sigma_inv @ AI_SIGMAS[j])
            + diff.T @ Sigma_inv @ diff
        )
    log_pref -= log_pref.max()
    log_pref /= abs(log_pref.min())

    scores_ns = gn_next @ log_pref
    scores_ew = ge_next @ log_pref
    denom = NS_obs + EW_obs + 1e-300
    pv_ns = (NS_obs / denom) * scores_ns[0] + (EW_obs / denom) * scores_ns[1]
    pv_ew = (EW_obs / denom) * scores_ew[0] + (NS_obs / denom) * scores_ew[1]
    norm = abs(pv_ns + pv_ew + 1e-300)
    pv_ns /= norm
    pv_ew /= norm

    eps = 1e-6
    gn = np.clip(gn_next, eps, 1)
    ge = np.clip(ge_next, eps, 1)
    pr = np.clip(prior_states, eps, 1)
    kl_ns = np.sum(gn * (np.log(gn) - np.log(pr)))
    kl_ew = np.sum(ge * (np.log(ge) - np.log(pr)))
    knorm = abs(kl_ns + kl_ew + 1e-300)
    kl_ns /= knorm
    kl_ew /= knorm

    EFE_ns = -pv_ns - 0.5 * kl_ns
    EFE_ew = -pv_ew - 0.5 * kl_ew
    beta = -3
    exp_ns = math.exp(beta * EFE_ns)
    exp_ew = math.exp(beta * EFE_ew)
    p_ns = exp_ns / (exp_ns + exp_ew)
    decision = int(np.random.choice([0, 2], p=[p_ns, 1 - p_ns]))
    new_prior = gn_next if decision == 0 else ge_next
    new_prior /= new_prior.sum(axis=1, keepdims=True)
    return decision, new_prior


def run_active_inference(seed: int) -> Dict:
    np.random.seed(seed)
    rnd.seed(seed)
    start_sumo(seed, label=f"ai_{seed}_{wall_time.time()}")
    st = EpisodeState()
    prior = AI_PRIOR.copy()
    decision = PHASE_EW_GREEN
    traci.trafficlight.setPhase(TL_ID, decision)
    trace = empty_trace()
    eval_time = 10
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            t = traci.simulation.getTime()
            update_weather(t, st)
            update_accident(t, st)
            if t == eval_time:
                eval_time += EVAL_EVERY
                NS_cars, NS_bus, NS_truck, EW_cars, EW_bus, EW_truck = classify_vehicles()
                NS_all = NS_cars + NS_bus + NS_truck
                EW_all = EW_cars + EW_bus + EW_truck
                NS_idle = safe_sum(traci.vehicle.getWaitingTime, NS_all)
                EW_idle = safe_sum(traci.vehicle.getWaitingTime, EW_all)
                NS_CO2 = safe_sum(traci.vehicle.getCO2Emission, NS_all)
                EW_CO2 = safe_sum(traci.vehicle.getCO2Emission, EW_all)
                NS_true = len(NS_all); EW_true = len(EW_all)
                NS_obs, EW_obs = noisy_observe(NS_true, EW_true,
                                               NS_truck, NS_bus, EW_truck, EW_bus,
                                               st.base_weather)
                decision, prior = ai_decide(NS_obs, EW_obs, NS_CO2, EW_CO2,
                                            NS_bus, EW_bus, prior)
                traci.trafficlight.setPhase(TL_ID, decision)
                trace_append(trace, t, NS_CO2, EW_CO2, NS_true, EW_true,
                             NS_obs, EW_obs, len(NS_bus), len(EW_bus), decision,
                             NS_idle, EW_idle)
    finally:
        traci.close()
    return trace


# ============================================================================ #
# DQN controller                                                               #
# ============================================================================ #
STATE_DIM = 6
ACTION_DIM = 2
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
BUFFER_SIZE = 50000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.95
TARGET_UPDATE = 360


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, ACTION_DIM),
        )

    def forward(self, x):
        return self.net(x)


def run_dqn_episode(seed: int, policy_net, target_net, optimizer, memory,
                    epsilon: float, steps_done: int,
                    train: bool):
    np.random.seed(seed)
    rnd.seed(seed)
    torch.manual_seed(seed)
    start_sumo(seed, label=f"dqn_{seed}_{wall_time.time()}")
    st = EpisodeState()
    prev_state, prev_action = None, None
    trace = empty_trace()
    eval_time = 10
    try:
        while (traci.simulation.getMinExpectedNumber() > 0
               and traci.simulation.getTime() < 4000):
            traci.simulationStep()
            t = traci.simulation.getTime()
            update_weather(t, st)
            update_accident(t, st)
            if t == eval_time:
                eval_time += EVAL_EVERY
                NS_cars, NS_bus, NS_truck, EW_cars, EW_bus, EW_truck = classify_vehicles()
                NS_all = NS_cars + NS_bus + NS_truck
                EW_all = EW_cars + EW_bus + EW_truck
                NS_idle = safe_sum(traci.vehicle.getWaitingTime, NS_all)
                EW_idle = safe_sum(traci.vehicle.getWaitingTime, EW_all)
                NS_CO2 = safe_sum(traci.vehicle.getCO2Emission, NS_all)
                EW_CO2 = safe_sum(traci.vehicle.getCO2Emission, EW_all)
                NS_true = len(NS_all); EW_true = len(EW_all)
                NS_obs, EW_obs = noisy_observe(NS_true, EW_true,
                                               NS_truck, NS_bus, EW_truck, EW_bus,
                                               st.base_weather)
                state = np.array([NS_obs, EW_obs, NS_CO2, EW_CO2,
                                  len(NS_bus), len(EW_bus)], dtype=np.float32)
                if train and rnd.random() < epsilon:
                    action = rnd.randint(0, 1)
                else:
                    with torch.no_grad():
                        action = int(policy_net(torch.from_numpy(state).unsqueeze(0))
                                     .argmax().item())
                reward = -(
                    (NS_idle + EW_idle)
                    + 0.0001 * (NS_CO2 + EW_CO2)
                    - 0.5 * (len(NS_bus) + len(EW_bus))
                )
                if train and prev_state is not None:
                    memory.append((prev_state, prev_action, reward, state, 0))
                    if len(memory) >= BATCH_SIZE:
                        batch = rnd.sample(memory, BATCH_SIZE)
                        s, a, r, sn, d_ = zip(*batch)
                        s = torch.from_numpy(np.stack(s))
                        sn = torch.from_numpy(np.stack(sn))
                        a_t = torch.LongTensor(a).unsqueeze(1)
                        r_t = torch.FloatTensor(r)
                        d_t = torch.FloatTensor(d_)
                        q = policy_net(s).gather(1, a_t).squeeze()
                        with torch.no_grad():
                            nq = target_net(sn).max(1)[0]
                            tgt = r_t + GAMMA * nq * (1 - d_t)
                        loss = nn.MSELoss()(q, tgt)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                decision = PHASE_NS_GREEN if action == 0 else PHASE_EW_GREEN
                traci.trafficlight.setPhase(TL_ID, decision)
                prev_state, prev_action = state, action
                steps_done += 1
                if train and steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                if train:
                    epsilon = max(EPS_END, epsilon * EPS_DECAY)
                trace_append(trace, t, NS_CO2, EW_CO2, NS_true, EW_true,
                             NS_obs, EW_obs, len(NS_bus), len(EW_bus), decision,
                             NS_idle, EW_idle)
    finally:
        traci.close()
    return trace, epsilon, steps_done


def train_dqn(n_episodes: int = 100, train_seed: int = 42):
    np.random.seed(train_seed)
    rnd.seed(train_seed)
    torch.manual_seed(train_seed)
    policy_net = DQN()
    target_net = DQN()
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = deque(maxlen=BUFFER_SIZE)
    epsilon = EPS_START
    steps_done = 0
    for ep in range(n_episodes):
        ep_seed = train_seed * 1000 + ep
        trace, epsilon, steps_done = run_dqn_episode(
            ep_seed, policy_net, target_net, optimizer, memory,
            epsilon, steps_done, train=True)
        if (ep + 1) % 10 == 0 or ep == 0:
            s = summarise(trace)
            print(f"  train ep {ep+1}/{n_episodes}  "
                  f"cum_idle={s['cum_idle']:.0f}  eps={epsilon:.3f}",
                  flush=True)
    return policy_net


def run_dqn_eval(seed: int, policy_net) -> Dict:
    target_net = DQN()
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = deque(maxlen=1)
    trace, _, _ = run_dqn_episode(seed, policy_net, target_net, optimizer, memory,
                                  epsilon=0.0, steps_done=0, train=False)
    return trace


# ============================================================================ #
# Scenario plumbing + main                                                     #
# ============================================================================ #
SCENARIO_TOGGLES = {
    1: (False, False, False),
    2: (False, False, True),
    3: (True, False, True),
    4: (True, True, True),
}


def set_scenario(scenario: int):
    global WEATHER_ON, ACCIDENT_ON, OCCLUSION_ON
    WEATHER_ON, ACCIDENT_ON, OCCLUSION_ON = SCENARIO_TOGGLES[scenario]
    print(f"Scenario {scenario}: weather={WEATHER_ON} "
          f"accident={ACCIDENT_ON} occlusion={OCCLUSION_ON}", flush=True)


CONTROLLER_LABEL = {"rb": "Rule-Based", "ai": "Active Inference", "dqn": "DQN"}


def write_trace(traces_path: str, scenario: int, controller: str, seed: int,
                trace: Dict[str, list], header_written: List[bool]):
    n = len(trace["ttime"])
    rows = []
    for i in range(n):
        rows.append([
            scenario, controller, seed,
            trace["ttime"][i], trace["North_CO2"][i], trace["East_CO2"][i],
            trace["vehicles_NS"][i], trace["vehicles_EW"][i],
            trace["observed_NS"][i], trace["observed_EW"][i],
            trace["bus_count_NS"][i], trace["bus_count_EW"][i],
            trace["phase"][i], trace["idle_time_NS"][i], trace["idle_time_EW"][i],
        ])
    mode = "a" if header_written[0] else "w"
    with open(traces_path, mode, newline="") as f:
        w = csv.writer(f)
        if not header_written[0]:
            w.writerow(["scenario", "controller", "seed"] + TRACE_COLUMNS)
            header_written[0] = True
        w.writerows(rows)


def write_summary(summary_path: str, scenario: int, controller: str, seed: int,
                  summary: Dict[str, float], header_written: List[bool]):
    mode = "a" if header_written[0] else "w"
    with open(summary_path, mode, newline="") as f:
        w = csv.writer(f)
        if not header_written[0]:
            w.writerow(["scenario", "controller", "seed",
                        "cum_idle", "cum_co2", "bus_pct", "switches", "n_ticks"])
            header_written[0] = True
        w.writerow([scenario, controller, seed,
                    f"{summary['cum_idle']:.3f}",
                    f"{summary['cum_co2']:.3f}",
                    f"{summary['bus_pct']:.4f}",
                    summary["switches"],
                    summary["n_ticks"]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controllers", nargs="+",
                   default=["rb", "ai", "dqn"], choices=["rb", "ai", "dqn"])
    p.add_argument("--seeds", type=int, nargs=2, default=[1, 100])
    p.add_argument("--scenarios", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--dqn-episodes", type=int, default=100)
    p.add_argument("--traces-out", required=True,
                   help="CSV file for per-timestep traces")
    p.add_argument("--summary-out", required=True,
                   help="CSV file for per-run summary stats")
    args = p.parse_args()

    seed_lo, seed_hi = args.seeds
    seeds = list(range(seed_lo, seed_hi + 1))
    trace_hdr = [False]
    sum_hdr = [False]
    if os.path.exists(args.traces_out):
        os.remove(args.traces_out)
    if os.path.exists(args.summary_out):
        os.remove(args.summary_out)
    total_t0 = wall_time.time()

    for scenario in args.scenarios:
        set_scenario(scenario)

        if "rb" in args.controllers:
            print(f"\n=== S{scenario} Rule-Based: {len(seeds)} seeds ===", flush=True)
            for s in seeds:
                t0 = wall_time.time()
                trace = run_rule_based(s)
                summary = summarise(trace)
                write_trace(args.traces_out, scenario, "Rule-Based", s, trace, trace_hdr)
                write_summary(args.summary_out, scenario, "Rule-Based", s, summary, sum_hdr)
                print(f"  S{scenario} rb  seed={s:3d}  "
                      f"cum_idle={summary['cum_idle']:8.0f}  "
                      f"sw={summary['switches']:3d}  "
                      f"({wall_time.time()-t0:.1f}s)", flush=True)

        if "ai" in args.controllers:
            print(f"\n=== S{scenario} Active Inference: {len(seeds)} seeds ===", flush=True)
            for s in seeds:
                t0 = wall_time.time()
                trace = run_active_inference(s)
                summary = summarise(trace)
                write_trace(args.traces_out, scenario, "Active Inference", s, trace, trace_hdr)
                write_summary(args.summary_out, scenario, "Active Inference", s, summary, sum_hdr)
                print(f"  S{scenario} ai  seed={s:3d}  "
                      f"cum_idle={summary['cum_idle']:8.0f}  "
                      f"sw={summary['switches']:3d}  "
                      f"({wall_time.time()-t0:.1f}s)", flush=True)

        if "dqn" in args.controllers:
            print(f"\n=== S{scenario} DQN: train {args.dqn_episodes} ep, "
                  f"eval {len(seeds)} seeds ===", flush=True)
            policy_net = train_dqn(n_episodes=args.dqn_episodes)
            for s in seeds:
                t0 = wall_time.time()
                trace = run_dqn_eval(s, policy_net)
                summary = summarise(trace)
                write_trace(args.traces_out, scenario, "DQN", s, trace, trace_hdr)
                write_summary(args.summary_out, scenario, "DQN", s, summary, sum_hdr)
                print(f"  S{scenario} dqn seed={s:3d}  "
                      f"cum_idle={summary['cum_idle']:8.0f}  "
                      f"sw={summary['switches']:3d}  "
                      f"({wall_time.time()-t0:.1f}s)", flush=True)

    print(f"\nWrote {args.traces_out} and {args.summary_out} "
          f"(total elapsed: {(wall_time.time()-total_t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
