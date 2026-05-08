import traci
import numpy as np
import pandas as pd
import math
import os
import time as time_module
from scipy.stats import multivariate_normal

np.set_printoptions(suppress=True)

SUMO_BINARY = "sumo-gui"
SUMO_CFG    = "C:/Users/giorg/OneDrive/Laptop Lenovo/BIG STUFF/intersection2.sumocfg"
TL_ID       = "center"
EVAL_EVERY  = 10

PHASE_NS_GREEN = 0
PHASE_EW_GREEN = 2

NS_LANES = ["north_in_0", "south_in_0", "north_in_1", "south_in_1"]
EW_LANES = ["east_in_0",  "west_in_0",  "east_in_1",  "west_in_1"]

# -------------------------- TOGGLE SWITCHES -------------------------- #
WEATHER_ON   = False
ACCIDENT_ON  = False
OCCLUSION_ON = False
# --------------------------------------------------------------------- #

# -------------------------- Rule-Based Parameters -------------------- #
MIN_GREEN  = 20
MAX_GREEN  = 60
BUS_BONUS  = 5
MAIN_BONUS = 2

# -------------------------- Accident Parameters ---------------------- #
ACCIDENT_TRIGGER_TIME = 2000
ACCIDENT_DURATION     = 300
ACCIDENT_ZONE_M       = 10

# -------------------------- Switching Penalty ------------------------ #
SWITCH_PENALTY = 0.0

# -------------------------- Start SUMO ------------------------------- #
traci.start([SUMO_BINARY, "-c", SUMO_CFG, "--start"], label=str(time_module.time()))

# -------------------------- Baseline States -------------------------- #
np.random.seed(42)
current_phase  = PHASE_EW_GREEN
next_eval_time = 0.0
base_weather   = 0

weather_start = 1500
weather_end   = 2000
eval_time     = 10

# -------------------------- Active Inference State ------------------- #
STATES    = ["null", "low", "medium", "high", "extra_high", "jam"]
N_STATES  = len(STATES)
lights    = 2

PRIOR_STATES = np.array(lights * [[0.5, 0.4, 0.08, 0.01, 0.01, 0]])

TRANSITION_RED_NS = np.array([
    [0.4,  0.55,  0.04,  0.01,  0.00,  0.00],
    [0.05, 0.30,  0.55,  0.09,  0.01,  0.00],
    [0.00, 0.00,  0.20,  0.70,  0.08,  0.02],
    [0.00, 0.00,  0.00,  0.20,  0.60,  0.20],
    [0.00, 0.00,  0.00,  0.00,  0.70,  0.30],
    [0.00, 0.00,  0.00,  0.00,  0.10,  0.90],
])

TRANSITION_GREEN_NS = np.array([
    [0.80, 0.18, 0.02, 0.00, 0.00, 0.00],
    [0.15, 0.60, 0.22, 0.03, 0.00, 0.00],
    [0.08, 0.32, 0.44, 0.14, 0.02, 0.00],
    [0.02, 0.08, 0.33, 0.42, 0.13, 0.02],
    [0.01, 0.03, 0.33, 0.43, 0.15, 0.05],
    [0.05, 0.10, 0.25, 0.30, 0.20, 0.10],
])

TRANSITION_RED_EW = np.array([
    [0.4,  0.55,  0.04,  0.01,  0.00,  0.00],
    [0.05, 0.30,  0.55,  0.09,  0.01,  0.00],
    [0.00, 0.00,  0.40,  0.50,  0.08,  0.02],
    [0.00, 0.00,  0.00,  0.20,  0.60,  0.20],
    [0.00, 0.00,  0.00,  0.00,  0.70,  0.30],
    [0.00, 0.00,  0.00,  0.00,  0.10,  0.90],
])

TRANSITION_GREEN_EW = np.array([
    [0.80, 0.18, 0.02, 0.00, 0.00, 0.00],
    [0.15, 0.60, 0.22, 0.03, 0.00, 0.00],
    [0.08, 0.32, 0.44, 0.14, 0.02, 0.00],
    [0.02, 0.08, 0.33, 0.42, 0.13, 0.02],
    [0.01, 0.03, 0.33, 0.43, 0.15, 0.05],
    [0.00, 0.00, 0.00, 0.30, 0.50, 0.20],
])

# -------------------------- Load Frozen Params ----------------------- #
_script_dir  = os.path.dirname(os.path.abspath(__file__))
_params_path = os.path.join(_script_dir, "frozen_params.npz")
data_npz     = np.load(_params_path)
MEANS        = data_npz['MEANS']
SIGMAS       = data_npz['SIGMAS']

decision = 0

PREFERRED_MEANS       = [5, 14, 0]
PREFERRED_COVARIANCES = SIGMAS[0]

# -------------------------- Accident State --------------------------- #
accident_pending   = False
accident_active    = False
accident_done      = False
victim             = None
accident_stop_time = None
east_lane_length   = None

# -------------------------- Summary Table ---------------------------- #
sim_time                  = []
true_vehicle_count_NS     = []
true_vehicle_count_EW     = []
true_CO2_NS               = []
true_CO2_EW               = []
buses_NS                  = []
buses_EW                  = []
phase                     = []
observed_vehicle_count_NS = []
observed_vehicle_count_EW = []
idle_time_NS              = []
idle_time_EW              = []

# ===================================================================== #
try:
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        t = traci.simulation.getTime()

        # -------------------------- Lane Length ------------------------ #
        if east_lane_length is None:
            try:
                east_lane_length = traci.lane.getLength("east_in_0")
            except Exception:
                east_lane_length = 100.0

        # -------------------------- WEATHER ---------------------------- #
        if WEATHER_ON:
            if t == weather_start:
                traci.polygon.add(
                    "bg_overlay",
                    shape=[(-10000, -10000), (10000, -10000),
                           (10000, 10000), (-10000, 10000)],
                    color=(0, 0, 255, 100),
                    fill=True,
                    layer=-1
                )
                base_weather = 1
                print(f"[t={t:.0f}] Weather event started: reduced visibility.")

            if t == weather_end:
                try:
                    traci.polygon.remove("bg_overlay")
                except traci.TraCIException:
                    pass
                base_weather = 0
                print(f"[t={t:.0f}] Weather event ended: visibility restored.")

        # -------------------------- ACCIDENT --------------------------- #
        if ACCIDENT_ON:
            if (t >= ACCIDENT_TRIGGER_TIME
                and not accident_pending
                and not accident_active
                and not accident_done
                and victim is None):

                accident_pending = True
                print(f"[t={t:.0f}] Accident armed — scanning for East-to-West car near traffic light...")

            if accident_pending:
                near_light_threshold = east_lane_length - ACCIDENT_ZONE_M
                best_victim   = None
                best_position = -1.0

                for vid in traci.vehicle.getIDList():
                    if traci.vehicle.getRoadID(vid) == "east_in":
                        pos = traci.vehicle.getLanePosition(vid)
                        if pos >= near_light_threshold and pos > best_position:
                            best_position = pos
                            best_victim   = vid

                if best_victim is not None:
                    victim             = best_victim
                    accident_pending   = False
                    accident_active    = True
                    accident_stop_time = t

                    traci.vehicle.setSpeed(victim, 0)
                    traci.vehicle.setColor(victim, (255, 0, 0, 255))
                    traci.vehicle.setSpeedMode(victim, 0)

                    print(f"[t={t:.0f}] Accident triggered: '{victim}' stopped at "
                          f"{best_position:.1f}m on east_in "
                          f"(road length: {east_lane_length:.1f}m). "
                          f"Clears at t={t + ACCIDENT_DURATION:.0f}.")

            if accident_active and victim is not None:
                try:
                    traci.vehicle.setSpeed(victim, 0)
                except traci.TraCIException:
                    accident_active = False
                    accident_done   = True
                    victim          = None

            if accident_active and victim is not None:
                if t >= accident_stop_time + ACCIDENT_DURATION:
                    try:
                        traci.vehicle.remove(victim)
                        print(f"[t={t:.0f}] Accident cleared: '{victim}' removed.")
                    except traci.TraCIException:
                        pass
                    accident_active = False
                    victim          = None
                    accident_done   = True

        # -------------------------- EVALUATION -------------------------- #
        if t == eval_time:
            eval_time += 10
            sim_time.append(t)

            # ---- Vehicle classification ---- #
            NS_cars  = []; NS_bus = []; NS_truck = []
            EW_cars  = []; EW_bus = []; EW_truck = []

            for i in traci.vehicle.getIDList():
                road = traci.vehicle.getRoadID(i)
                if road in ["north_in", "south_in", "east_in", "west_in"]:
                    if any(x in i for x in ["car_s",   "car_n"]):   NS_cars.append(i)
                    if any(x in i for x in ["bus_s",   "bus_n"]):   NS_bus.append(i)
                    if any(x in i for x in ["truck_s", "truck_n"]): NS_truck.append(i)
                    if any(x in i for x in ["car_e",   "car_w"]):   EW_cars.append(i)
                    if any(x in i for x in ["bus_e",   "bus_w"]):   EW_bus.append(i)
                    if any(x in i for x in ["truck_e", "truck_w"]): EW_truck.append(i)

            # ---- CO2 ---- #
            NS_CO2 = sum(traci.vehicle.getCO2Emission(i) for i in NS_cars + NS_bus + NS_truck)
            EW_CO2 = sum(traci.vehicle.getCO2Emission(i) for i in EW_cars + EW_bus + EW_truck)

            # ---- Idle ---- #
            NS_idle = sum(traci.vehicle.getWaitingTime(i) for i in NS_cars + NS_bus + NS_truck)
            EW_idle = sum(traci.vehicle.getWaitingTime(i) for i in EW_cars + EW_bus + EW_truck)

            # ---- True counts ---- #
            NS_true = len(NS_cars) + len(NS_bus) + len(NS_truck)
            EW_true = len(EW_cars) + len(EW_bus) + len(EW_truck)

            # ---- Sensor noise ---- #
            alpha         = 1 / 200
            weather_alpha = 0.8

            p_NS = (weather_alpha ** base_weather) * (1 - max(1, NS_true - 1) * alpha)
            p_EW = (weather_alpha ** base_weather) * (1 - max(1, EW_true - 1) * alpha)

            NS_obs = np.random.binomial(NS_true, p_NS)
            EW_obs = np.random.binomial(EW_true, p_EW)

            # ---- OCCLUSION ---- #
            if OCCLUSION_ON:
                occ_NS = 0
                if NS_true >= 3 * (len(NS_truck) + len(NS_bus)) and (len(NS_truck) + len(NS_bus)) > 0:
                    occ_NS = int(np.sum(np.random.choice(
                        [0, 1, 2, 3],
                        len(NS_truck) + len(NS_bus),
                        p=[0.5, 0.3, 0.15, 0.05]
                    )))

                occ_EW = 0
                if EW_true >= 3 * (len(EW_truck) + len(EW_bus)) and (len(EW_truck) + len(EW_bus)) > 0:
                    occ_EW = int(np.sum(np.random.choice(
                        [0, 1, 2, 3],
                        len(EW_truck) + len(EW_bus),
                        p=[0.5, 0.3, 0.15, 0.05]
                    )))

                NS_obs = max(0, NS_obs - occ_NS)
                EW_obs = max(0, EW_obs - occ_EW)

            # -------------------------- ACTIVE INFERENCE ---------------- #
            observations = np.array([
                [NS_obs, NS_CO2 / 1000, len(NS_bus)],
                [EW_obs, EW_CO2 / 1000, len(EW_bus)]
            ])

            posterior = np.zeros((lights, N_STATES))

            for i in range(lights):
                for j in range(N_STATES):
                    mean = MEANS[j] if j < len(MEANS) else MEANS[0]
                    cov  = SIGMAS[j] if j < len(SIGMAS) else SIGMAS[0]
                    mvn  = multivariate_normal(mean=mean, cov=cov)
                    posterior[i, j] = mvn.logpdf(observations[i]) + np.log(PRIOR_STATES[i, j] + 1e-300)

            posterior -= posterior.max(axis=1, keepdims=True)
            posterior  = np.exp(posterior)
            posterior /= posterior.sum(axis=1, keepdims=True)

            # ---- Predict next beliefs ---- #
            green_ns_next = np.array([
                posterior[0] @ TRANSITION_GREEN_NS,
                posterior[1] @ TRANSITION_RED_EW
            ])

            green_ew_next = np.array([
                posterior[0] @ TRANSITION_RED_NS,
                posterior[1] @ TRANSITION_GREEN_EW
            ])

            # ---- Preference model ---- #
            log_pref   = np.zeros(N_STATES)
            Sigma_inv  = np.linalg.inv(PREFERRED_COVARIANCES)
            log_det    = np.log(np.linalg.det(PREFERRED_COVARIANCES))
            d          = len(PREFERRED_MEANS)

            for j in range(N_STATES):
                mu_j    = MEANS[j] if j < len(MEANS) else MEANS[0]
                Sigma_j = SIGMAS[j] if j < len(SIGMAS) else SIGMAS[0]
                diff    = np.array(PREFERRED_MEANS) - mu_j
                log_pref[j] = -0.5 * (
                    d * np.log(2 * np.pi)
                    + log_det
                    + np.trace(Sigma_inv @ Sigma_j)
                    + diff.T @ Sigma_inv @ diff
                )

            log_pref -= log_pref.max()
            log_pref /= abs(log_pref.min())

            # ---- Pragmatic value ---- #
            scores_ns = green_ns_next @ log_pref
            scores_ew = green_ew_next @ log_pref

            pv_ns = (NS_obs / (NS_obs + EW_obs + 1e-300)) * scores_ns[0] + \
                    (EW_obs / (NS_obs + EW_obs + 1e-300)) * scores_ns[1]

            pv_ew = (EW_obs / (NS_obs + EW_obs + 1e-300)) * scores_ew[0] + \
                    (NS_obs / (NS_obs + EW_obs + 1e-300)) * scores_ew[1]

            norm = abs(pv_ns + pv_ew + 1e-300)
            pv_ns /= norm
            pv_ew /= norm

            # ---- Epistemic value ---- #
            eps = 1e-6
            gn  = np.clip(green_ns_next, eps, 1)
            ge  = np.clip(green_ew_next, eps, 1)
            pr  = np.clip(PRIOR_STATES,  eps, 1)

            kl_ns = np.sum(gn * (np.log(gn) - np.log(pr)))
            kl_ew = np.sum(ge * (np.log(ge) - np.log(pr)))

            norm  = abs(kl_ns + kl_ew + 1e-300)
            kl_ns /= norm
            kl_ew /= norm

            # ---- Expected Free Energy ---- #
            EFE_ns = -pv_ns - 0.5 * kl_ns
            EFE_ew = -pv_ew - 0.5 * kl_ew

            # ---- Softmax decision ---- #
            beta   = -3
            exp_ns = np.exp(beta * EFE_ns)
            exp_ew = np.exp(beta * EFE_ew)
            p_ns   = exp_ns / (exp_ns + exp_ew)
            p_ew   = 1 - p_ns

            decision = int(np.random.choice([0, 2], p=[p_ns, p_ew]))
            traci.trafficlight.setPhase("center", decision)

            PRIOR_STATES = green_ns_next if decision == 0 else green_ew_next
            PRIOR_STATES /= PRIOR_STATES.sum(axis=1, keepdims=True)

            # ---- Logging ---- #
            true_CO2_NS.append(NS_CO2)
            true_CO2_EW.append(EW_CO2)
            true_vehicle_count_NS.append(NS_true)
            true_vehicle_count_EW.append(EW_true)
            buses_NS.append(len(NS_bus))
            buses_EW.append(len(EW_bus))
            phase.append(decision)
            observed_vehicle_count_NS.append(NS_obs)
            observed_vehicle_count_EW.append(EW_obs)
            idle_time_NS.append(NS_idle)
            idle_time_EW.append(EW_idle)

except Exception as e:
    print(f"Simulation error: {e}")

finally:
    try:
        traci.close()
    except Exception:
        pass

    if sim_time:
        try:
            df = pd.DataFrame({
                "ttime"        : sim_time,
                "North_CO2"    : true_CO2_NS,
                "East_CO2"     : true_CO2_EW,
                "vehicles_NS"  : true_vehicle_count_NS,
                "vehicles_EW"  : true_vehicle_count_EW,
                "observed_NS"  : observed_vehicle_count_NS,
                "observed_EW"  : observed_vehicle_count_EW,
                "bus_count_NS" : buses_NS,
                "bus_count_EW" : buses_EW,
                "phase"        : phase,
                "idle_time_NS" : idle_time_NS,
                "idle_time_EW" : idle_time_EW
            })
            df.to_excel(
                "C:/Users/giorg/OneDrive/Laptop Lenovo/BIG STUFF/output_activeinference.xlsx",
                index=False
            )
            print("Results saved successfully.")
        except Exception as e:
            print(f"Export error: {e}")
    else:
        print("No data collected — output file not written.")
