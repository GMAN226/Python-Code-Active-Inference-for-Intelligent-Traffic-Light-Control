import traci
import numpy as np
import random as rnd
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

SUMO_BINARY = "sumo"
SUMO_CFG    = r"C:/Users/giorg/OneDrive/Laptop Lenovo/BIG STUFF/intersection2.sumocfg" 
TL_ID       = "center"

PHASE_NS_GREEN = 0
PHASE_EW_GREEN = 2

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
EPISODES = 100

# ---------------- TOGGLE SWITCHES ----------------
OCCLUSION_ON = False
WEATHER_ON   = False
ACCIDENT_ON  = False
# -------------------------------------------------


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_DIM)
        )

    def forward(self, x):
        return self.net(x)


policy_net = DQN()
target_net = DQN()
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = deque(maxlen=BUFFER_SIZE)

epsilon = EPS_START
steps_done = 0


def safe_CO2(vehicle_list):
    total = 0
    for v in vehicle_list:
        try:
            total += traci.vehicle.getCO2Emission(v)
        except:
            pass
    return total


def safe_wait(vehicle_list):
    total = 0
    for v in vehicle_list:
        try:
            total += traci.vehicle.getWaitingTime(v)
        except:
            pass
    return total


def select_action(state):
    global epsilon
    if rnd.random() < epsilon:
        return rnd.randint(0, 1)
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0)
        return policy_net(state_t).argmax().item()


def optimize():
    if len(memory) < BATCH_SIZE:
        return

    batch = rnd.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states      = torch.FloatTensor(states)
    actions     = torch.LongTensor(actions).unsqueeze(1)
    rewards     = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones       = torch.FloatTensor(dones)

    q_values = policy_net(states).gather(1, actions).squeeze()

    with torch.no_grad():
        next_q = target_net(next_states).max(1)[0]
        target = rewards + GAMMA * next_q * (1 - dones)

    loss = nn.MSELoss()(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


episode_avg_idle = []


for episode in range(EPISODES):
    print(f"\nEpisode {episode+1}")

    traci.start([SUMO_BINARY, "-c", SUMO_CFG, "--start"])

    eval_time = 10
    if episode == EPISODES-1:
        weather_start = 1500
        weather_end   = 2000
    else:
        weather_start = 1500 + np.random.choice([0, -5000], p=[0.40, 0.60])
        weather_end   = 2000 + np.random.choice([0, -5000], p=[0.40, 0.60])

    if episode == EPISODES-1:
         accident_time  = 2000
    else:
         accident_time  = 2000 + np.random.choice([0, -5000], p=[0.05, 0.95])

    base_weather = 0

    prev_state = None
    prev_action = None
    victim = None

    episode_idle_sum = 0
    episode_eval_steps = 0

    time = []
    true_vehicle_count_NS = []
    true_vehicle_count_EW = []
    true_CO2_NS = []
    true_CO2_EW = []
    buses_NS = []
    buses_EW = []
    phase = []
    observed_vehicle_count_NS = []
    observed_vehicle_count_EW = []
    idle_time_NS = []
    idle_time_EW = []

    while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() < 4000:
        traci.simulationStep()
        t = traci.simulation.getTime()

        # ---------------- WEATHER ----------------
        if WEATHER_ON:
            if t == weather_start:
                base_weather = 1
                traci.polygon.add(
                    "bg_overlay",
                    shape=[(-10000, -10000), (10000, -10000), (10000, 10000), (-10000, 10000)],
                    color=(0, 0, 255, 100),
                    fill=True,
                    layer=-1
                )

            if t == weather_end:
                base_weather = 0
                try:
                    traci.polygon.remove("bg_overlay")
                except:
                    pass

        # ---------------- ACCIDENT ----------------
        if ACCIDENT_ON:
            if t == accident_time:
                ids = traci.vehicle.getIDList()
                ew_candidates = []

                for v in ids:
                    try:
                        road = traci.vehicle.getRoadID(v)
                        if road in ["east_in", "west_in"]:
                            pos = traci.vehicle.getLanePosition(v)
                            ew_candidates.append((pos, v))
                    except:
                        continue

                if len(ew_candidates) > 0:
                    victim = max(ew_candidates, key=lambda x: x[0])[1]
                    try:
                        traci.vehicle.setSpeed(victim, 0)
                        traci.vehicle.setColor(victim, (255, 0, 0, 255))
                    except:
                        victim = None

            if t == accident_time + 300 and victim is not None:
                try:
                    if victim in traci.vehicle.getIDList():
                        traci.vehicle.remove(victim)
                except:
                    pass

        # ---------------- EVALUATION ----------------
        if t == eval_time:
            eval_time += 10
            time.append(t)

            NS_cars=[]; NS_bus=[]; NS_truck=[]
            EW_cars=[]; EW_bus=[]; EW_truck=[]

            ids = traci.vehicle.getIDList()

            for i in ids:
                try:
                    j = traci.vehicle.getRoadID(i)
                except:
                    continue

                if j in ["north_in","south_in","east_in","west_in"]:
                    if any(x in i for x in ["car_s","car_n"]): NS_cars.append(i)
                    if any(x in i for x in ["bus_s","bus_n"]): NS_bus.append(i)
                    if any(x in i for x in ["truck_s","truck_n"]): NS_truck.append(i)
                    if any(x in i for x in ["car_e","car_w"]): EW_cars.append(i)
                    if any(x in i for x in ["bus_e","bus_w"]): EW_bus.append(i)
                    if any(x in i for x in ["truck_e","truck_w"]): EW_truck.append(i)

            NS_CO2 = safe_CO2(NS_cars + NS_bus + NS_truck)
            EW_CO2 = safe_CO2(EW_cars + EW_bus + EW_truck)

            NS_idle = safe_wait(NS_cars + NS_bus + NS_truck)
            EW_idle = safe_wait(EW_cars + EW_bus + EW_truck)

            episode_idle_sum += (NS_idle + EW_idle)
            episode_eval_steps += 1

            NS_true = len(NS_cars) + len(NS_bus) + len(NS_truck)
            EW_true = len(EW_cars) + len(EW_bus) + len(EW_truck)

            # ---------------- SENSOR MODEL ----------------
            alpha = 1 / 200
            weather_alpha = 0.8

            p_NS = (weather_alpha ** base_weather) * (1 - max(1, NS_true - 1) * alpha)
            p_EW = (weather_alpha ** base_weather) * (1 - max(1, EW_true - 1) * alpha)

            p_NS = np.clip(p_NS, 0.05, 1.0)
            p_EW = np.clip(p_EW, 0.05, 1.0)

            NS_obs = np.random.binomial(NS_true, p_NS)
            EW_obs = np.random.binomial(EW_true, p_EW)

            # ---------------- OCCLUSION ----------------
            occ_NS = 0
            if OCCLUSION_ON:
                if NS_true >= 3 * (len(NS_truck) + len(NS_bus)) and (len(NS_truck) + len(NS_bus)) > 0:
                    occ_NS = int(np.sum(np.random.choice([0,1,2,3],
                                                         len(NS_truck)+len(NS_bus),
                                                         p=[0.5,0.3,0.15,0.05])))

            occ_EW = 0
            if OCCLUSION_ON:
                if EW_true >= 3 * (len(EW_truck) + len(EW_bus)) and (len(EW_truck) + len(EW_bus)) > 0:
                    occ_EW = int(np.sum(np.random.choice([0,1,2,3],
                                                         len(EW_truck)+len(EW_bus),
                                                         p=[0.5,0.3,0.15,0.05])))

            NS_obs = max(0, NS_obs - occ_NS)
            EW_obs = max(0, EW_obs - occ_EW)

            # ---------------- STATE ----------------
            state = np.array([NS_obs, EW_obs, NS_CO2, EW_CO2, len(NS_bus), len(EW_bus)], dtype=np.float32)

            reward = -(
                (NS_idle + EW_idle)
                + 0.0001 * (NS_CO2 + EW_CO2)
                - 0.5 * (len(NS_bus) + len(EW_bus))
            )

            if prev_state is not None:
                memory.append((prev_state, prev_action, reward, state, 0))
                optimize()

            action = select_action(state)

            decision = PHASE_NS_GREEN if action == 0 else PHASE_EW_GREEN
            traci.trafficlight.setPhase("center", decision)

            prev_state = state
            prev_action = action

            steps_done += 1
            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            epsilon = max(EPS_END, epsilon * EPS_DECAY)

            # logging
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

    traci.close()

    avg_idle = episode_idle_sum / max(1, episode_eval_steps)
    episode_avg_idle.append(avg_idle)
    print(f"Episode {episode+1} - Avg Idle Time: {avg_idle:.4f}")


data = pd.DataFrame({
    "ttime": time,
    "North_CO2": true_CO2_NS,
    "East_CO2": true_CO2_EW,
    "vehicles_NS": true_vehicle_count_NS,
    "vehicles_EW": true_vehicle_count_EW,
    "observed_NS": observed_vehicle_count_NS,
    "observed_EW": observed_vehicle_count_EW,
    "bus_count_NS": buses_NS,
    "bus_count_EW": buses_EW,
    "phase": phase,
    "idle_time_NS": idle_time_NS,
    "idle_time_EW": idle_time_EW,
})

data.to_excel("C:/Users/giorg/OneDrive/Laptop Lenovo/BIG STUFF/output_dqn.xlsx", index=False)

print("Results saved.")
