import traci
import numpy as np
import pandas as pd

SUMO_BINARY = "sumo-gui"
SUMO_CFG    = r"C:/Users/giorg/OneDrive/Laptop Lenovo/BIG STUFF/intersection2.sumocfg"
TL_ID       = "center"

PHASE_NS_GREEN = 0
PHASE_EW_GREEN = 2

# --------------------------Rule-Based Controller Parameters------------------ #
EVAL_EVERY  = 10
MIN_EVALS   = 2
MAX_EVALS   = 6
BUS_BONUS   = 5
MAIN_BONUS  = 2

# --------------------------Accident Parameters------------------------------- #
ACCIDENT_TRIGGER_TIME = 2000
ACCIDENT_DURATION     = 300
ACCIDENT_ZONE_M       = 10

# --------------------------TOGGLE SWITCHES---------------------------------- #
WEATHER_ON   = False
ACCIDENT_ON  = False
OCCLUSION_ON = False
# ---------------------------------------------------------------------------- #

def rule_based_decision(NS_observe, EW_observe, NS_bus, EW_bus,
                        rb_phase, evals_in_phase):

    if evals_in_phase < MIN_EVALS:
        return rb_phase

    if evals_in_phase >= MAX_EVALS:
        return PHASE_EW_GREEN if rb_phase == PHASE_NS_GREEN else PHASE_NS_GREEN

    NS_score = NS_observe + BUS_BONUS * len(NS_bus)
    EW_score = EW_observe + BUS_BONUS * len(EW_bus) + MAIN_BONUS

    if rb_phase == PHASE_NS_GREEN and EW_score > NS_score:
        return PHASE_EW_GREEN
    if rb_phase == PHASE_EW_GREEN and NS_score > EW_score:
        return PHASE_NS_GREEN

    return rb_phase


# --------------------------Start SUMO--------------------------------------- #
traci.start([SUMO_BINARY, "-c", SUMO_CFG, "--start"])

np.random.seed(42)
base_weather   = 0
weather_start  = 1500
weather_end    = 2000
eval_time      = 10

# --------------------------Controller State---------------------------------- #
rb_phase        = PHASE_EW_GREEN
evals_in_phase  = MIN_EVALS

traci.trafficlight.setPhase(TL_ID, rb_phase)
traci.trafficlight.setPhaseDuration(TL_ID, EVAL_EVERY)

# --------------------------Accident State------------------------------------ #
accident_pending   = False
accident_active    = False
accident_done      = False
victim             = None
accident_stop_time = None
east_lane_length   = None

# --------------------------Summary Table------------------------------------- #
time                      = []
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

# ============================================================================ #
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    t = traci.simulation.getTime()

    # --------------------------Fetch east_in lane length once---------------- #
    if east_lane_length is None:
        try:
            east_lane_length = traci.lane.getLength("east_in_0")
        except Exception:
            east_lane_length = 100.0

    # --------------------------Weather-------------------------------------- #
    if WEATHER_ON:
        if t == weather_start:
            traci.polygon.add(
                "bg_overlay",
                shape=[(-10000, -10000), (10000, -10000), (10000, 10000), (-10000, 10000)],
                color=(0, 0, 255, 100),
                fill=True,
                layer=-1
            )
            base_weather = 1
            print(f"[t={t:.0f}] Weather event started: reduced visibility.")

        if t == weather_end:
            try:
                traci.polygon.remove("bg_overlay")
            except:
                pass
            base_weather = 0
            print(f"[t={t:.0f}] Weather event ended: visibility restored.")

    # --------------------------Accident Logic-------------------------------- #
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

    # --------------------------Eval Step------------------------------------- #
    if t == eval_time:
        eval_time += EVAL_EVERY
        time.append(t)

        # ---- Count vehicles ---- #
        NS_cars  = []; NS_bus = []; NS_truck = []
        EW_cars  = []; EW_bus = []; EW_truck = []

        for i in traci.vehicle.getIDList():
            j = traci.vehicle.getRoadID(i)
            if j in ["north_in", "south_in", "east_in", "west_in"]:
                if any(x in i for x in ["car_s",   "car_n"]):   NS_cars.append(i)
                if any(x in i for x in ["bus_s",   "bus_n"]):   NS_bus.append(i)
                if any(x in i for x in ["truck_s", "truck_n"]): NS_truck.append(i)
                if any(x in i for x in ["car_e",   "car_w"]):   EW_cars.append(i)
                if any(x in i for x in ["bus_e",   "bus_w"]):   EW_bus.append(i)
                if any(x in i for x in ["truck_e", "truck_w"]): EW_truck.append(i)

        # ---- CO2 & idle ---- #
        NS_CO2  = sum(traci.vehicle.getCO2Emission(i) for i in NS_cars + NS_bus + NS_truck)
        EW_CO2  = sum(traci.vehicle.getCO2Emission(i) for i in EW_cars + EW_bus + EW_truck)
        NS_idle = sum(traci.vehicle.getWaitingTime(i) for i in NS_cars + NS_bus + NS_truck)
        EW_idle = sum(traci.vehicle.getWaitingTime(i) for i in EW_cars + EW_bus + EW_truck)

        # ---- Noisy observation ---- #
        NS_true_count = len(NS_cars) + len(NS_bus) + len(NS_truck)
        EW_true_count = len(EW_cars) + len(EW_bus) + len(EW_truck)

        alpha         = 1 / 200
        weather_alpha = 0.3

        p_NS = (weather_alpha ** base_weather) * (1 - max(1, NS_true_count - 1) * alpha)
        p_EW = (weather_alpha ** base_weather) * (1 - max(1, EW_true_count - 1) * alpha)

        # ---- Occlusion ---- #
        occ_NS = 0
        if OCCLUSION_ON:
            if NS_true_count >= 3 * (len(NS_truck) + len(NS_bus)) and (len(NS_truck) + len(NS_bus)) > 0:
                occ_NS = int(np.sum(np.random.choice([0, 1, 2, 3],
                                                     len(NS_truck) + len(NS_bus),
                                                     p=[0.5, 0.3, 0.15, 0.05])))

        occ_EW = 0
        if OCCLUSION_ON:
            if EW_true_count >= 3 * (len(EW_truck) + len(EW_bus)) and (len(EW_truck) + len(EW_bus)) > 0:
                occ_EW = int(np.sum(np.random.choice([0, 1, 2, 3],
                                                     len(EW_truck) + len(EW_bus),
                                                     p=[0.5, 0.3, 0.15, 0.05])))

        NS_observe = max(0, int(np.random.binomial(NS_true_count, p_NS)) - occ_NS)
        EW_observe = max(0, int(np.random.binomial(EW_true_count, p_EW)) - occ_EW)

        # ---- Decision ---- #
        evals_in_phase += 1

        desired_phase = rule_based_decision(
            NS_observe, EW_observe, NS_bus, EW_bus,
            rb_phase, evals_in_phase
        )

        if desired_phase != rb_phase:
            traci.trafficlight.setPhase(TL_ID, desired_phase)
            rb_phase       = desired_phase
            evals_in_phase = 0

            print(f"[t={t:.0f}] Switched to {'EW' if desired_phase == PHASE_EW_GREEN else 'NS'} green.")

        # ---- Logging ---- #
        true_CO2_NS.append(NS_CO2)
        true_CO2_EW.append(EW_CO2)
        true_vehicle_count_NS.append(NS_true_count)
        true_vehicle_count_EW.append(EW_true_count)
        buses_NS.append(len(NS_bus))
        buses_EW.append(len(EW_bus))
        phase.append(traci.trafficlight.getPhase(TL_ID))
        observed_vehicle_count_NS.append(NS_observe)
        observed_vehicle_count_EW.append(EW_observe)
        idle_time_NS.append(NS_idle)
        idle_time_EW.append(EW_idle)

# --------------------------Save Output-------------------------------------- #
try:
    data = pd.DataFrame({
        "ttime"        : time,
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
    data.to_excel("C:/Users/giorg/OneDrive/Laptop Lenovo/BIG STUFF/output_rulebased.xlsx", index=False)
except Exception as e:
    print(f"Error saving output: {e}")

traci.close()
