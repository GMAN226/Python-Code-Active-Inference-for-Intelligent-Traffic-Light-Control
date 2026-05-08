import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from cycler import cycler
import scienceplots   # NEW

# --------------------------Configuration -------------------------- #
RULEBASED_FILE = r"C:/Users/giorg/OneDrive/Laptop Lenovo/BIG STUFF/output_rulebased.xlsx"
DQN_FILE       = r"C:/Users/giorg/OneDrive/Laptop Lenovo/BIG STUFF/output_dqn.xlsx"
AI_FILE        = r"C:/Users/giorg/OneDrive/Laptop Lenovo/BIG STUFF/output_activeInference.xlsx"
SAVE_DIR       = r"C:/Users/giorg/OneDrive/Laptop Lenovo/BIG STUFF/scenario1_AllTogglesOFF"

os.makedirs(SAVE_DIR, exist_ok=True)

# --------------------------Load data -------------------------- #
print("Loading data...")
rb  = pd.read_excel(RULEBASED_FILE)
dqn = pd.read_excel(DQN_FILE)
ai  = pd.read_excel(AI_FILE)

print(f"  Rule-Based:       {len(rb)} steps")
print(f"  DQN:              {len(dqn)} steps")
print(f"  Active Inference: {len(ai)} steps")

# --------------------------Derived metrics -------------------------- #
rb["total_idle"]  = rb["idle_time_NS"]  + rb["idle_time_EW"]
dqn["total_idle"] = dqn["idle_time_NS"] + dqn["idle_time_EW"]
ai["total_idle"]  = ai["idle_time_NS"]  + ai["idle_time_EW"]

rb["total_CO2"]  = rb["North_CO2"]  + rb["East_CO2"]
dqn["total_CO2"] = dqn["North_CO2"] + dqn["East_CO2"]
ai["total_CO2"]  = ai["North_CO2"]  + ai["East_CO2"]

rb["phase_switch"]  = (rb["phase"].diff().fillna(0)  != 0).astype(int)
dqn["phase_switch"] = (dqn["phase"].diff().fillna(0) != 0).astype(int)
ai["phase_switch"]  = (ai["phase"].diff().fillna(0)  != 0).astype(int)

def bus_served(row):
    ns_served = row["bus_count_NS"] > 0 and row["phase"] == 0
    ew_served = row["bus_count_EW"] > 0 and row["phase"] == 2
    if (row["bus_count_NS"] + row["bus_count_EW"]) == 0:
        return np.nan
    return 1.0 if (ns_served or ew_served) else 0.0

rb["bus_served"]  = rb.apply(bus_served,  axis=1)
dqn["bus_served"] = dqn.apply(bus_served, axis=1)
ai["bus_served"]  = ai.apply(bus_served,  axis=1)

# --------------------------SciencePlots Style -------------------------- #
plt.style.use(["science", "ieee"])

# Clean academic color palette
colors = ['#0C5DA5', '#00B945', '#FF9500']  # blue, green, orange

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.prop_cycle": cycler(color=colors),
    "figure.autolayout": True,
})

CONTROLLERS = [
    (rb,  "Rule-Based"),
    (dqn, "DQN"),
    (ai,  "Active Inference"),
]

# --------------------------Plot 1: Idle Time -------------------------- #
print("Plot 1: Idle Time...")
fig, ax = plt.subplots(figsize=(6.5, 3.2))

for (df, label), c in zip(CONTROLLERS, colors):
    ax.plot(df["ttime"], df["total_idle"], label=label, linewidth=1.2)

ax.set_xlabel("Simulation Time (s)")
ax.set_ylabel("Idle Time (s)")
ax.legend()
fig.savefig(os.path.join(SAVE_DIR, "idle_time.pdf"))   # VECTOR OUTPUT
plt.close()

# --------------------------Plot 2: CO2 -------------------------- #
print("Plot 2: CO2...")
fig, ax = plt.subplots(figsize=(6.5, 3.2))

for (df, label), c in zip(CONTROLLERS, colors):
    ax.plot(df["ttime"], df["total_CO2"], label=label, linewidth=1.2)

ax.set_xlabel("Simulation Time (s)")
ax.set_ylabel(r"CO$_2$ Emissions (mg/s)")
ax.legend()
fig.savefig(os.path.join(SAVE_DIR, "co2.pdf"))
plt.close()

# --------------------------Plot 3: Summary Bars -------------------------- #
print("Plot 3: Summary Bars...")
metrics = [
    ("Avg Idle Time (s)", "total_idle", "mean"),
    ("Avg CO$_2$ per Step (mg/s)", "total_CO2", "mean"),
    ("Phase Switches",    "phase_switch", "sum"),
]

fig, axes = plt.subplots(1, 3, figsize=(10, 3))

labels = ["Rule-Based", "DQN", "Active Inf."]
x = np.arange(3)

for ax, (title, col, agg) in zip(axes, metrics):
    vals = [
        rb[col].mean()  if agg == "mean" else rb[col].sum(),
        dqn[col].mean() if agg == "mean" else dqn[col].sum(),
        ai[col].mean()  if agg == "mean" else ai[col].sum(),
    ]
    ax.bar(x, vals, color=colors)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)

fig.savefig(os.path.join(SAVE_DIR, "summary_bars.pdf"))
plt.close()

# --------------------------Plot 4: Bus Service Rate -------------------------- #
print("Plot 4: Bus Service Rate...")
fig, ax = plt.subplots(figsize=(6, 3))

rates = [
    rb["bus_served"].mean() * 100,
    dqn["bus_served"].mean() * 100,
    ai["bus_served"].mean() * 100,
]

ax.bar(labels, rates, color=colors)
ax.set_ylabel("Service Rate (%)")
ax.set_title("Bus Service Rate")

fig.savefig(os.path.join(SAVE_DIR, "bus_service_rate.pdf"))
plt.close()

# --------------------------Plot 5: Cumulative Idle -------------------------- #
print("Plot 5: Cumulative Idle...")
fig, ax = plt.subplots(figsize=(6.5, 3.2))

ax.plot(rb["ttime"],  rb["total_idle"].cumsum(),  label="Rule-Based")
ax.plot(dqn["ttime"], dqn["total_idle"].cumsum(), label="DQN")
ax.plot(ai["ttime"],  ai["total_idle"].cumsum(),  label="Active Inf.")

ax.set_xlabel("Simulation Time (s)")
ax.set_ylabel("Cumulative Idle Time (s)")
ax.legend()

fig.savefig(os.path.join(SAVE_DIR, "cumulative_idle.pdf"))
plt.close()

# --------------------------Plot 6: Phase Switch Frequency -------------------------- #
print("Plot 6: Phase Switch Frequency...")

window = 20  # rolling window size, matching your example

fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
fig.suptitle("PHASE SWITCH FREQUENCY (switches per 20 steps)", fontsize=12)

controller_names = ["Rule-Based", "DQN", "Active Inference"]
dfs = [rb, dqn, ai]

for ax, df, name, color in zip(axes, dfs, controller_names, colors):
    rolling_switches = df["phase_switch"].rolling(window).sum()

    ax.plot(df["ttime"], rolling_switches, color=color, linewidth=1.2)
    ax.fill_between(df["ttime"], rolling_switches, alpha=0.15, color=color)

    ax.set_title(f"{name}  |  switches per {window} steps", fontsize=10)
    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel("Switch Count")
    ax.grid(True)
    ax.set_axisbelow(True)

plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "phase_switch_frequency.pdf"))
plt.close()


print("\nAll thesis‑ready plots saved as PDF.")

# --------------------------Excel Summary Output -------------------------- #
print("Generating Excel summary...")

# Compute metrics
metrics = [
    "Total idle time (s)",
    "Avg idle time / step (s)",
    "Total CO2 (mg)",
    "Avg CO2 / step (mg/s)",
    "Total phase switches",
    "Bus service rate (%)"
]

rb_vals = [
    rb["total_idle"].sum(),
    rb["total_idle"].mean(),
    rb["total_CO2"].sum(),
    rb["total_CO2"].mean(),
    rb["phase_switch"].sum(),
    rb["bus_served"].mean() * 100
]

dqn_vals = [
    dqn["total_idle"].sum(),
    dqn["total_idle"].mean(),
    dqn["total_CO2"].sum(),
    dqn["total_CO2"].mean(),
    dqn["phase_switch"].sum(),
    dqn["bus_served"].mean() * 100
]

ai_vals = [
    ai["total_idle"].sum(),
    ai["total_idle"].mean(),
    ai["total_CO2"].sum(),
    ai["total_CO2"].mean(),
    ai["phase_switch"].sum(),
    ai["bus_served"].mean() * 100
]

# Determine winners
winners = []
for i in range(len(metrics)):
    values = {
        "Rule-Based": rb_vals[i],
        "DQN": dqn_vals[i],
        "Active Inference": ai_vals[i]
    }

    # For bus service rate → higher is better
    if metrics[i] == "Bus service rate (%)":
        winner = max(values, key=values.get)
    else:
        winner = min(values, key=values.get)

    winners.append(winner)

# Create DataFrame
summary_df = pd.DataFrame({
    "Metric": metrics,
    "Rule-Based": rb_vals,
    "DQN": dqn_vals,
    "Active Inference": ai_vals,
    "Winner": winners
})

# Round numeric values for cleaner output
summary_df.iloc[:, 1:4] = summary_df.iloc[:, 1:4].round(2)

# Save to Excel
excel_path = os.path.join(SAVE_DIR, "summary_statistics.xlsx")
summary_df.to_excel(excel_path, index=False)

print(f"Excel summary saved to: {excel_path}")