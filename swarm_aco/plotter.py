import pandas as pd
import matplotlib.pyplot as plt

# If you have it in a file:
df = pd.read_csv("/tmp/aco_metrics.csv")

# If you pasted it into a string, replace this with io.StringIO(...)
# import io
# df = pd.read_csv(io.StringIO(text))

# Clean up: empty strings -> NaN
df = df.replace(r'^\s*$', pd.NA, regex=True)

# Ensure numeric columns are numeric where possible
for c in df.columns:
    if c not in ["policy", "mode"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.sort_values("t_sec").reset_index(drop=True)

# Detect mode switches
df["mode_prev"] = df["mode"].shift(1)
switches = df[(df["mode_prev"].notna()) & (df["mode"] != df["mode_prev"])][["t_sec","mode_prev","mode","mode_switches"]]

print("Detected switches:")
print(switches.to_string(index=False))

# --- Plot: mode over time (as 0/1) ---
mode_map = {"EXPLORE": 0, "CONVERGE": 1}
df["mode01"] = df["mode"].map(mode_map)

plt.figure()
plt.plot(df["t_sec"], df["mode01"])
plt.yticks([0,1], ["EXPLORE","CONVERGE"])
plt.xlabel("t_sec")
plt.title("Mode over time")
plt.show()

# --- Plot: peak coordinates over time (pos + tau) ---
plt.figure()
plt.plot(df["t_sec"], df["pos_peak_x"], label="pos_peak_x")
plt.plot(df["t_sec"], df["pos_peak_y"], label="pos_peak_y")
plt.plot(df["t_sec"], df["tau_peak_x"], label="tau_peak_x")
plt.plot(df["t_sec"], df["tau_peak_y"], label="tau_peak_y")
plt.xlabel("t_sec")
plt.title("Peak coordinates over time")
plt.legend()
plt.show()

# --- Plot: entropy & concentration ---
plt.figure()
plt.plot(df["t_sec"], df["pos_entropy"], label="pos_entropy")
plt.plot(df["t_sec"], df["neg_entropy"], label="neg_entropy")
plt.plot(df["t_sec"], df["tau_entropy"], label="tau_entropy")
plt.xlabel("t_sec")
plt.title("Entropy over time")
plt.legend()
plt.show()

plt.figure()
plt.plot(df["t_sec"], df["pos_concentration"], label="pos_concentration")
plt.plot(df["t_sec"], df["neg_concentration"], label="neg_concentration")
plt.plot(df["t_sec"], df["tau_concentration"], label="tau_concentration")
plt.xlabel("t_sec")
plt.title("Concentration over time")
plt.legend()
plt.show()
