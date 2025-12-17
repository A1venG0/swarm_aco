import pandas as pd
import matplotlib.pyplot as plt

base = "/home/artem/maga/"

files = {
    base + "aco_metrics_alpha_two.csv": "rho=0.05",
    base + "aco_metrics_rho_point_nine.csv":  "rho=0.9",
}

dfs = []

for fname, label in files.items():
    print(f"Loading {fname} as {label}")
    df = pd.read_csv(fname)

    df = df.replace(r'^\s*$', pd.NA, regex=True)

    for c in df.columns:
        if c not in ["policy", "mode"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # sort by time
    df = df.sort_values("t_sec").reset_index(drop=True)

    # tag with rho
    df["rho_label"] = label

    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

mode_map = {"EXPLORE": 0, "CONVERGE": 1}
df_all["mode01"] = df_all["mode"].map(mode_map)

for label, sub in df_all.groupby("rho_label"):
    sub = sub.copy()
    sub["mode_prev"] = sub["mode"].shift(1)
    switches = sub[(sub["mode_prev"].notna()) & (sub["mode"] != sub["mode_prev"])]
    print(f"\nDetected switches for {label}:")
    if not switches.empty:
        print(switches[["t_sec", "mode_prev", "mode", "mode_switches"]].to_string(index=False))
    else:
        print("  (no switches detected)")


def plot_metric(metric: str, title: str = None, ylabel: str = None):
    if metric not in df_all.columns:
        print(f"Skip metric '{metric}' – not found in df.")
        return

    plt.figure()
    for label, g in df_all.groupby("rho_label"):
        plt.plot(g["t_sec"], g[metric], label=label)
    plt.xlabel("t_sec")
    if ylabel:
        plt.ylabel(ylabel)
    plt.title(title or metric)
    plt.legend()
    plt.grid(True)
    plt.show()


plot_metric(
    "mode01",
    "Mode over time (0 = EXPLORE, 1 = CONVERGE)",
    "mode (0/1)"
)

plot_metric("mode_switches_per_min", "Mode switches per minute")

plot_metric("frac_time_explore",  "Fraction of time in EXPLORE")
plot_metric("frac_time_converge", "Fraction of time in CONVERGE")


plot_metric("avg_pairwise_dist",
            "Average pairwise distance between drones",
            "distance")

plot_metric("path_len_total",
            "Total path length of swarm",
            "length")

if "t_first_detect" in df_all.columns:
    print("\n t_first_detect per rho:")
    for label, g in df_all.groupby("rho_label"):
        vals = g["t_first_detect"].dropna().unique()
        print(f"  {label}: {vals}")

for m in ["pos_mean", "neg_mean", "tau_mean"]:
    if m in df_all.columns:
        plot_metric(m, f"{m} over time")

for m in ["pos_max", "neg_max", "tau_max"]:
    if m in df_all.columns:
        plot_metric(m, f"{m} over time")

for m in ["pos_entropy", "neg_entropy", "tau_entropy"]:
    if m in df_all.columns:
        plot_metric(m, f"{m} over time")

for m in ["pos_concentration", "neg_concentration", "tau_concentration"]:
    if m in df_all.columns:
        plot_metric(m, f"{m} over time")

for m in ["pos_peak_x", "pos_peak_y", "tau_peak_x", "tau_peak_y"]:
    if m in df_all.columns:
        plot_metric(m, f"{m} over time")
