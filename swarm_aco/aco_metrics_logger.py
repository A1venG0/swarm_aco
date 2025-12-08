#!/usr/bin/env python3
import math
import csv
from typing import Dict, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Float32MultiArray


def euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def avg_pairwise_distance(points: List[Tuple[float, float]]) -> Optional[float]:
    n = len(points)
    if n < 2:
        return None
    s = 0.0
    c = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += euclid(points[i], points[j])
            c += 1
    return s / c if c else None


def entropy_and_concentration(tau_flat: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """
    Entropy over normalized tau (tau>=0). Concentration = 1 - H/log(N).
    If tau sum is 0 -> returns (None, None).
    """
    tau = np.maximum(tau_flat.astype(np.float64), 0.0)
    s = float(tau.sum())
    n = int(tau.size)
    if n <= 1 or s <= 0.0:
        return (None, None)

    p = tau / s
    p = p[p > 0.0]
    if p.size == 0:
        return (None, None)

    H = float(-(p * np.log(p)).sum())
    H_norm = H / math.log(n)
    H_norm = max(0.0, min(1.0, H_norm))
    conc = 1.0 - H_norm
    return (H, conc)


def stats_for_field(
    field: Optional[np.ndarray],
    thr_abs: Optional[float],
    thr_frac_of_max: float,
) -> Dict[str, Optional[float]]:
    """
    Returns mean, max, sum, entropy, concentration, count_above_threshold.
    If field is None -> all None.
    """
    if field is None:
        return {
            "mean": None,
            "max": None,
            "sum": None,
            "entropy": None,
            "concentration": None,
            "n_above": None,
            "n": None,
        }

    flat = field.reshape(-1).astype(np.float64)
    n = int(flat.size)
    mx = float(np.max(flat)) if n else None
    sm = float(np.sum(flat)) if n else None
    mn = float(np.mean(flat)) if n else None

    H, conc = entropy_and_concentration(flat)

    # threshold
    thr = None
    if thr_abs is not None:
        thr = float(thr_abs)
    elif mx is not None:
        thr = float(thr_frac_of_max) * mx

    n_above = None
    if thr is not None and n:
        n_above = int(np.sum(flat >= thr))

    return {
        "mean": mn,
        "max": mx,
        "sum": sm,
        "entropy": H,
        "concentration": conc,
        "n_above": n_above,
        "n": n,
    }


class ACOMetricsLogger(Node):
    def __init__(self):
        super().__init__("aco_metrics_logger")

        # --- experiment labels (so you can compare alpha/beta/rho) ---
        self.declare_parameter("policy", "stigmergic_aco")
        self.declare_parameter("seed", 0)
        self.declare_parameter("alpha", float("nan"))
        self.declare_parameter("beta", float("nan"))
        self.declare_parameter("rho", float("nan"))
        self.declare_parameter("out_csv", "aco_metrics.csv")

        # --- timing ---
        self.declare_parameter("sample_dt", 1.0)
        self.declare_parameter("t_end_sec", 300.0)

        # --- map geometry (must match your decision node) ---
        self.declare_parameter("resolution", 1.0)
        self.declare_parameter("origin_x", -50.0)
        self.declare_parameter("origin_y", -50.0)
        self.declare_parameter("width", 100)
        self.declare_parameter("height", 100)

        # --- pheromone topics (matches your decision node) ---
        self.declare_parameter("pheromone_pos_topic", "/pheromone_map")
        self.declare_parameter("pheromone_neg_topic", "/pheromone_map_neg")

        # Compute stats for: pos, neg, tau=max(0,pos-neg)
        self.declare_parameter("tau_use_combined_pos_minus_neg", True)

        # Thresholding (applies to each field independently)
        # If tau_threshold_abs is NaN, uses tau_threshold_frac_of_max * max(field)
        self.declare_parameter("tau_threshold_abs", float("nan"))
        self.declare_parameter("tau_threshold_frac_of_max", 0.8)

        # --- swarm topics ---
        self.declare_parameter(
            "pose_topics",
            [
                "/drone1/local_position/pose",
                "/drone2/local_position/pose",
                "/drone3/local_position/pose",
            ],
        )
        self.declare_parameter("mode_topic", "/aco_mode")
        self.declare_parameter("detection_mode_name", "CONVERGE")

        # --- read params ---
        self.policy = str(self.get_parameter("policy").value)
        self.seed = int(self.get_parameter("seed").value)
        self.alpha = float(self.get_parameter("alpha").value)
        self.beta = float(self.get_parameter("beta").value)
        self.rho = float(self.get_parameter("rho").value)
        self.out_csv = str(self.get_parameter("out_csv").value)

        self.sample_dt = float(self.get_parameter("sample_dt").value)
        self.t_end = float(self.get_parameter("t_end_sec").value)

        self.res = float(self.get_parameter("resolution").value)
        self.ox = float(self.get_parameter("origin_x").value)
        self.oy = float(self.get_parameter("origin_y").value)
        self.W = int(self.get_parameter("width").value)
        self.H = int(self.get_parameter("height").value)

        self.pos_topic = str(self.get_parameter("pheromone_pos_topic").value)
        self.neg_topic = str(self.get_parameter("pheromone_neg_topic").value)
        self.use_tau = bool(self.get_parameter("tau_use_combined_pos_minus_neg").value)

        thr_abs_raw = float(self.get_parameter("tau_threshold_abs").value)
        self.thr_abs = None if math.isnan(thr_abs_raw) else float(thr_abs_raw)
        self.thr_frac = float(self.get_parameter("tau_threshold_frac_of_max").value)

        self.pose_topics = list(self.get_parameter("pose_topics").value)
        self.mode_topic = str(self.get_parameter("mode_topic").value)
        self.detection_mode_name = str(self.get_parameter("detection_mode_name").value).strip().upper()

        # --- state ---
        self.t0 = self.get_clock().now()
        self.last_sample_t = 0.0

        # Mode accounting
        self.mode = "EXPLORE"
        self.mode_switches = 0
        self.mode_start_t = 0.0
        self.mode_durations: Dict[str, float] = {}
        self.t_first_detect: Optional[float] = None

        # Swarm state
        self.pose_latest: Dict[str, Optional[Tuple[float, float]]] = {t: None for t in self.pose_topics}
        self.pose_prev: Dict[str, Optional[Tuple[float, float]]] = {t: None for t in self.pose_topics}
        self.path_len: Dict[str, float] = {t: 0.0 for t in self.pose_topics}

        # Pheromone maps
        self.map_pos: Optional[np.ndarray] = None
        self.map_neg: Optional[np.ndarray] = None
        self.map_tau: Optional[np.ndarray] = None
        self.map_stamp_sec: Optional[float] = None

        # --- ROS I/O ---
        # pheromone
        self.create_subscription(Float32MultiArray, self.pos_topic, self.pos_cb, 10)
        self.create_subscription(Float32MultiArray, self.neg_topic, self.neg_cb, 10)

        # poses
        for topic in self.pose_topics:
            self.create_subscription(
                PoseStamped,
                topic,
                lambda msg, tp=topic: self.pose_cb(msg, tp),
                qos_profile_sensor_data,
            )

        # mode
        self.create_subscription(String, self.mode_topic, self.mode_cb, 10)

        # timer
        self.timer = self.create_timer(0.1, self.tick)

        # csv header flag
        self._wrote_header = False

        self.get_logger().info(
            f"ACO Metrics Logger: pos={self.pos_topic}, neg={self.neg_topic}, "
            f"tau={'pos-neg' if self.use_tau else 'disabled'}, map={self.W}x{self.H}@{self.res}m"
        )

    def elapsed(self) -> float:
        return (self.get_clock().now() - self.t0).nanoseconds / 1e9

    def cell_to_world(self, cx: int, cy: int) -> Tuple[float, float]:
        x = self.ox + (cx + 0.5) * self.res
        y = self.oy + (cy + 0.5) * self.res
        return (x, y)

    # ---------- callbacks ----------
    def mode_cb(self, msg: String):
        new_mode = msg.data.strip().upper()
        if not new_mode:
            return

        t = self.elapsed()
        if new_mode != self.mode:
            # accumulate previous mode duration
            dt = max(0.0, t - self.mode_start_t)
            self.mode_durations[self.mode] = self.mode_durations.get(self.mode, 0.0) + dt

            self.mode = new_mode
            self.mode_switches += 1
            self.mode_start_t = t

            if self.t_first_detect is None and self.mode == self.detection_mode_name:
                self.t_first_detect = t

    def pose_cb(self, msg: PoseStamped, topic: str):
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        self.pose_latest[topic] = (x, y)

        prev = self.pose_prev.get(topic)
        if prev is not None:
            self.path_len[topic] += euclid(prev, (x, y))
        self.pose_prev[topic] = (x, y)

    def _reshape_map(self, arr: np.ndarray) -> Optional[np.ndarray]:
        expected = self.W * self.H
        if arr.size != expected:
            self.get_logger().warn(f"Pheromone map size mismatch ({arr.size} != {expected})")
            return None
        return arr.reshape((self.H, self.W))

    def pos_cb(self, msg: Float32MultiArray):
        arr = np.array(msg.data, dtype=np.float32)
        m = self._reshape_map(arr)
        if m is None:
            return
        self.map_pos = m
        self._update_tau()
        self.map_stamp_sec = self.elapsed()

    def neg_cb(self, msg: Float32MultiArray):
        arr = np.array(msg.data, dtype=np.float32)
        m = self._reshape_map(arr)
        if m is None:
            return
        self.map_neg = m
        self._update_tau()
        self.map_stamp_sec = self.elapsed()

    def _update_tau(self):
        if not self.use_tau:
            self.map_tau = None
            return
        if self.map_pos is None or self.map_neg is None:
            self.map_tau = None
            return
        self.map_tau = np.maximum(self.map_pos - self.map_neg, 0.0)

    # ---------- derived metrics ----------
    def _mode_fractions(self, t_now: float) -> Dict[str, float]:
        durations = dict(self.mode_durations)
        # include ongoing current mode
        durations[self.mode] = durations.get(self.mode, 0.0) + max(0.0, t_now - self.mode_start_t)
        total = sum(durations.values())
        if total <= 0.0:
            return {}
        return {k: v / total for k, v in durations.items()}

    def _peak_info(self, field: Optional[np.ndarray]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Returns (peak_x, peak_y, peak_val) for a map."""
        if field is None:
            return (None, None, None)
        idx = int(np.argmax(field))
        cy, cx = divmod(idx, self.W)  # since row-major flatten
        peak_val = float(field[cy, cx])
        px, py = self.cell_to_world(cx, cy)
        return (px, py, peak_val)

    # ---------- csv ----------
    def write_row(self, t: float):
        # swarm metrics
        points = [p for p in self.pose_latest.values() if p is not None]
        avg_pw = avg_pairwise_distance(points)
        path_total = sum(self.path_len.values())

        # mode metrics
        frac = self._mode_fractions(t)
        frac_explore = frac.get("EXPLORE", 0.0)
        frac_converge = frac.get("CONVERGE", 0.0)
        switches_per_min = (self.mode_switches / (t / 60.0)) if t > 1e-6 else 0.0

        # pheromone stats
        pos_s = stats_for_field(self.map_pos, self.thr_abs, self.thr_frac)
        neg_s = stats_for_field(self.map_neg, self.thr_abs, self.thr_frac)
        tau_s = stats_for_field(self.map_tau, self.thr_abs, self.thr_frac) if self.use_tau else {
            "mean": None, "max": None, "sum": None, "entropy": None, "concentration": None, "n_above": None, "n": None
        }

        # peaks (these are super diagnostic for ACO)
        pos_px, pos_py, pos_peak = self._peak_info(self.map_pos)
        tau_px, tau_py, tau_peak = self._peak_info(self.map_tau) if self.use_tau else (None, None, None)

        write_header = not self._wrote_header
        with open(self.out_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow([
                    # identifiers
                    "policy", "seed", "alpha", "beta", "rho",
                    # time+mode
                    "t_sec", "mode",
                    "mode_switches", "mode_switches_per_min",
                    "t_first_detect",
                    "frac_time_explore", "frac_time_converge",
                    # swarm behavior
                    "avg_pairwise_dist",
                    "path_len_total",
                    *[f"path_len_{i+1}" for i in range(len(self.pose_topics))],
                    # pheromone meta
                    "map_stamp_sec",
                    # pos stats
                    "pos_n", "pos_mean", "pos_max", "pos_sum", "pos_entropy", "pos_concentration", "pos_n_above",
                    "pos_peak_x", "pos_peak_y", "pos_peak_val",
                    # neg stats
                    "neg_n", "neg_mean", "neg_max", "neg_sum", "neg_entropy", "neg_concentration", "neg_n_above",
                    # tau stats (pos-neg clipped)
                    "tau_n", "tau_mean", "tau_max", "tau_sum", "tau_entropy", "tau_concentration", "tau_n_above",
                    "tau_peak_x", "tau_peak_y", "tau_peak_val",
                    # threshold config
                    "thr_abs", "thr_frac_of_max",
                ])
                self._wrote_header = True

            def fmt(x: Optional[float], nd=6) -> str:
                return "" if x is None else f"{x:.{nd}f}"

            w.writerow([
                self.policy, self.seed,
                f"{self.alpha:.6g}", f"{self.beta:.6g}", f"{self.rho:.6g}",
                f"{t:.3f}", self.mode,
                self.mode_switches, f"{switches_per_min:.6f}",
                "" if self.t_first_detect is None else f"{self.t_first_detect:.3f}",
                f"{frac_explore:.6f}", f"{frac_converge:.6f}",
                "" if avg_pw is None else f"{avg_pw:.6f}",
                f"{path_total:.6f}",
                *[f"{self.path_len[self.pose_topics[i]]:.6f}" for i in range(len(self.pose_topics))],
                "" if self.map_stamp_sec is None else f"{self.map_stamp_sec:.3f}",

                int(pos_s["n"]) if pos_s["n"] is not None else "",
                fmt(pos_s["mean"]), fmt(pos_s["max"]), fmt(pos_s["sum"]),
                fmt(pos_s["entropy"]), fmt(pos_s["concentration"]),
                "" if pos_s["n_above"] is None else int(pos_s["n_above"]),
                fmt(pos_px), fmt(pos_py), fmt(pos_peak),

                int(neg_s["n"]) if neg_s["n"] is not None else "",
                fmt(neg_s["mean"]), fmt(neg_s["max"]), fmt(neg_s["sum"]),
                fmt(neg_s["entropy"]), fmt(neg_s["concentration"]),
                "" if neg_s["n_above"] is None else int(neg_s["n_above"]),

                int(tau_s["n"]) if tau_s["n"] is not None else "",
                fmt(tau_s["mean"]), fmt(tau_s["max"]), fmt(tau_s["sum"]),
                fmt(tau_s["entropy"]), fmt(tau_s["concentration"]),
                "" if tau_s["n_above"] is None else int(tau_s["n_above"]),
                fmt(tau_px), fmt(tau_py), fmt(tau_peak),

                "" if self.thr_abs is None else f"{self.thr_abs:.6f}",
                f"{self.thr_frac:.6f}",
            ])

    def finalize(self):
        t = self.elapsed()
        self.write_row(t)
        self.get_logger().info(f"Wrote ACO metrics to {self.out_csv} (policy={self.policy}, seed={self.seed})")
        rclpy.shutdown()

    def tick(self):
        t = self.elapsed()
        do_sample = (t - self.last_sample_t) >= self.sample_dt
        do_end = t >= self.t_end

        if do_sample:
            self.last_sample_t = t
            self.write_row(t)

        if do_end:
            self.finalize()


def main(args=None):
    rclpy.init(args=args)
    node = ACOMetricsLogger()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
