#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import numpy as np
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PointStamped, PoseStamped
import math
import random

class ACODecisionNode(Node):
    def __init__(self):
        super().__init__('aco_decision_node')

        # --- Parameters ---
        # Converge (legacy)
        self.declare_parameter('alpha', 1.0)
        self.declare_parameter('beta',  2.0)

        # Explore (novelty-biased)
        self.declare_parameter('explore_alpha', 0.7)          # weight on (low) positive pheromone
        self.declare_parameter('explore_beta',  1.2)          # heuristic (distance) weight
        self.declare_parameter('explore_gamma', 0.5)          # weight on negative/visited suppression
        self.declare_parameter('explore_epsilon', 0.10)       # epsilon-greedy in EXPLORE
        self.declare_parameter('explore_tau_clip', 10.0)      # cap pos pheromone in EXPLORE
        self.declare_parameter('explore_mask_radius', 6.0)    # m; suppress near last peak
        self.declare_parameter('explore_step', 6.0)           # m; projected step from pose
        self.declare_parameter('explore_block_radius_cells', 2)  # skip candidates too near pose cell
        self.declare_parameter('min_wp_separation', 1.5)      # m; avoid re-issuing near-identical WPs
        self.declare_parameter('explore_neg_amount', 1.0)     # small negative "visit" per EXPLORE tick

        # Map geometry
        self.declare_parameter('resolution', 1.0)
        self.declare_parameter('origin_x', -50.0)
        self.declare_parameter('origin_y', -50.0)
        self.declare_parameter('width', 100)
        self.declare_parameter('height', 100)
        self.declare_parameter('decision_rate', 1.0)

        # Waypoint set (discrete candidates)
        self.declare_parameter(
            'waypoints',
            [0.0, 0.0],
            ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY)
        )

        # Hotspot logic
        self.declare_parameter('hotspot_on_threshold', 50.0)
        self.declare_parameter('hotspot_off_threshold', 10.0)
        self.declare_parameter('hotspot_detect_cycles', 3)    # hysteresis to enter CONVERGE
        self.declare_parameter('visit_neg_amount', 100.0)     # used to "cool" peak (3x3) at dwell
        self.declare_parameter('mode_cooldown_secs', 10.0)
        self.declare_parameter('hotspot_step', 8.0)
        self.declare_parameter('hotspot_direct', False)
        self.declare_parameter('converge_z', 2.0)
        self.declare_parameter('explore_z', 2.0)
        self.declare_parameter('close_radius', 2.0)
        self.declare_parameter('dwell_cycles', 5)

        # --- Read params ---
        self.alpha_converge = float(self.get_parameter('alpha').value)
        self.beta_converge  = float(self.get_parameter('beta').value)

        self.alpha_explore  = float(self.get_parameter('explore_alpha').value)
        self.beta_explore   = float(self.get_parameter('explore_beta').value)
        self.gamma_explore  = float(self.get_parameter('explore_gamma').value)
        self.eps_explore    = float(self.get_parameter('explore_epsilon').value)
        self.tau_clip_explore = float(self.get_parameter('explore_tau_clip').value)
        self.mask_radius_explore = float(self.get_parameter('explore_mask_radius').value)
        self.explore_step   = float(self.get_parameter('explore_step').value)
        self.block_r_cells  = int(self.get_parameter('explore_block_radius_cells').value)
        self.min_wp_sep     = float(self.get_parameter('min_wp_separation').value)
        self.explore_neg_amount = float(self.get_parameter('explore_neg_amount').value)

        self.resolution = float(self.get_parameter('resolution').value)
        self.origin_x = float(self.get_parameter('origin_x').value)
        self.origin_y = float(self.get_parameter('origin_y').value)
        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        self.decision_rate = float(self.get_parameter('decision_rate').value)

        self.hotspot_on = float(self.get_parameter('hotspot_on_threshold').value)
        self.hotspot_off = float(self.get_parameter('hotspot_off_threshold').value)
        self.hotspot_detect_cycles = int(self.get_parameter('hotspot_detect_cycles').value)
        self.hotspot_step = float(self.get_parameter('hotspot_step').value)
        self.hotspot_direct = bool(self.get_parameter('hotspot_direct').value)
        self.converge_z = float(self.get_parameter('converge_z').value)
        self.explore_z  = float(self.get_parameter('explore_z').value)
        self.close_radius = float(self.get_parameter('close_radius').value)
        self.dwell_cycles = int(self.get_parameter('dwell_cycles').value)
        self.visit_neg_amount = float(self.get_parameter('visit_neg_amount').value)
        self.mode_cooldown_secs = float(self.get_parameter('mode_cooldown_secs').value)

        # --- Internal state ---
        self.map_pos = None
        self.map_neg = None
        self.mode = 'EXPLORE'
        self.current_idx = 0
        self.peak_xy = (0.0, 0.0)
        self.converge_dwell = 0
        self.current_pose = None
        self.last_converge_exit = None
        self.detect_counter = 0
        self.last_wp_sent = None  # (x, y) of last published waypoint
        self.eps = 1e-6

        # Parse waypoints
        raw = self.get_parameter('waypoints').value
        self.waypoints = []
        if isinstance(raw, list) and len(raw) >= 2 and len(raw) % 2 == 0:
            for i in range(0, len(raw), 2):
                self.waypoints.append({'x': float(raw[i]), 'y': float(raw[i+1])})
        else:
            self.get_logger().warn("No valid waypoints provided; using [0,0].")
            self.waypoints = [{'x': 0.0, 'y': 0.0}]

        # --- ROS 2 I/O ---
        self.map_sub = self.create_subscription(
            Float32MultiArray, '/pheromone_map', self.map_callback, 10
        )
        self.map_neg_sub = self.create_subscription(
            Float32MultiArray, '/pheromone_map_neg', self.map_neg_callback, 10
        )
        self.pose_sub = self.create_subscription(
            PoseStamped, '/local_position/pose', self.pose_callback, qos_profile_sensor_data
        )

        self.pub_next_wp = self.create_publisher(PointStamped, '/aco_next_waypoint', 10)
        self.neg_pub = self.create_publisher(PointStamped, '/pheromone_deposit_neg', 10)
        self.mode_pub = self.create_publisher(String, '/aco_mode', 10)

        # Timer
        self.timer = self.create_timer(max(0.01, 1.0 / self.decision_rate), self.main_loop)

        self.get_logger().info(
            f"ACO Decision node started: "
            f"explore(alpha={self.alpha_explore:.2f}, beta={self.beta_explore:.2f}, eps={self.eps_explore:.2f}), "
            f"converge(alpha={self.alpha_converge:.2f}, beta={self.beta_converge:.2f}), "
            f"rate={self.decision_rate:.2f} Hz, waypoints={len(self.waypoints)}, "
            f"map={self.width}x{self.height}@{self.resolution}m"
        )

    # ---------------- Callbacks ----------------
    def map_callback(self, msg: Float32MultiArray):
        arr = np.array(msg.data, dtype=np.float32)
        expected = self.width * self.height
        if arr.size != expected:
            self.get_logger().warn(f"Pheromone map size mismatch ({arr.size} != {expected})")
            return
        self.map_pos = arr.reshape((self.height, self.width))

    def map_neg_callback(self, msg: Float32MultiArray):
        arr = np.array(msg.data, dtype=np.float32)
        expected = self.width * self.height
        if arr.size != expected:
            self.get_logger().debug(f"Neg map size mismatch ({arr.size} != {expected})")
            return
        self.map_neg = arr.reshape((self.height, self.width))

    def pose_callback(self, msg: PoseStamped):
        self.current_pose = (float(msg.pose.position.x), float(msg.pose.position.y))

    def publish_mode(self):
        m = String()
        m.data = self.mode
        self.mode_pub.publish(m)

    # --------------- Utilities -----------------
    def cell_to_world(self, cx, cy):
        x = self.origin_x + (cx + 0.5) * self.resolution
        y = self.origin_y + (cy + 0.5) * self.resolution
        return x, y

    def world_to_cell(self, x, y):
        cx = int((x - self.origin_x) / self.resolution)
        cy = int((y - self.origin_y) / self.resolution)
        return cx, cy

    def map_value_pos(self, x, y):
        cx, cy = self.world_to_cell(x, y)
        if self.map_pos is None:
            return 0.0
        h, w = self.map_pos.shape
        if 0 <= cx < w and 0 <= cy < h:
            return float(self.map_pos[cy, cx])
        return 0.0

    def map_value_neg(self, x, y):
        cx, cy = self.world_to_cell(x, y)
        if self.map_neg is None:
            return 0.0
        h, w = self.map_neg.shape
        if 0 <= cx < w and 0 <= cy < h:
            return float(self.map_neg[cy, cx])
        return 0.0

    def world_to_cell_value(self, x, y):
        """Return combined pheromone tau = max(0, pos - neg)."""
        val_pos = self.map_value_pos(x, y)
        val_neg = self.map_value_neg(x, y)
        return max(0.0, val_pos - val_neg)

    def find_hotspot_peak(self):
        """Return (peak_x, peak_y, peak_val) of the POSITIVE map."""
        if self.map_pos is None:
            return None
        cy, cx = np.unravel_index(np.argmax(self.map_pos), self.map_pos.shape)
        peak_val = float(self.map_pos[cy, cx])
        px, py = self.cell_to_world(cx, cy)
        return (px, py, peak_val)

    def tau_for_explore(self, x, y):
        """Mode-aware pheromone for EXPLORE: clip and optionally mask around last peak."""
        tau = self.world_to_cell_value(x, y)
        if self.tau_clip_explore > 0.0:
            tau = min(tau, self.tau_clip_explore)
        if self.mask_radius_explore > 0.0 and self.peak_xy is not None:
            dx = x - self.peak_xy[0]
            dy = y - self.peak_xy[1]
            if dx*dx + dy*dy <= self.mask_radius_explore * self.mask_radius_explore:
                tau = 0.0
        return tau

    def candidate_ok_for_pose(self, wp):
        """Reject candidates that are too close to the current pose's cell (prevents self-gluing)."""
        if self.current_pose is None:
            return True
        px, py = self.current_pose
        cpx, cpy = self.world_to_cell(px, py)
        cwx, cwy = self.world_to_cell(wp['x'], wp['y'])
        return abs(cpx - cwx) > self.block_r_cells or abs(cpy - cwy) > self.block_r_cells

    def far_enough_from_last_wp(self, x, y):
        if self.last_wp_sent is None:
            return True
        lx, ly = self.last_wp_sent
        return math.hypot(x - lx, y - ly) >= self.min_wp_sep

    def choose_next(self, current_idx: int, mode: str) -> int:
        # Epsilon-greedy random hop in EXPLORE
        if mode == 'EXPLORE' and random.random() < max(0.0, min(1.0, self.eps_explore)):
            candidates = [i for i in range(len(self.waypoints)) if i != current_idx and
                          (self.candidate_ok_for_pose(self.waypoints[i]) and
                           self.far_enough_from_last_wp(self.waypoints[i]['x'], self.waypoints[i]['y']))]
            if candidates:
                return random.choice(candidates)

        current = self.waypoints[current_idx]
        scores = []
        for j, wp in enumerate(self.waypoints):
            if j == current_idx:
                scores.append(0.0)
                continue

            # basic filters to avoid re-issuing near-identical goals
            if not self.candidate_ok_for_pose(wp):
                scores.append(0.0)
                continue
            if not self.far_enough_from_last_wp(wp['x'], wp['y']):
                scores.append(0.0)
                continue

            if mode == 'EXPLORE':
                # Novelty-seeking: prefer low positive pheromone + low "visited"/negative
                tau = self.tau_for_explore(wp['x'], wp['y']) + self.eps
                neg = self.map_value_neg(wp['x'], wp['y'])
                a = self.alpha_explore
                b = self.beta_explore
                g = self.gamma_explore

                # distance heuristic (mild)
                dist = math.hypot(wp['x'] - current['x'], wp['y'] - current['y']) + self.eps
                eta = 1.0 / dist

                # (tau)^(-a) * (eta)^(b) * (1+neg)^(-g)
                score = (tau ** (-a)) * (eta ** b) * ((1.0 + neg) ** (-g))

            else:
                # Converge: your original high-pheromone preference
                tau = self.world_to_cell_value(wp['x'], wp['y']) + self.eps
                a = self.alpha_converge
                b = self.beta_converge
                dist = math.hypot(wp['x'] - current['x'], wp['y'] - current['y']) + self.eps
                eta = 1.0 / dist
                score = (tau ** a) * (eta ** b)

            scores.append(score)

        total = sum(scores)
        if total <= 0.0:
            candidates = [i for i in range(len(self.waypoints)) if i != current_idx and
                          (self.candidate_ok_for_pose(self.waypoints[i]) and
                           self.far_enough_from_last_wp(self.waypoints[i]['x'], self.waypoints[i]['y']))]
            if not candidates:
                candidates = [i for i in range(len(self.waypoints)) if i != current_idx]
            return random.choice(candidates)
        probs = [v / total for v in scores]
        return int(np.random.choice(range(len(probs)), p=probs))

    # --------------- Main loop -----------------
    def main_loop(self):
        if self.map_pos is None:
            return

        # --- Hotspot hysteresis + cooldown ---
        peak = self.find_hotspot_peak()
        now = self.get_clock().now().nanoseconds / 1e9
        in_cooldown = False
        if self.last_converge_exit is not None:
            in_cooldown = (now - self.last_converge_exit) < self.mode_cooldown_secs

        if peak is not None:
            px, py, pval = peak

            if self.mode == 'EXPLORE':
                if pval >= self.hotspot_on and not in_cooldown:
                    self.detect_counter += 1
                else:
                    self.detect_counter = 0

                if self.detect_counter >= self.hotspot_detect_cycles:
                    self.mode = 'CONVERGE'
                    self.peak_xy = (px, py)
                    self.converge_dwell = 0
                    self.detect_counter = 0
                    self.publish_mode()
                    self.get_logger().info(f"Hotspot detected (stable {self.hotspot_detect_cycles}): {pval:.1f} → CONVERGE")

            elif self.mode == 'CONVERGE':
                if pval <= self.hotspot_off:
                    self.mode = 'EXPLORE'
                    self.last_converge_exit = now
                    self.converge_dwell = 0
                    self.publish_mode()
                    self.get_logger().info(f"Hotspot faded → EXPLORE (cooldown {self.mode_cooldown_secs:.0f}s)")

        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"

        if self.mode == 'CONVERGE':
            # Follow the latest peak
            latest = self.find_hotspot_peak()
            if latest is not None:
                self.peak_xy = (latest[0], latest[1])
            target_x, target_y = self.peak_xy

            # Step from actual pose if available
            if not self.hotspot_direct and self.current_pose is not None:
                px, py = self.current_pose
                dx = target_x - px
                dy = target_y - py
                dist = math.hypot(dx, dy)
                if dist > 1e-3:
                    scale = min(1.0, self.hotspot_step / dist)
                    target_x = px + dx * scale
                    target_y = py + dy * scale

            msg.point.x = float(target_x)
            msg.point.y = float(target_y)
            msg.point.z = float(self.converge_z)
            self.pub_next_wp.publish(msg)
            self.last_wp_sent = (msg.point.x, msg.point.y)
            self.get_logger().info(
                f"[CONVERGE] → ({target_x:.1f}, {target_y:.1f}) peak=({self.peak_xy[0]:.1f},{self.peak_xy[1]:.1f})"
            )

            # dwell detection using actual pose
            d = float('inf')
            if self.current_pose is not None:
                px, py = self.current_pose
                d = math.hypot(self.peak_xy[0] - px, self.peak_xy[1] - py)

            if d <= self.close_radius:
                self.converge_dwell += 1
            else:
                self.converge_dwell = 0

            if self.converge_dwell >= self.dwell_cycles:
                # cool locally with a 3x3 negative deposit
                cx, cy = self.world_to_cell(*self.peak_xy)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nx, ny = cx + dx, cy + dy
                        wx, wy = self.cell_to_world(nx, ny)
                        m = PointStamped()
                        m.header.stamp = msg.header.stamp
                        m.header.frame_id = "world"
                        m.point.x = float(wx)
                        m.point.y = float(wy)
                        m.point.z = float(self.visit_neg_amount)
                        self.neg_pub.publish(m)
                self.get_logger().info(
                    f"[CONVERGE] cooled peak at ({self.peak_xy[0]:.1f},{self.peak_xy[1]:.1f}); → EXPLORE"
                )
                self.mode = 'EXPLORE'
                self.converge_dwell = 0
                self.publish_mode()
            return

        # ----------------- EXPLORE -----------------
        # Small negative "visit" deposit to discourage re-sampling
        if self.current_pose is not None and self.explore_neg_amount > 0.0:
            vx, vy = self.current_pose
            vmsg = PointStamped()
            vmsg.header.stamp = msg.header.stamp
            vmsg.header.frame_id = "world"
            vmsg.point.x = float(vx)
            vmsg.point.y = float(vy)
            vmsg.point.z = float(self.explore_neg_amount)
            self.neg_pub.publish(vmsg)

        next_idx = self.choose_next(self.current_idx, mode='EXPLORE')
        target = self.waypoints[next_idx]

        # Project a fixed step from current pose toward the chosen waypoint
        tx, ty = target['x'], target['y']
        if self.current_pose is not None:
            px, py = self.current_pose
            dx = tx - px
            dy = ty - py
            dist = math.hypot(dx, dy)
            if dist > 1e-3:
                step = self.explore_step
                scale = min(1.0, step / dist)
                tx = px + dx * scale
                ty = py + dy * scale

        # If projected point is still too close to last sent waypoint, jitter a bit
        if not self.far_enough_from_last_wp(tx, ty):
            ang = random.uniform(0.0, 2.0 * math.pi)
            tx += self.min_wp_sep * math.cos(ang)
            ty += self.min_wp_sep * math.sin(ang)

        msg.point.x = float(tx)
        msg.point.y = float(ty)
        msg.point.z = float(self.explore_z)
        self.pub_next_wp.publish(msg)
        self.last_wp_sent = (msg.point.x, msg.point.y)
        self.get_logger().info(
            f"[EXPLORE] From {self.current_idx} → {next_idx} cand=({self.waypoints[next_idx]['x']:.1f},{self.waypoints[next_idx]['y']:.1f}) "
            f"→ step_to=({tx:.1f}, {ty:.1f})"
        )
        self.current_idx = next_idx

def main(args=None):
    rclpy.init(args=args)
    node = ACODecisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
