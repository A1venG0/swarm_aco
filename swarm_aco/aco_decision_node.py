#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import numpy as np
from std_msgs.msg import Float32MultiArray, String
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PointStamped, PoseStamped
import math
import random


class ACODecisionNode(Node):
    def __init__(self):
        super().__init__('aco_decision_node')

        # Converge params
        self.declare_parameter('alpha', 1.0)
        self.declare_parameter('beta',  2.0)

        # Explore params
        self.declare_parameter('explore_alpha', 0.5)
        self.declare_parameter('explore_beta',  0.5)
        self.declare_parameter('explore_gamma', 0.5)
        self.declare_parameter('explore_epsilon', 0.10)
        self.declare_parameter('explore_tau_clip', 10.0)
        self.declare_parameter('explore_mask_radius', 6.0)
        self.declare_parameter('explore_step', 6.0)
        self.declare_parameter('explore_block_radius_cells', 2)
        self.declare_parameter('min_wp_separation', 1.5)
        self.declare_parameter('explore_neg_amount', 1.0)

        self.declare_parameter('explore_min_eta_dist', 10.0)
        self.declare_parameter('explore_min_target_sep', 1.0)
        self.declare_parameter('explore_neg_block_threshold', -1.0)  # <0 disables hard blocking

        self.declare_parameter('converge_min_eta_dist', 1.0)

        self.declare_parameter('resolution', 1.0)
        self.declare_parameter('origin_x', -50.0)
        self.declare_parameter('origin_y', -50.0)
        self.declare_parameter('width', 100)
        self.declare_parameter('height', 100)
        self.declare_parameter('decision_rate', 1.0)

        self.declare_parameter(
            'waypoints',
            [0.0, 0.0],
            ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY)
        )

        self.declare_parameter('hotspot_on_threshold', 50.0)
        self.declare_parameter('hotspot_off_threshold', 10.0)
        self.declare_parameter('hotspot_detect_cycles', 3)
        self.declare_parameter('visit_neg_amount', 100.0)
        self.declare_parameter('mode_cooldown_secs', 10.0)
        self.declare_parameter('hotspot_step', 8.0)
        self.declare_parameter('hotspot_direct', False)
        self.declare_parameter('converge_z', 2.0)
        self.declare_parameter('explore_z', 2.0)
        self.declare_parameter('close_radius', 2.0)
        self.declare_parameter('dwell_cycles', 5)

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

        self.explore_min_eta_dist = float(self.get_parameter('explore_min_eta_dist').value)
        self.explore_min_target_sep = float(self.get_parameter('explore_min_target_sep').value)
        self.explore_neg_block_threshold = float(self.get_parameter('explore_neg_block_threshold').value)

        self.converge_min_eta_dist = float(self.get_parameter('converge_min_eta_dist').value)

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

        self.map_pos = None
        self.map_neg = None
        self.mode = 'EXPLORE'

        self.peak_xy = (0.0, 0.0)
        self.hotspot_xy = None

        self.converge_dwell = 0
        self.current_pose = None
        self.last_converge_exit = None
        self.detect_counter = 0
        self.last_wp_sent = None
        self.eps = 1e-6

        self._X = None
        self._Y = None

        raw = self.get_parameter('waypoints').value
        self.waypoints = []
        if isinstance(raw, list) and len(raw) >= 2 and len(raw) % 2 == 0:
            for i in range(0, len(raw), 2):
                self.waypoints.append({'x': float(raw[i]), 'y': float(raw[i + 1])})
        else:
            self.waypoints = [{'x': 0.0, 'y': 0.0}]

        # Subscriptions
        self.map_sub = self.create_subscription(Float32MultiArray, '/pheromone_map', self.map_callback, 10)
        self.map_neg_sub = self.create_subscription(Float32MultiArray, '/pheromone_map_neg', self.map_neg_callback, 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/local_position/pose', self.pose_callback, qos_profile_sensor_data)

        # Publishers
        self.pub_next_wp = self.create_publisher(PointStamped, '/aco_next_waypoint', 10)
        self.neg_pub = self.create_publisher(PointStamped, '/pheromone_deposit_neg', 10)
        self.mode_pub = self.create_publisher(String, '/aco_mode', 10)

        # Timer
        self.timer = self.create_timer(max(0.01, 1.0 / self.decision_rate), self.main_loop)

        self.get_logger().info(
            f"ACO Decision node started: "
            f"EXPLORE(grid sample) a={self.alpha_explore:.2f} b={self.beta_explore:.2f} g={self.gamma_explore:.2f} eps={self.eps_explore:.2f} "
            f"| CONVERGE(argmax score) a={self.alpha_converge:.2f} b={self.beta_converge:.2f} "
            f"| map={self.width}x{self.height}@{self.resolution}m rate={self.decision_rate:.2f}Hz"
        )

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
            return
        self.map_neg = arr.reshape((self.height, self.width))

    def pose_callback(self, msg: PoseStamped):
        self.current_pose = (float(msg.pose.position.x), float(msg.pose.position.y))

    def publish_mode(self):
        m = String()
        m.data = self.mode
        self.mode_pub.publish(m)

    def cell_to_world(self, cx, cy):
        x = self.origin_x + (cx + 0.5) * self.resolution
        y = self.origin_y + (cy + 0.5) * self.resolution
        return x, y

    def world_to_cell(self, x, y):
        cx = int((x - self.origin_x) / self.resolution)
        cy = int((y - self.origin_y) / self.resolution)
        return cx, cy

    def _world_grids(self):
        if self._X is not None and self._Y is not None:
            return self._X, self._Y
        xs = self.origin_x + (np.arange(self.width, dtype=np.float32) + 0.5) * self.resolution
        ys = self.origin_y + (np.arange(self.height, dtype=np.float32) + 0.5) * self.resolution
        self._X, self._Y = np.meshgrid(xs, ys)  # (H,W)
        return self._X, self._Y

    def find_hotspot_peak(self):
        """Return (peak_x, peak_y, peak_val) from POSITIVE map."""
        if self.map_pos is None:
            return None
        cy, cx = np.unravel_index(np.argmax(self.map_pos), self.map_pos.shape)
        peak_val = float(self.map_pos[cy, cx])
        px, py = self.cell_to_world(cx, cy)
        return (px, py, peak_val)

    def _effective_tau_grid(self):
        """tau = max(0, pos - neg) over the whole grid."""
        if self.map_pos is None:
            return None
        if self.map_neg is None:
            return np.maximum(self.map_pos, 0.0).astype(np.float32)
        return np.maximum(self.map_pos - self.map_neg, 0.0).astype(np.float32)

    def sample_explore_cell_simple(self):
        if self.map_pos is None or self.current_pose is None:
            return None

        tau_grid = self._effective_tau_grid()
        if tau_grid is None:
            return None

        if self.tau_clip_explore > 0.0:
            tau_grid = np.minimum(tau_grid, float(self.tau_clip_explore))

        neg_grid = np.zeros_like(tau_grid, dtype=np.float32) if self.map_neg is None else self.map_neg.astype(np.float32)

        px, py = self.current_pose
        a, b, g = self.alpha_explore, self.beta_explore, self.gamma_explore

        X, Y = self._world_grids()
        dx = X - float(px)
        dy = Y - float(py)
        dist = np.sqrt(dx * dx + dy * dy).astype(np.float32)

        dist_eta = np.maximum(dist, max(self.explore_min_eta_dist, self.eps)).astype(np.float32)
        eta = (1.0 / dist_eta).astype(np.float32)

        tau = tau_grid.astype(np.float32) + self.eps
        neg = np.maximum(0.0, neg_grid).astype(np.float32)

        score = (tau ** (-a)) * (eta ** b) * ((1.0 + neg) ** (-g))

        score[~np.isfinite(score)] = 0.0
        score[score < 0.0] = 0.0

        # too close to drone
        score[dist < self.explore_min_target_sep] = 0.0

        if self.block_r_cells > 0:
            cpx, cpy = self.world_to_cell(px, py)
            x0 = max(0, cpx - self.block_r_cells)
            x1 = min(self.width, cpx + self.block_r_cells + 1)
            y0 = max(0, cpy - self.block_r_cells)
            y1 = min(self.height, cpy + self.block_r_cells + 1)
            score[y0:y1, x0:x1] = 0.0

        # optional hard-block on neg
        if self.explore_neg_block_threshold >= 0.0:
            score[neg_grid > self.explore_neg_block_threshold] = 0.0

        # mask around last hotspot
        if self.mask_radius_explore > 0.0 and self.peak_xy is not None:
            hx, hy = self.peak_xy
            mdx = X - float(hx)
            mdy = Y - float(hy)
            m2 = mdx * mdx + mdy * mdy
            score[m2 <= (self.mask_radius_explore ** 2)] = 0.0

        # avoid repeating last target
        if self.last_wp_sent is not None:
            lx, ly = self.last_wp_sent
            ldx = X - float(lx)
            ldy = Y - float(ly)
            l2 = ldx * ldx + ldy * ldy
            score[l2 < (self.min_wp_sep ** 2)] = 0.0

        weights = score.astype(np.float64).ravel()
        nz = np.flatnonzero(weights > 0.0)
        if nz.size == 0:
            return None

        # epsilon random among valid cells (optional)
        if random.random() < max(0.0, min(1.0, self.eps_explore)):
            idx = int(np.random.choice(nz))
        else:
            w = weights[nz]
            s = float(w.sum())
            if not math.isfinite(s) or s <= 0.0:
                return None
            p = w / s
            p = p / float(p.sum())
            idx = int(np.random.choice(nz, p=p))

        cy = idx // self.width
        cx = idx % self.width
        wx, wy = self.cell_to_world(int(cx), int(cy))
        return (wx, wy, float(score[cy, cx]))


    def find_converge_argmax_score_cell(self):
        if self.map_pos is None:
            return None

        tau_grid = self._effective_tau_grid()
        if tau_grid is None:
            return None

        tau = tau_grid.astype(np.float32) + self.eps
        a, b = self.alpha_converge, self.beta_converge

        if self.current_pose is None:
            cy, cx = np.unravel_index(np.argmax(tau), tau.shape)
            wx, wy = self.cell_to_world(int(cx), int(cy))
            return (wx, wy, float(tau[cy, cx] ** a))

        px, py = self.current_pose
        X, Y = self._world_grids()

        dx = X - float(px)
        dy = Y - float(py)
        dist = np.sqrt(dx * dx + dy * dy).astype(np.float32)

        dist_eta = np.maximum(dist, max(self.converge_min_eta_dist, self.eps)).astype(np.float32)
        eta = (1.0 / dist_eta).astype(np.float32)

        score = (tau ** a) * (eta ** b)
        score[~np.isfinite(score)] = -np.inf

        cy, cx = np.unravel_index(np.argmax(score), score.shape)
        wx, wy = self.cell_to_world(int(cx), int(cy))
        return (wx, wy, float(score[cy, cx]))

    def main_loop(self):
        if self.map_pos is None:
            return

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
                    self.hotspot_xy = (px, py)
                    self.peak_xy = self.hotspot_xy
                    self.converge_dwell = 0
                    self.detect_counter = 0
                    self.publish_mode()
                    self.get_logger().info(
                        f"Hotspot detected (stable {self.hotspot_detect_cycles}): {pval:.1f} → CONVERGE "
                        f"hotspot=({px:.1f},{py:.1f})"
                    )

            elif self.mode == 'CONVERGE':
                if pval <= self.hotspot_off:
                    self.mode = 'EXPLORE'
                    self.last_converge_exit = now
                    self.converge_dwell = 0
                    self.publish_mode()
                    self.get_logger().info(
                        f"Hotspot faded → EXPLORE (cooldown {self.mode_cooldown_secs:.0f}s)"
                    )

        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"

        if self.mode == 'CONVERGE':
            latest = self.find_hotspot_peak()
            if latest is not None:
                self.hotspot_xy = (latest[0], latest[1])
                self.peak_xy = self.hotspot_xy

            if self.hotspot_xy is None:
                self.mode = 'EXPLORE'
                self.publish_mode()
                return

            best = self.find_converge_argmax_score_cell()
            if best is None:
                target_x, target_y = self.hotspot_xy
            else:
                target_x, target_y, _ = best

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

            hx, hy = self.hotspot_xy
            d = float('inf')
            if self.current_pose is not None:
                px, py = self.current_pose
                d = math.hypot(hx - px, hy - py)

            if d <= self.close_radius:
                self.converge_dwell += 1
            else:
                self.converge_dwell = 0

            if self.converge_dwell >= self.dwell_cycles:
                cx, cy = self.world_to_cell(hx, hy)
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

                self.get_logger().info(f"[CONVERGE] cooled hotspot at ({hx:.1f},{hy:.1f}); → EXPLORE")
                self.mode = 'EXPLORE'
                self.converge_dwell = 0
                self.last_converge_exit = now
                self.publish_mode()

            return

        if self.current_pose is not None and self.explore_neg_amount > 0.0:
            vx, vy = self.current_pose
            vmsg = PointStamped()
            vmsg.header.stamp = msg.header.stamp
            vmsg.header.frame_id = "world"
            vmsg.point.x = float(vx)
            vmsg.point.y = float(vy)
            vmsg.point.z = float(self.explore_neg_amount)
            self.neg_pub.publish(vmsg)

        choice = self.sample_explore_cell_simple()
        if choice is None:
            return
        bx, by, _ = choice

        tx, ty = bx, by
        if self.current_pose is not None:
            px, py = self.current_pose
            dx = tx - px
            dy = ty - py
            dist = math.hypot(dx, dy)
            if dist > 1e-3:
                scale = min(1.0, self.explore_step / dist)
                tx = px + dx * scale
                ty = py + dy * scale

        if self.last_wp_sent is not None:
            lx, ly = self.last_wp_sent
            if math.hypot(tx - lx, ty - ly) < self.min_wp_sep:
                ang = random.uniform(0.0, 2.0 * math.pi)
                tx += self.min_wp_sep * math.cos(ang)
                ty += self.min_wp_sep * math.sin(ang)

        msg.point.x = float(tx)
        msg.point.y = float(ty)
        msg.point.z = float(self.explore_z)
        self.pub_next_wp.publish(msg)
        self.last_wp_sent = (msg.point.x, msg.point.y)

        self.get_logger().info(f"[EXPLORE(grid sample)] sampled=({bx:.1f},{by:.1f}) step_to=({tx:.1f},{ty:.1f})")


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
