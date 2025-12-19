#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Header
import random
import math
import time

class PheromoneHotspotNode(Node):
    def __init__(self):
        super().__init__('pheromone_hotspot_node')

        # Map bounds
        self.width      = self.declare_parameter('width', 120).value
        self.height     = self.declare_parameter('height', 120).value
        self.resolution = float(self.declare_parameter('resolution', 1.0).value)
        self.origin_x   = float(self.declare_parameter('origin_x', -50.0).value)
        self.origin_y   = float(self.declare_parameter('origin_y', -50.0).value)

        self.publish_rate = float(self.declare_parameter('publish_rate', 2.0).value)
        self.amount       = float(self.declare_parameter('amount', 60.0).value)
        self.spread_cells = int(self.declare_parameter('spread_cells', 2).value)
        self.falloff_sigma = float(self.declare_parameter('falloff_sigma', 1.0).value)
        self.randomize   = bool(self.declare_parameter('randomize', True).value)
        self.respawn_secs = float(self.declare_parameter('respawn_secs', 60.0).value)
        self.hotspot_x   = float(self.declare_parameter('hotspot_x', 0.0).value)
        self.hotspot_y   = float(self.declare_parameter('hotspot_y', 0.0).value)

        self.quiet_after_respawn_secs = float(self.declare_parameter('quiet_after_respawn_secs', 10.0).value)
        self.duty_on_secs  = float(self.declare_parameter('duty_on_secs', 0.0).value)
        self.duty_off_secs = float(self.declare_parameter('duty_off_secs', 0.0).value)
        self.max_active_duration_secs = float(self.declare_parameter('max_active_duration_secs', 0.0).value)

        self.ph_pub  = self.create_publisher(PointStamped, '/pheromone_deposit', 10)
        self.neg_pub = self.create_publisher(PointStamped, '/pheromone_deposit_neg', 10)
        self.marker_pub = self.create_publisher(Marker, '/aco_hotspot_marker', 10)

        # State
        if self.randomize:
            self._randomize_position()
        else:
            self.x, self.y = float(self.hotspot_x), float(self.hotspot_y)
        self.prev_xy = (self.x, self.y)

        # Internal state machine
        self.deposit_enabled = True
        self.last_respawn_time = time.time()
        self.last_toggle_time  = time.time()
        self.state_enter_time  = time.time()
        self._enter_quiet_if_needed(initial=True)

        # Timers
        self.deposit_timer = self.create_timer(max(0.01, 1.0 / self.publish_rate), self._tick_deposit)
        if self.randomize and self.respawn_secs > 0.0:
            self.respawn_timer = self.create_timer(self.respawn_secs, self._tick_respawn)
        else:
            self.respawn_timer = None

        self.get_logger().info(
            f'Hotspot ready at ({self.x:.2f},{self.y:.2f}); randomize={self.randomize}, '
            f'respawn={self.respawn_secs}s, amount={self.amount}, spread_cells={self.spread_cells}, '
            f'quiet_after_respawn={self.quiet_after_respawn_secs}s, duty=({self.duty_on_secs}/{self.duty_off_secs})s'
        )

    # helpers
    def _randomize_position(self):
        margin = max(2, self.spread_cells + 1)
        min_cx, max_cx = margin, self.width  - margin - 1
        min_cy, max_cy = margin, self.height - margin - 1
        cx = random.randint(min_cx, max_cx)
        cy = random.randint(min_cy, max_cy)
        self.x = self.origin_x + (cx + 0.5) * self.resolution
        self.y = self.origin_y + (cy + 0.5) * self.resolution

    def _clear_old_peak(self, ox, oy):
        """Blast a small negative patch around the old hotspot so drones release quickly."""
        clear_amt = 120.0
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                m = PointStamped()
                m.header.frame_id = 'world'
                m.header.stamp = self.get_clock().now().to_msg()
                m.point.x = float(ox + dx * self.resolution)
                m.point.y = float(oy + dy * self.resolution)
                m.point.z = float(clear_amt)
                self.neg_pub.publish(m)

    def _enter_quiet_if_needed(self, initial=False):
        """Enter a quiet period (no deposits) after respawn, or at startup if configured."""
        if self.quiet_after_respawn_secs > 0.0:
            self.deposit_enabled = False
            self.state_enter_time = time.time()
            if not initial:
                self.get_logger().info(f'Hotspot quiet for {self.quiet_after_respawn_secs:.0f}s to allow exploration')
        else:
            self.deposit_enabled = True
            self.state_enter_time = time.time()

    def _maybe_exit_quiet(self):
        if not self.deposit_enabled and (time.time() - self.state_enter_time >= self.quiet_after_respawn_secs):
            self.deposit_enabled = True
            self.last_toggle_time = time.time()
            self.state_enter_time = time.time()
            self.get_logger().info('Hotspot resumed depositing')

    def _apply_duty_cycle(self):
        """Toggle deposit_enabled according to duty_on/off if configured."""
        if self.duty_on_secs <= 0.0 or self.duty_off_secs <= 0.0:
            return  # disabled
        now = time.time()
        if self.deposit_enabled:
            # currently ON
            if now - self.last_toggle_time >= self.duty_on_secs:
                self.deposit_enabled = False
                self.last_toggle_time = now
        else:
            # currently OFF
            if now - self.last_toggle_time >= self.duty_off_secs:
                self.deposit_enabled = True
                self.last_toggle_time = now

    def _maybe_force_quiet_break(self):
        if self.max_active_duration_secs <= 0.0:
            return
        if self.deposit_enabled and (time.time() - self.state_enter_time >= self.max_active_duration_secs):
            duration = self.duty_off_secs if self.duty_off_secs > 0 else self.quiet_after_respawn_secs
            self.deposit_enabled = False
            self.last_toggle_time = time.time()
            self.state_enter_time = time.time()
            self.get_logger().info(f'Hotspot forced quiet for {duration:.0f}s (max_active_duration reached)')

    def _tick_respawn(self):
        if not self.randomize:
            return
        self._clear_old_peak(self.x, self.y)
        self.prev_xy = (self.x, self.y)
        self._randomize_position()
        self.last_respawn_time = time.time()
        self.get_logger().info(f'Hotspot moved to ({self.x:.2f},{self.y:.2f})')
        self._enter_quiet_if_needed(initial=False)

    def _publish_marker(self):
        m = Marker()
        m.header = Header()
        m.header.frame_id = 'world'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'aco_hotspot'
        m.id = 1
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(self.x)
        m.pose.position.y = float(self.y)
        m.pose.position.z = 0.5
        m.pose.orientation.w = 1.0
        m.scale.x = 2.0
        m.scale.y = 2.0
        m.scale.z = 1.0
        m.color.r = 1.0
        m.color.g = 0.5
        m.color.b = 0.0
        m.color.a = 0.9
        self.marker_pub.publish(m)

    # main depositing tick
    def _tick_deposit(self):
        # state housekeeping
        self._maybe_exit_quiet()
        self._apply_duty_cycle()
        self._maybe_force_quiet_break()

        self._publish_marker()

        if not self.deposit_enabled:
            return

        # center
        self._deposit(self.x, self.y, self.amount)
        R = self.spread_cells
        if R > 0:
            for dy in range(-R, R + 1):
                for dx in range(-R, R + 1):
                    if dx == 0 and dy == 0:
                        continue
                    dist = math.hypot(dx, dy)
                    w = math.exp(- (dist**2) / (2.0 * (self.falloff_sigma**2)))
                    if w < 1e-3:
                        continue
                    self._deposit(self.x + dx * self.resolution,
                                  self.y + dy * self.resolution,
                                  self.amount * w)

    def _deposit(self, x, y, amount):
        msg = PointStamped()
        msg.header.frame_id = 'world'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x = float(x)
        msg.point.y = float(y)
        msg.point.z = float(amount)
        self.ph_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = PheromoneHotspotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
