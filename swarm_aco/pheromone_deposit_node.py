#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration

from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Header
from visualization_msgs.msg import Marker


class PheromoneDepositNode(Node):
    def __init__(self):
        super().__init__('pheromone_deposit_node')

        # Parameters
        self.declare_parameter('drone_id', 'drone1')
        self.declare_parameter('deposit_rate', 2.0)

        # Base amounts (pre-scale)
        self.declare_parameter('base_deposit', 5.0)
        self.declare_parameter('success_deposit', 20.0)

        # Negative pheromone support
        self.declare_parameter('enable_negative', True)
        self.declare_parameter('base_neg_deposit', 1.5)
        self.declare_parameter('success_neg_deposit', 8.0)

        # Mode-aware scaling
        self.declare_parameter('explore_pos_scale', 0.0)
        self.declare_parameter('explore_neg_scale', 1.0)
        self.declare_parameter('converge_pos_scale', 1.0)
        self.declare_parameter('converge_neg_scale', 0.25)

        # Min movement before dropping (meters)
        self.declare_parameter('min_move_m', 0.4)

        # Mode topic (String: "EXPLORE"/"CONVERGE")
        self.declare_parameter('mode_topic', '/aco_mode')

        # Marker viz params
        self.declare_parameter('viz_enable', True)
        self.declare_parameter('viz_lifetime_sec', 3.0)
        self.declare_parameter('viz_size_base', 0.18) # ground footprint (m)
        self.declare_parameter('viz_size_gain', 0.01) # +amount -> bigger blob
        self.declare_parameter('frame_id', 'world')

        # --- Read params ---
        self.drone_id = self.get_parameter('drone_id').value
        self.deposit_rate = float(self.get_parameter('deposit_rate').value)

        self.base_deposit = float(self.get_parameter('base_deposit').value)
        self.success_deposit = float(self.get_parameter('success_deposit').value)

        self.enable_negative = bool(self.get_parameter('enable_negative').value)
        self.base_neg_deposit = float(self.get_parameter('base_neg_deposit').value)
        self.success_neg_deposit = float(self.get_parameter('success_neg_deposit').value)

        self.explore_pos_scale = float(self.get_parameter('explore_pos_scale').value)
        self.explore_neg_scale = float(self.get_parameter('explore_neg_scale').value)
        self.converge_pos_scale = float(self.get_parameter('converge_pos_scale').value)
        self.converge_neg_scale = float(self.get_parameter('converge_neg_scale').value)

        self.min_move_m = float(self.get_parameter('min_move_m').value)
        self.mode_topic = self.get_parameter('mode_topic').value

        self.viz_enable = bool(self.get_parameter('viz_enable').value)
        self.viz_lifetime_sec = float(self.get_parameter('viz_lifetime_sec').value)
        self.viz_size_base = float(self.get_parameter('viz_size_base').value)
        self.viz_size_gain = float(self.get_parameter('viz_size_gain').value)
        self.frame_id = self.get_parameter('frame_id').value

        # State
        self.current_pose = None
        self.last_drop_xy = None
        self.task_success = False
        self.current_mode = 'EXPLORE'
        self._viz_id = 0

        # Subscriptions
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, qos_profile_sensor_data
        )
        self.status_sub = self.create_subscription(
            String, '/task_status', self.status_callback, 10
        )
        self.mode_sub = self.create_subscription(
            String, self.mode_topic, self.mode_callback, 10
        )

        # Publishers
        self.deposit_pub = self.create_publisher(PointStamped, '/pheromone_deposit', 10)
        self.deposit_neg_pub = self.create_publisher(PointStamped, '/pheromone_deposit_neg', 10)

        # Marker mirrors (toggle in RViz)
        self.viz_pos_pub = self.create_publisher(Marker, '/pheromone_viz_pos', 10)
        self.viz_neg_pub = self.create_publisher(Marker, '/pheromone_viz_neg', 10)

        # Timer
        self.timer = self.create_timer(max(0.01, 1.0 / self.deposit_rate), self.deposit_callback)

        self.get_logger().info(
            f'{self.drone_id}: deposit node up | +base={self.base_deposit} +succ={self.success_deposit} '
            f'| neg={self.enable_negative} (-base={self.base_neg_deposit} -succ={self.success_neg_deposit}) '
            f'| mode scales: EXPLORE(+x{self.explore_pos_scale},-x{self.explore_neg_scale}) '
            f'CONVERGE(+x{self.converge_pos_scale},-x{self.converge_neg_scale}) '
            f'| min_move={self.min_move_m}m | mode_topic={self.mode_topic} | viz={self.viz_enable}'
        )

    # Callbacks
    def odom_callback(self, msg: Odometry):
        self.current_pose = msg.pose.pose.position

    def status_callback(self, msg: String):
        if 'success' in msg.data.lower() or 'completed' in msg.data.lower():
            self.task_success = True
            self.get_logger().info(f'{self.drone_id} task completed → bonus pheromone next tick')

    def mode_callback(self, msg: String):
        incoming = msg.data.strip().upper()
        if incoming in ('EXPLORE', 'CONVERGE') and incoming != self.current_mode:
            self.current_mode = incoming
            self.get_logger().info(f'{self.drone_id}: mode → {self.current_mode}')

    # Helpers
    def moved_enough(self, x, y):
        if self.last_drop_xy is None:
            return True
        lx, ly = self.last_drop_xy
        return math.hypot(x - lx, y - ly) >= self.min_move_m

    def _publish_viz(self, x: float, y: float, amount: float, positive: bool):
        if not self.viz_enable:
            return
        m = Marker()
        m.header = Header()
        m.header.frame_id = self.frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'pheromone_pos' if positive else 'pheromone_neg'
        self._viz_id = (self._viz_id + 1) % 100000
        m.id = self._viz_id
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = 0.05
        s = max(0.12, min(0.8, self.viz_size_base + self.viz_size_gain * amount))
        m.scale.x = s
        m.scale.y = s
        m.scale.z = 0.06
        if positive:
            m.color.r, m.color.g, m.color.b = 0.1, 1.0, 0.1  # green
        else:
            m.color.r, m.color.g, m.color.b = 1.0, 0.0, 1.0  # magenta
        m.color.a = max(0.25, min(0.95, 0.25 + 0.01 * amount))
        m.lifetime = Duration(seconds=self.viz_lifetime_sec).to_msg()
        (self.viz_pos_pub if positive else self.viz_neg_pub).publish(m)

    # Main deposit loop
    def deposit_callback(self):
        if self.current_pose is None:
            self.get_logger().debug('Waiting for odom...', throttle_duration_sec=5.0)
            return

        x = float(self.current_pose.x)
        y = float(self.current_pose.y)

        if not self.moved_enough(x, y):
            return

        # base amounts (+/-) with optional success bonus
        pos_amt = self.success_deposit if self.task_success else self.base_deposit
        neg_amt = self.success_neg_deposit if self.task_success else self.base_neg_deposit

        # apply mode scaling
        if self.current_mode == 'EXPLORE':
            pos_amt *= self.explore_pos_scale
            neg_amt *= self.explore_neg_scale
        else:  # CONVERGE
            pos_amt *= self.converge_pos_scale
            neg_amt *= self.converge_neg_scale

        # Publish positive
        if pos_amt > 0.0:
            pos = PointStamped()
            pos.header.stamp = self.get_clock().now().to_msg()
            pos.header.frame_id = self.frame_id
            pos.point.x = x
            pos.point.y = y
            pos.point.z = float(pos_amt)
            self.deposit_pub.publish(pos)
            self._publish_viz(x, y, pos_amt, positive=True)

        # Publish negative
        if self.enable_negative and neg_amt > 0.0:
            neg = PointStamped()
            neg.header.stamp = self.get_clock().now().to_msg()
            neg.header.frame_id = self.frame_id
            neg.point.x = x
            neg.point.y = y
            neg.point.z = float(neg_amt)
            self.deposit_neg_pub.publish(neg)
            self._publish_viz(x, y, neg_amt, positive=False)

        # Logging & state
        if self.task_success:
            self.get_logger().info(
                f'{self.drone_id} SUCCESS drop @ ({x:.1f},{y:.1f}) '
                f'mode={self.current_mode} +{pos_amt:.1f} / -{neg_amt:.1f}'
            )
            self.task_success = False
        else:
            self.get_logger().debug(
                f'{self.drone_id} drop @ ({x:.1f},{y:.1f}) '
                f'mode={self.current_mode} +{pos_amt:.1f} / -{neg_amt:.1f}',
                throttle_duration_sec=5.0
            )

        self.last_drop_xy = (x, y)


def main(args=None):
    rclpy.init(args=args)
    node = PheromoneDepositNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
