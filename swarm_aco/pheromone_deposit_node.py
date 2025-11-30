#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String

class PheromoneDepositNode(Node):
    def __init__(self):
        super().__init__('pheromone_deposit_node')

        # Parameters
        self.declare_parameter('drone_id', 'drone1')
        self.declare_parameter('deposit_rate', 2.0)
        self.declare_parameter('base_deposit', 5.0)
        self.declare_parameter('success_deposit', 20.0)

        # NEW: negative pheromone support
        self.declare_parameter('enable_negative', True)
        self.declare_parameter('base_neg_deposit', 1.5)
        self.declare_parameter('success_neg_deposit', 8.0)

        self.drone_id = self.get_parameter('drone_id').value
        self.deposit_rate = float(self.get_parameter('deposit_rate').value)
        self.base_deposit = float(self.get_parameter('base_deposit').value)
        self.success_deposit = float(self.get_parameter('success_deposit').value)

        self.enable_negative = bool(self.get_parameter('enable_negative').value)
        self.base_neg_deposit = float(self.get_parameter('base_neg_deposit').value)
        self.success_neg_deposit = float(self.get_parameter('success_neg_deposit').value)

        # State
        self.current_pose = None
        self.task_success = False

        # Odometry subscription (remap to /droneN/local_position/odom in launch)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, qos_profile_sensor_data
        )

        # Optional task-status
        self.status_sub = self.create_subscription(
            String, '/task_status', self.status_callback, 10
        )

        # Publishers
        self.deposit_pub = self.create_publisher(PointStamped, '/pheromone_deposit', 10)
        self.deposit_neg_pub = self.create_publisher(PointStamped, '/pheromone_deposit_neg', 10)

        # Timer
        self.timer = self.create_timer(max(0.01, 1.0 / self.deposit_rate), self.deposit_callback)

        self.get_logger().info(
            f'{self.drone_id}: deposit node up | +base={self.base_deposit} +succ={self.success_deposit} '
            f'| neg={self.enable_negative} (-base={self.base_neg_deposit} -succ={self.success_neg_deposit})'
        )

    def odom_callback(self, msg: Odometry):
        self.current_pose = msg.pose.pose.position

    def status_callback(self, msg: String):
        if 'success' in msg.data.lower() or 'completed' in msg.data.lower():
            self.task_success = True
            self.get_logger().info(f'{self.drone_id} task completed → bonus pheromone next tick')

    def deposit_callback(self):
        if self.current_pose is None:
            # keep it quiet but visible when needed
            self.get_logger().debug('Waiting for odom...', throttle_duration_sec=5.0)
            return

        # Choose amounts
        pos_amt = self.success_deposit if self.task_success else self.base_deposit
        neg_amt = self.success_neg_deposit if self.task_success else self.base_neg_deposit

        # Positive deposit
        pos = PointStamped()
        pos.header.stamp = self.get_clock().now().to_msg()
        pos.header.frame_id = 'world'
        pos.point.x = float(self.current_pose.x)
        pos.point.y = float(self.current_pose.y)
        pos.point.z = float(pos_amt)  # z carries amount
        self.deposit_pub.publish(pos)

        # Negative deposit (optional)
        if self.enable_negative:
            neg = PointStamped()
            neg.header.stamp = pos.header.stamp
            neg.header.frame_id = 'world'
            neg.point.x = pos.point.x
            neg.point.y = pos.point.y
            neg.point.z = float(neg_amt)
            self.deposit_neg_pub.publish(neg)

        if self.task_success:
            self.get_logger().info(
                f'{self.drone_id} SUCCESS drop @ ({pos.point.x:.1f},{pos.point.y:.1f}) '
                f'+{pos_amt:.1f} / -{neg_amt:.1f}'
            )
            self.task_success = False
        else:
            self.get_logger().debug(
                f'{self.drone_id} drop @ ({pos.point.x:.1f},{pos.point.y:.1f}) '
                f'+{pos_amt:.1f} / -{neg_amt:.1f}',
                throttle_duration_sec=5.0
            )

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
