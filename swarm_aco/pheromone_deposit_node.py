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
        
        self.drone_id = self.get_parameter('drone_id').value
        self.deposit_rate = self.get_parameter('deposit_rate').value
        self.base_deposit = self.get_parameter('base_deposit').value
        self.success_deposit = self.get_parameter('success_deposit').value
        
        # State
        self.current_pose = None
        self.task_success = False
        
        # Subscribe to drone odometry
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            qos_profile_sensor_data
        )
        
        # Subscribe to task status (optional)
        self.status_sub = self.create_subscription(
            String,
            '/task_status',
            self.status_callback,
            10
        )
        
        # Publisher
        self.deposit_pub = self.create_publisher(
            PointStamped, 
            '/pheromone_deposit', 
            10
        )
        
        # Timer
        self.timer = self.create_timer(1.0 / self.deposit_rate, self.deposit_callback)
        
        self.get_logger().info(
            f'PheromoneDepositNode for {self.drone_id} started. '
            f'Base deposit: {self.base_deposit}, Success: {self.success_deposit}'
        )
    
    def odom_callback(self, msg: Odometry):
        """Update drone position from odometry."""
        self.current_pose = msg.pose.pose.position
    
    def status_callback(self, msg: String):
        """Listen for task completion signals."""
        if 'success' in msg.data.lower() or 'completed' in msg.data.lower():
            self.task_success = True
            self.get_logger().info(f'{self.drone_id} task completed! Depositing bonus pheromone.')
    
    def deposit_callback(self):
        """Periodically deposit pheromone at current location."""
        if self.current_pose is None:
            self.get_logger().debug('Waiting for odometry...', throttle_duration_sec=5.0)
            return
        
        # Determine deposit amount
        amount = self.success_deposit if self.task_success else self.base_deposit
        
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.point.x = self.current_pose.x
        msg.point.y = self.current_pose.y
        msg.point.z = amount  # Using z field for deposit amount
        
        self.deposit_pub.publish(msg)
        
        if self.task_success:
            self.get_logger().info(
                f'{self.drone_id} deposited SUCCESS pheromone at '
                f'({msg.point.x:.2f}, {msg.point.y:.2f}) amount={amount:.2f}'
            )
            self.task_success = False  # Reset flag
        else:
            self.get_logger().debug(
                f'{self.drone_id} deposited at ({msg.point.x:.2f}, {msg.point.y:.2f})',
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