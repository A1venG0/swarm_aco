#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State
from rclpy.qos import qos_profile_sensor_data
import math

class ACONavigationBridge(Node):
    def __init__(self):
        super().__init__('aco_navigation_bridge')
        
        self.declare_parameter('drone_id', 'drone1')
        self.declare_parameter('waypoint_tolerance', 1.0)
        
        self.drone_id = self.get_parameter('drone_id').value
        self.tolerance = self.get_parameter('waypoint_tolerance').value
        
        # State
        self.current_pose = None
        self.target_waypoint = None
        self.current_state = None
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            qos_profile_sensor_data
        )

        self.pose_sub = self.create_subscription(
            PoseStamped, '/local_position/pose', self.pose_callback, qos_profile_sensor_data
        )
        
        self.waypoint_sub = self.create_subscription(
            PointStamped,
            '/aco_next_waypoint',
            self.waypoint_callback,
            10
        )
        
        self.state_sub = self.create_subscription(
            State,
            '/state',
            self.state_callback,
            10
        )
        
        # Publishers
        self.setpoint_pub = self.create_publisher(
            PoseStamped,
            '/setpoint_position/local',
            10
        )
        
        # Services
        self.arming_client = self.create_client(CommandBool, '/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/set_mode')
        
        # Timer for sending setpoints (required by MAVROS)
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info(
            f'ACO Navigation Bridge for {self.drone_id} started. '
            f'Tolerance: {self.tolerance}m'
        )

    def pose_callback(self, msg: PoseStamped):
        self.current_pose = msg.pose.position

    def odom_callback(self, msg: Odometry):
        """Update drone position from odometry."""
        self.current_pose = msg.pose.pose.position
    
    def state_callback(self, msg: State):
        """Update MAVROS state."""
        self.current_state = msg
    
    def waypoint_callback(self, msg: PointStamped):
        """Receive new waypoint from ACO decision node."""
        self.target_waypoint = msg.point
        self.get_logger().info(
            f'{self.drone_id}: New ACO waypoint -> '
            f'({msg.point.x:.2f}, {msg.point.y:.2f})'
        )
    
    def control_loop(self):
        """Send setpoints to MAVROS at high frequency."""
        if self.target_waypoint is None or self.current_pose is None:
            return
        
        # Check if waypoint reached
        dist = math.hypot(
            self.target_waypoint.x - self.current_pose.x,
            self.target_waypoint.y - self.current_pose.y
        )
        
        if dist < self.tolerance:
            self.get_logger().info(
                f'{self.drone_id}: Waypoint reached! Distance: {dist:.2f}m',
                throttle_duration_sec=2.0
            )
        
        # Send setpoint to MAVROS
        setpoint = PoseStamped()
        setpoint.header.stamp = self.get_clock().now().to_msg()
        setpoint.header.frame_id = 'map'
        setpoint.pose.position.x = self.target_waypoint.x
        setpoint.pose.position.y = self.target_waypoint.y
        setpoint.pose.position.z = 2.0  # Fixed altitude
        
        # Keep orientation level (optional: add yaw control)
        setpoint.pose.orientation.w = 1.0
        setpoint.pose.orientation.x = 0.0
        setpoint.pose.orientation.y = 0.0
        setpoint.pose.orientation.z = 0.0
        
        self.setpoint_pub.publish(setpoint)

def main(args=None):
    rclpy.init(args=args)
    node = ACONavigationBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()