#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PointStamped
import math
import random


class ACODecisionNode(Node):
    def __init__(self):
        super().__init__('aco_decision_node')

        # --- Parameters ---
        self.declare_parameter('alpha', 1.0)
        self.declare_parameter('beta', 2.0)
        self.declare_parameter('resolution', 1.0)
        self.declare_parameter('origin_x', -50.0)
        self.declare_parameter('origin_y', -50.0)
        self.declare_parameter('width', 120)
        self.declare_parameter('height', 120)
        self.declare_parameter('decision_rate', 1.0)
        
        # Changed: waypoints as flat list [x1, y1, x2, y2, x3, y3, ...]
        # Use in your aco_decision_node init
        self.declare_parameter('waypoint_step', 10.0)
        step = float(self.get_parameter('waypoint_step').value)
        xs = np.arange(-50.0, 50.1, step)
        ys = np.arange(-50.0, 50.1, step)

        # Stagger rows a bit per-drone to reduce collisions (optional):
        phase = 0.0
        if self.get_name().endswith('_aco'):  # crude example
            if 'drone2' in self.get_namespace(): phase = step/3
            if 'drone3' in self.get_namespace(): phase = 2*step/3

        waypoints = []
        for iy, y in enumerate(ys):
            row = xs if iy % 2 == 0 else xs[::-1]  # boustrophedon
            for x in (row + (phase if (iy % 2 == 0) else -phase)):
                waypoints.append({'x': float(x), 'y': float(y)})

        self.waypoints = waypoints


        # --- Internal values ---
        self.alpha = self.get_parameter('alpha').value
        self.beta = self.get_parameter('beta').value
        self.resolution = self.get_parameter('resolution').value
        self.origin_x = self.get_parameter('origin_x').value
        self.origin_y = self.get_parameter('origin_y').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.decision_rate = self.get_parameter('decision_rate').value
        
        # Parse waypoints from flat list
        waypoints_flat = self.get_parameter('waypoints').value
        raw = self.get_parameter('waypoints').value
        self.waypoints = []
        if isinstance(raw, list) and len(raw) > 0:
            if isinstance(raw[0], dict):  # support [{'x':..,'y':..}, ...]
                self.waypoints = [{'x': d['x'], 'y': d['y']} for d in raw]
            else:  # assume flat list [x1, y1, x2, y2, ...]
                assert len(raw) % 2 == 0, "waypoints must be even-length flat list"
                for i in range(0, len(raw), 2):
                    self.waypoints.append({'x': raw[i], 'y': raw[i+1]})
        else:
            self.get_logger().warn("No waypoints given; using default")
            self.waypoints = [{'x':0.0,'y':0.0}]

        self.eps = 1e-6
        self.map = None

        # --- ROS 2 Interfaces ---
        self.map_sub = self.create_subscription(
            Float32MultiArray,
            '/pheromone_map',
            self.map_callback,
            10
        )
        self.pub_next_wp = self.create_publisher(PointStamped, '/aco_next_waypoint', 10)

        # --- Timer for main loop ---
        timer_period = 1.0 / self.decision_rate
        self.timer = self.create_timer(timer_period, self.main_loop)

        # --- State ---
        self.current_idx = 0

        self.get_logger().info(
            f"ACO Decision node started: alpha={self.alpha:.2f}, beta={self.beta:.2f}, "
            f"decision_rate={self.decision_rate:.2f} Hz, waypoints={len(self.waypoints)}"
        )
        self.get_logger().info(f"Waypoints: {self.waypoints}")

    # --- Callbacks & Utilities ---
    def map_callback(self, msg):
        """Receive pheromone map as a flattened Float32MultiArray."""
        arr = np.array(msg.data, dtype=np.float32)
        expected_size = int(self.width * self.height)

        if arr.size != expected_size:
            self.get_logger().warn(
                f"Pheromone map size mismatch ({arr.size} != {expected_size})"
            )
            return

        self.map = arr.reshape((self.height, self.width))

    def world_to_cell_value(self, x, y):
        """Convert world coordinates to map cell and return pheromone value."""
        if self.map is None:
            return 0.0

        cx = int((x - self.origin_x) / self.resolution)
        cy = int((y - self.origin_y) / self.resolution)
        h, w = self.map.shape

        if 0 <= cx < w and 0 <= cy < h:
            return float(self.map[cy, cx])
        else:
            return 0.0

    def choose_next(self, current_idx):
        """Select next waypoint probabilistically using ACO logic."""
        current = self.waypoints[current_idx]
        scores = []

        for j, wp in enumerate(self.waypoints):
            if j == current_idx:
                scores.append(0.0)
                continue

            tau = self.world_to_cell_value(wp['x'], wp['y']) + self.eps
            dist = math.hypot(wp['x'] - current['x'], wp['y'] - current['y']) + self.eps
            eta = 1.0 / dist
            score = (tau ** self.alpha) * (eta ** self.beta)
            scores.append(score)

        total = sum(scores)
        if total == 0.0:
            # fallback random
            candidates = [i for i in range(len(self.waypoints)) if i != current_idx]
            return random.choice(candidates)

        probs = [v / total for v in scores]
        next_idx = np.random.choice(range(len(probs)), p=probs)
        return next_idx

    def main_loop(self):
        """Periodically choose the next waypoint and publish it."""
        if self.map is None:
            self.get_logger().debug("Waiting for pheromone map...")
            return

        next_idx = self.choose_next(self.current_idx)
        wp = self.waypoints[next_idx]

        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.point.x = wp['x']
        msg.point.y = wp['y']
        msg.point.z = 0.0

        self.pub_next_wp.publish(msg)
        self.get_logger().info(
            f"From {self.current_idx} → Next {next_idx} ({wp['x']:.2f}, {wp['y']:.2f})"
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