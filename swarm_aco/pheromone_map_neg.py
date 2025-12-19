#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PointStamped
import numpy as np
from threading import Lock

class PheromoneMapNegNode(Node):
    def __init__(self):
        super().__init__('pheromone_map_neg_node')

        self.width      = self.declare_parameter('width', 120).value
        self.height     = self.declare_parameter('height', 120).value
        self.resolution = self.declare_parameter('resolution', 1.0).value
        self.origin_x   = self.declare_parameter('origin_x', -50.0).value
        self.origin_y   = self.declare_parameter('origin_y', -50.0).value

        self.rho          = self.declare_parameter('evaporation_neg', 0.02).value
        self.publish_rate = self.declare_parameter('publish_rate', 1.0).value
        self.max_val      = self.declare_parameter('max_val_neg', 100.0).value
        self.deposit_def  = self.declare_parameter('deposit_amount_neg', 2.0).value

        self.grid = np.zeros((self.height, self.width), dtype=float)
        self.lock = Lock()

        self.create_subscription(PointStamped, '/pheromone_deposit_neg',
                                 self.deposit_cb, 10)
        self.pub = self.create_publisher(Float32MultiArray,
                                         '/pheromone_map_neg', 10)

        self.create_timer(1.0 / self.publish_rate, self.tick)

        self.get_logger().info(
            f'PheromoneMapNegNode ready: {self.width}x{self.height}@{self.resolution}m'
        )

    def world_to_cell(self, x, y):
        cx = int((x - self.origin_x) / self.resolution)
        cy = int((y - self.origin_y) / self.resolution)
        return cx, cy

    def deposit_cb(self, msg: PointStamped):
        amount = msg.point.z if msg.point.z != 0.0 else self.deposit_def
        x, y = msg.point.x, msg.point.y
        self.add_deposit(x, y, amount)

    def add_deposit(self, x, y, amount, spread=1):
        cx, cy = self.world_to_cell(x, y)
        with self.lock:
            for dy in range(-spread, spread + 1):
                for dx in range(-spread, spread + 1):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        dist = np.hypot(dx, dy)
                        factor = np.exp(-(dist**2) / (2.0 * (spread/2.0 + 1e-3)**2))
                        self.grid[ny, nx] = min(
                            self.max_val,
                            self.grid[ny, nx] + amount * factor
                        )

    def tick(self):
        with self.lock:
            self.grid *= (1.0 - self.rho)
            self.grid[self.grid < 1e-4] = 0.0

            msg = Float32MultiArray()
            msg.data = self.grid.flatten().tolist()
            self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = PheromoneMapNegNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
