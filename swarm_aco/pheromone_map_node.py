#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PointStamped
import numpy as np
from threading import Lock

class PheromoneMapNode(Node):
    def __init__(self):
        super().__init__('pheromone_map_node')

        # grid params
        self.width = self.declare_parameter('width', 120).value
        self.height = self.declare_parameter('height', 120).value
        self.resolution = self.declare_parameter('resolution', 1.0).value
        self.origin_x = self.declare_parameter('origin_x', -50.0).value
        self.origin_y = self.declare_parameter('origin_y', -50.0).value

        # pheromone params
        self.rho = self.declare_parameter('evaporation', 0.05).value
        self.publish_rate = self.declare_parameter('publish_rate', 1.0).value
        self.max_val = self.declare_parameter('max_val', 100.0).value
        self.deposit_amount = self.declare_parameter('deposit_amount', 5.0).value

        # internal map
        self.pheromone = np.zeros((self.height, self.width), dtype=float)
        self.lock = Lock()

        # subscriptions
        self.create_subscription(PointStamped, '/pheromone_deposit', self.deposit_cb, 10)

        # publisher
        self.map_pub = self.create_publisher(Float32MultiArray, '/pheromone_map', 10)

        # timer for evaporation and publishing
        self.create_timer(1.0 / self.publish_rate, self.tick)

        self.get_logger().info(f'PheromoneMapNode ready: {self.width}x{self.height} @ {self.resolution}m')

    def world_to_cell(self, x, y):
        cx = int((x - self.origin_x) / self.resolution)
        cy = int((y - self.origin_y) / self.resolution)
        return cx, cy

    def deposit_cb(self, msg: PointStamped):
        amount = msg.point.z if msg.point.z != 0.0 else self.deposit_amount
        x = msg.point.x
        y = msg.point.y
        self.add_deposit(x, y, amount)

    def add_deposit(self, x, y, amount, spread=1):
        cx, cy = self.world_to_cell(x, y)
    
        #self.get_logger().info(f'=== add_deposit START ===')
        #self.get_logger().info(f'Parameters: x={x}, y={y}, amount={amount}, spread={spread}')
        #self.get_logger().info(f'Center cell: ({cx}, {cy})')
        #self.get_logger().info(f'Map dimensions: {self.pheromone.shape}')
        #self.get_logger().info(f'Before deposit - Total pheromone: {self.pheromone.sum():.2f}')
    
        with self.lock:
            cells_modified = 0
            for dy in range(-spread, spread+1):
                for dx in range(-spread, spread+1):
                    nx = cx + dx
                    ny = cy + dy
                    
                    #self.get_logger().info(f'Trying cell ({nx}, {ny})...')
                    
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        dist = np.hypot(dx, dy)
                        factor = np.exp(-(dist**2) / (2.0 * (spread / 2.0 + 0.001)**2))
                        
                        old_value = self.pheromone[ny, nx]
                        self.pheromone[ny, nx] += amount * factor
                        
                        if self.pheromone[ny, nx] > self.max_val:
                            self.pheromone[ny, nx] = self.max_val
                        
                        new_value = self.pheromone[ny, nx]
                        cells_modified += 1
                        
                        #self.get_logger().info(
                            #f'  ✓ Cell [{ny}, {nx}]: {old_value:.2f} -> {new_value:.2f} (factor={factor:.3f})'
                        #)
                    else:
                        self.get_logger().warn(f'  ✗ Cell ({nx}, {ny}) out of bounds!')
            
            # ADD THIS:
            #self.get_logger().info(f'After deposit - Total pheromone: {self.pheromone.sum():.2f}')
            #self.get_logger().info(f'Modified {cells_modified} cells')
            #self.get_logger().info(f'=== add_deposit END ===')

    def tick(self):
        with self.lock:
            # evaporation
            self.pheromone *= (1.0 - self.rho)
            self.pheromone[self.pheromone < 1e-4] = 0.0

            arr = Float32MultiArray()
            arr.data = self.pheromone.flatten().tolist()
            self.map_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = PheromoneMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
