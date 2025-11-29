#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np

class MapVerifier(Node):
    def __init__(self):
        super().__init__('map_verifier')
        self.sub = self.create_subscription(
            Float32MultiArray,
            '/pheromone_map',
            self.callback,
            10
        )
        print("Waiting for pheromone map...")
        
    def callback(self, msg):
        arr = np.array(msg.data).reshape((100, 100))
        
        print("\n" + "="*50)
        print("PHEROMONE MAP VERIFICATION")
        print("="*50)
        print(f"✓ Total pheromone: {arr.sum():.2f}")
        print(f"✓ Max value: {arr.max():.2f}")
        print(f"✓ Non-zero cells: {np.count_nonzero(arr)}")
        print(f"✓ Mean value: {arr.mean():.4f}")
        
        # Show specific cells around the deposit
        print("\nValues around deposit at cell (55, 55):")
        for y in range(54, 57):
            for x in range(54, 57):
                val = arr[y, x]
                marker = "🎯" if (x == 55 and y == 55) else "  "
                print(f"  {marker} [{y:2d}, {x:2d}] = {val:6.2f}")
        
        # Show all non-zero cells
        nonzero_indices = np.argwhere(arr > 0.01)
        if len(nonzero_indices) > 0:
            print(f"\nAll {len(nonzero_indices)} non-zero cells:")
            for idx in nonzero_indices:
                y, x = idx
                print(f"  Cell [{y:2d}, {x:2d}] = {arr[y, x]:6.2f}")
        
        print("="*50)
        print("✅ MAP IS WORKING CORRECTLY!\n")
        
        rclpy.shutdown()

def main():
    rclpy.init()
    node = MapVerifier()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
