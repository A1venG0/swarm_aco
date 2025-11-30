#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

from std_msgs.msg import Float32MultiArray, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose, Point
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class ACOVisualizerNode(Node):
    def __init__(self):
        super().__init__('aco_visualizer_node')

        # --- Map geometry (must match map nodes) ---
        self.declare_parameter('width', 100)
        self.declare_parameter('height', 100)
        self.declare_parameter('resolution', 1.0)
        self.declare_parameter('origin_x', -50.0)
        self.declare_parameter('origin_y', -50.0)
        self.declare_parameter('frame_id', 'map')

        # Topics (input)
        self.declare_parameter('topic_pos', '/pheromone_map')
        self.declare_parameter('topic_neg', '/pheromone_map_neg')

        # Visualization params
        self.declare_parameter('visualization_rate', 2.0)    # Hz
        self.declare_parameter('grid_skip', 4)
        self.declare_parameter('clip_max_pos', 100.0)
        self.declare_parameter('clip_max_neg', 100.0)
        self.declare_parameter('clip_max_combined', 100.0)
        self.declare_parameter('min_value', 0.1)
        self.declare_parameter('publish_grids', True)
        self.declare_parameter('publish_markers', True)

        # Read params
        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        self.resolution = float(self.get_parameter('resolution').value)
        self.origin_x = float(self.get_parameter('origin_x').value)
        self.origin_y = float(self.get_parameter('origin_y').value)
        self.frame_id = self.get_parameter('frame_id').value

        self.topic_pos = self.get_parameter('topic_pos').value
        self.topic_neg = self.get_parameter('topic_neg').value

        self.viz_rate = float(self.get_parameter('visualization_rate').value)
        self.grid_skip = int(self.get_parameter('grid_skip').value)
        self.clip_max_pos = float(self.get_parameter('clip_max_pos').value)
        self.clip_max_neg = float(self.get_parameter('clip_max_neg').value)
        self.clip_max_combined = float(self.get_parameter('clip_max_combined').value)
        self.min_value = float(self.get_parameter('min_value').value)
        self.publish_grids = bool(self.get_parameter('publish_grids').value)
        self.publish_markers = bool(self.get_parameter('publish_markers').value)

        # Internal maps
        self.map_pos = None
        self.map_neg = None

        # Cached grids for periodic republish
        self.last_pos_grid = None
        self.last_neg_grid = None
        self.last_comb_grid = None

        # Latched QoS for OccupancyGrid
        qos_grid = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # Publishers: grids
        self.grid_pos_pub = self.create_publisher(OccupancyGrid, '/pheromone_grid_pos', qos_grid)
        self.grid_neg_pub = self.create_publisher(OccupancyGrid, '/pheromone_grid_neg', qos_grid)
        self.grid_comb_pub = self.create_publisher(OccupancyGrid, '/pheromone_grid_combined', qos_grid)

        # Publishers: marker arrays
        self.mark_pos_pub = self.create_publisher(MarkerArray, '/aco_visualization_pos', 10)
        self.mark_neg_pub = self.create_publisher(MarkerArray, '/aco_visualization_neg', 10)
        self.mark_comb_pub = self.create_publisher(MarkerArray, '/aco_visualization_combined', 10)

        # Subscribers
        self.sub_pos = self.create_subscription(Float32MultiArray, self.topic_pos, self.pos_cb, 10)
        self.sub_neg = self.create_subscription(Float32MultiArray, self.topic_neg, self.neg_cb, 10)

        # Timer for markers + periodic grid re-publish
        self.timer = self.create_timer(1.0 / max(self.viz_rate, 1e-3), self.periodic_tick)

        # Static map info
        self.meta = MapMetaData()
        self.meta.resolution = self.resolution
        self.meta.width = self.width
        self.meta.height = self.height
        self.meta.origin = Pose()
        self.meta.origin.position.x = self.origin_x
        self.meta.origin.position.y = self.origin_y
        self.meta.origin.orientation.w = 1.0

        # Publish initial empty grids so topics exist immediately
        if self.publish_grids:
            zero = np.zeros((self.height, self.width), dtype=np.float32)
            self.last_pos_grid = self.make_grid(zero, self.clip_max_pos)
            self.last_neg_grid = self.make_grid(zero, self.clip_max_neg)
            self.last_comb_grid = self.make_grid(zero, self.clip_max_combined)
            self.grid_pos_pub.publish(self.last_pos_grid)
            self.grid_neg_pub.publish(self.last_neg_grid)
            self.grid_comb_pub.publish(self.last_comb_grid)

        self.get_logger().info(
            f'VisualizerDual ready: {self.width}x{self.height}@{self.resolution}m in frame "{self.frame_id}"\n'
            f'  inputs: pos={self.topic_pos}, neg={self.topic_neg}\n'
            f'  grids: /pheromone_grid_pos, /pheromone_grid_neg, /pheromone_grid_combined\n'
            f'  markers: /aco_visualization_pos, /aco_visualization_neg, /aco_visualization_combined'
        )

    # Callbacks
    def pos_cb(self, msg: Float32MultiArray):
        arr = np.array(msg.data, dtype=np.float32)
        if arr.size != self.width * self.height:
            self.get_logger().warn(f'POS size {arr.size} != {self.width*self.height}')
            return
        first = self.map_pos is None
        self.map_pos = arr.reshape((self.height, self.width))
        if first:
            self.get_logger().info('Received first POS map')
        if self.publish_grids:
            grid = self.make_grid(self.map_pos, self.clip_max_pos)
            self.last_pos_grid = grid
            self.grid_pos_pub.publish(grid)
            if self.map_neg is not None:
                comb = np.maximum(self.map_pos - self.map_neg, 0.0)
                gcomb = self.make_grid(comb, self.clip_max_combined)
                self.last_comb_grid = gcomb
                self.grid_comb_pub.publish(gcomb)

    def neg_cb(self, msg: Float32MultiArray):
        arr = np.array(msg.data, dtype=np.float32)
        if arr.size != self.width * self.height:
            self.get_logger().warn(f'NEG size {arr.size} != {self.width*self.height}')
            return
        first = self.map_neg is None
        self.map_neg = arr.reshape((self.height, self.width))
        if first:
            self.get_logger().info('Received first NEG map')
        if self.publish_grids:
            grid = self.make_grid(self.map_neg, self.clip_max_neg)
            self.last_neg_grid = grid
            self.grid_neg_pub.publish(grid)
            if self.map_pos is not None:
                comb = np.maximum(self.map_pos - self.map_neg, 0.0)
                gcomb = self.make_grid(comb, self.clip_max_combined)
                self.last_comb_grid = gcomb
                self.grid_comb_pub.publish(gcomb)

    # Publishing helpers
    def make_grid(self, mat: np.ndarray, clip_max: float) -> OccupancyGrid:
        clip = max(clip_max, 1e-6)
        scaled = np.clip(mat / clip, 0.0, 1.0)
        occ = (scaled * 100.0).astype(np.int8)
        grid = OccupancyGrid()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = self.frame_id
        grid.info = self.meta
        grid.data = occ.flatten().tolist()
        return grid

    def publish_marker_layer(self, mat: np.ndarray, pub, layer: str):
        marr = MarkerArray()
        mdel = Marker()
        mdel.header.frame_id = self.frame_id
        mdel.action = Marker.DELETEALL
        marr.markers.append(mdel)

        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = f'grid_{layer}'
        m.id = 0
        m.type = Marker.CUBE_LIST
        m.action = Marker.ADD
        m.scale.x = self.resolution * self.grid_skip
        m.scale.y = self.resolution * self.grid_skip
        m.scale.z = self.resolution
        m.pose.orientation.w = 1.0

        if layer == 'pos':
            clip = max(self.clip_max_pos, 1e-6)
        elif layer == 'neg':
            clip = max(self.clip_max_neg, 1e-6)
        else:
            clip = max(self.clip_max_combined, 1e-6)

        dyn_max = float(np.max(mat)) if mat.size else 0.0
        if dyn_max < 1e-6:
            dyn_max = clip

        pts, cols = [], []
        for cy in range(0, self.height, self.grid_skip):
            for cx in range(0, self.width, self.grid_skip):
                val = float(mat[cy, cx])
                if val < self.min_value:
                    continue
                x = self.origin_x + (cx + 0.5) * self.resolution
                y = self.origin_y + (cy + 0.5) * self.resolution
                p = Point()
                p.x, p.y, p.z = x, y, 0.0
                pts.append(p)
                t = min(max(val / dyn_max, 0.0), 1.0)
                cols.append(self.color_for_layer(t, layer))

        m.points = pts
        m.colors = cols
        marr.markers.append(m)
        pub.publish(marr)

    def color_for_layer(self, t: float, layer: str) -> ColorRGBA:
        c = ColorRGBA()
        c.a = 0.85
        if layer == 'pos':
            c.r, c.g, c.b = 0.0, 0.2 + 0.8 * t, 0.0       # green
            return c
        if layer == 'neg':
            c.r, c.g, c.b = 0.2 + 0.8 * t, 0.0, 0.2 + 0.8 * t  # magenta
            return c
        # combined heat: blue->cyan->green->yellow->red
        if t < 0.25:
            u = t / 0.25;  c.r, c.g, c.b = 0.0, u, 1.0
        elif t < 0.5:
            u = (t - 0.25) / 0.25;  c.r, c.g, c.b = 0.0, 1.0, 1.0 - u
        elif t < 0.75:
            u = (t - 0.5) / 0.25;   c.r, c.g, c.b = u, 1.0, 0.0
        else:
            u = (t - 0.75) / 0.25;  c.r, c.g, c.b = 1.0, 1.0 - u, 0.0
        return c

    def periodic_tick(self):
        # Re-publish the cached grids (latched) so late RViz still gets them
        if self.publish_grids:
            now = self.get_clock().now().to_msg()
            def bump(g):
                if g is None: return None
                g.header.stamp = now
                return g
            if self.last_pos_grid: self.grid_pos_pub.publish(bump(self.last_pos_grid))
            if self.last_neg_grid: self.grid_neg_pub.publish(bump(self.last_neg_grid))
            if self.last_comb_grid: self.grid_comb_pub.publish(bump(self.last_comb_grid))

        # Publish marker layers
        if not self.publish_markers:
            return
        if self.map_pos is not None:
            self.publish_marker_layer(self.map_pos, self.mark_pos_pub, 'pos')
        if self.map_neg is not None:
            self.publish_marker_layer(self.map_neg, self.mark_neg_pub, 'neg')
        if (self.map_pos is not None) and (self.map_neg is not None):
            comb = np.maximum(self.map_pos - self.map_neg, 0.0)
            self.publish_marker_layer(comb, self.mark_comb_pub, 'combined')

def main(args=None):
    rclpy.init(args=args)
    node = ACOVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
