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

        # --- Parameters (must match pheromone_map_node) ---
        self.declare_parameter('width', 100)
        self.declare_parameter('height', 100)
        self.declare_parameter('resolution', 1.0)
        self.declare_parameter('origin_x', -50.0)
        self.declare_parameter('origin_y', -50.0)
        self.declare_parameter('frame_id', 'map')

        # Visualization params
        self.declare_parameter('visualization_rate', 2.0)    # Hz
        self.declare_parameter('grid_skip', 4)               # subsample for markers
        self.declare_parameter('clip_max', 100.0)            # clamp pheromone for color scaling
        self.declare_parameter('invert', False)              # invert colors
        self.declare_parameter('min_value', 0.1)             # marker threshold
        self.declare_parameter('cube_height_scale', 2.0)     # meters at max
        self.declare_parameter('publish_grid', True)         # OccupancyGrid for RViz "Map"
        self.declare_parameter('publish_markers', True)      # CUBE_LIST markers for 3D look

        # Read params
        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        self.resolution = float(self.get_parameter('resolution').value)
        self.origin_x = float(self.get_parameter('origin_x').value)
        self.origin_y = float(self.get_parameter('origin_y').value)
        self.frame_id = self.get_parameter('frame_id').value

        self.viz_rate = float(self.get_parameter('visualization_rate').value)
        self.grid_skip = int(self.get_parameter('grid_skip').value)
        self.clip_max = float(self.get_parameter('clip_max').value)
        self.invert = bool(self.get_parameter('invert').value)
        self.min_value = float(self.get_parameter('min_value').value)
        self.cube_height_scale = float(self.get_parameter('cube_height_scale').value)
        self.publish_grid = bool(self.get_parameter('publish_grid').value)
        self.publish_markers = bool(self.get_parameter('publish_markers').value)

        # Internal state
        self.pheromone = None

        # QoS: make the grid latched so RViz gets the last one on connect
        qos_grid = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # Publishers
        self.grid_pub = self.create_publisher(OccupancyGrid, '/pheromone_grid', qos_grid)
        self.marker_pub = self.create_publisher(MarkerArray, '/aco_visualization', 10)

        # Subscriber (plain queue is fine)
        self.sub = self.create_subscription(Float32MultiArray, '/pheromone_map', self.map_cb, 10)

        # Timer for markers (grid publishes on every map update)
        self.timer = self.create_timer(1.0 / max(self.viz_rate, 1e-3), self.publish_markers_tick)

        # Static map info
        self.meta = MapMetaData()
        self.meta.resolution = self.resolution
        self.meta.width = self.width
        self.meta.height = self.height
        self.meta.origin = Pose()
        self.meta.origin.position.x = self.origin_x
        self.meta.origin.position.y = self.origin_y
        self.meta.origin.orientation.w = 1.0

        self.get_logger().info(
            f'Visualizer ready: {self.width}x{self.height}@{self.resolution}m in frame "{self.frame_id}"'
        )

    # ---------- Callbacks ----------

    def map_cb(self, msg: Float32MultiArray):
        arr = np.array(msg.data, dtype=np.float32)
        expected = self.width * self.height
        if arr.size != expected:
            self.get_logger().warn(f'Array size {arr.size} != {expected} (width*height)')
            return

        self.pheromone = arr.reshape((self.height, self.width))

        # Immediately publish OccupancyGrid (cheap, and RViz "Map" updates smoothly)
        if self.publish_grid:
            self.publish_grid_msg(self.pheromone)

    def publish_grid_msg(self, mat: np.ndarray):
        # Scale to 0..100 int8 for OccupancyGrid
        clip_max = max(self.clip_max, 1e-6)
        scaled = np.clip(mat / clip_max, 0.0, 1.0)
        if self.invert:
            scaled = 1.0 - scaled

        occ = (scaled * 100.0).astype(np.int8)

        grid = OccupancyGrid()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = self.frame_id
        grid.info = self.meta
        # If you want to flip vertically depending on origin conventions, uncomment:
        # occ = np.flipud(occ)
        grid.data = occ.flatten().tolist()

        self.grid_pub.publish(grid)

    def publish_markers_tick(self):
        if not self.publish_markers:
            return
        if self.pheromone is None:
            return

        marker_array = MarkerArray()

        # Clean previous markers
        mdel = Marker()
        mdel.header.frame_id = self.frame_id
        mdel.action = Marker.DELETEALL
        marker_array.markers.append(mdel)

        # Build one efficient CUBE_LIST marker with per-vertex colors
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'pheromone_grid'
        m.id = 0
        m.type = Marker.CUBE_LIST
        m.action = Marker.ADD

        # Each cube has same x/y size; z (height) is via scale.z
        m.scale.x = self.resolution * self.grid_skip
        m.scale.y = self.resolution * self.grid_skip
        m.scale.z = self.resolution  # base height; actual value is encoded via color alpha or we just scale z below

        # We’ll encode value by BOTH height and color; set base height proportional too
        pts = []
        cols = []

        # Determine dynamic max for contrast (fallback to clip_max)
        dyn_max = float(np.max(self.pheromone))
        if dyn_max < 0.1:
            dyn_max = self.clip_max

        for cy in range(0, self.height, self.grid_skip):
            for cx in range(0, self.width, self.grid_skip):
                val = float(self.pheromone[cy, cx])
                if val < self.min_value:
                    continue

                x, y = self.cell_to_world(cx, cy)
                p = Point()
                p.x, p.y, p.z = x, y, 0.0  # cube centers on ground
                pts.append(p)

                color = self.value_to_color(val, dyn_max)
                cols.append(color)

        m.points = pts
        m.colors = cols

        # Optional: scale.z modulation via max value (visual height effect).
        # Since CUBE_LIST uses one scale for all cubes, we encode height only via color by default.
        # If you want real per-cube height, you’d need multiple markers or a mesh. Keeping it simple here.

        marker_array.markers.append(m)
        self.marker_pub.publish(marker_array)

    # ---------- Helpers ----------

    def cell_to_world(self, cx, cy):
        x = self.origin_x + (cx + 0.5) * self.resolution
        y = self.origin_y + (cy + 0.5) * self.resolution
        return x, y

    def value_to_color(self, value, max_value):
        # Normalize
        t = min(max(value / max(max_value, 1e-6), 0.0), 1.0)
        if self.invert:
            t = 1.0 - t

        # Blue -> Cyan -> Green -> Yellow -> Red
        c = ColorRGBA()
        c.a = 0.8
        if t < 0.25:
            u = t / 0.25
            c.r, c.g, c.b = 0.0, u, 1.0
        elif t < 0.5:
            u = (t - 0.25) / 0.25
            c.r, c.g, c.b = 0.0, 1.0, 1.0 - u
        elif t < 0.75:
            u = (t - 0.5) / 0.25
            c.r, c.g, c.b = u, 1.0, 0.0
        else:
            u = (t - 0.75) / 0.25
            c.r, c.g, c.b = 1.0, 1.0 - u, 0.0
        return c


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
