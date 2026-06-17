"""
Subscribes to /lidar/points, runs voxel downsampling + DBSCAN clustering,
and publishes detected obstacles to /obstacles/alerts and /obstacles/markers.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
from sklearn.cluster import DBSCAN

ALERT_DISTANCE_M = 10.0
VOXEL_SIZE = 0.3
DBSCAN_EPSILON = 1.5
DBSCAN_MIN_SAMPLES = 5
MIN_CLUSTER_POINTS = 5
MAX_CLUSTER_POINTS = 2000
GROUND_Z_THRESHOLD = -1.0
MIN_HORIZ_DIST = 2.5


def voxel_downsample(points, voxel_size):
    indices = np.floor(points / voxel_size).astype(int)
    unique_voxels = np.unique(indices, axis=0)
    return (unique_voxels + 0.5) * voxel_size


class ObstacleDetector(Node):
    def __init__(self):
        super().__init__('obstacle_detector')
        self.subscription = self.create_subscription(
            PointCloud2, '/lidar/points', self.callback, 1)
        self.alert_publisher = self.create_publisher(String, '/obstacles/alerts', 10)
        self.marker_publisher = self.create_publisher(MarkerArray, '/obstacles/markers', 10)
        self.get_logger().info(
            f'Obstacle detector running — alerting on objects within {ALERT_DISTANCE_M}m'
        )

    def callback(self, msg):
        n_points = msg.width
        raw = np.frombuffer(msg.data, dtype=np.float32).reshape(n_points, 3).copy()

        # Filter ground plane
        raw = raw[raw[:, 2] > GROUND_Z_THRESHOLD]

        # Filter car body / near noise
        horiz_dist = np.sqrt(raw[:, 0]**2 + raw[:, 1]**2)
        raw = raw[horiz_dist > MIN_HORIZ_DIST]

        if len(raw) == 0:
            return

        # Voxel downsample
        downsampled = voxel_downsample(raw, VOXEL_SIZE)

        # DBSCAN clustering
        labels = DBSCAN(
            eps=DBSCAN_EPSILON,
            min_samples=DBSCAN_MIN_SAMPLES
        ).fit_predict(downsampled)

        unique_labels = set(labels)
        unique_labels.discard(-1)

        obstacles = []
        for label in unique_labels:
            cluster = downsampled[labels == label]
            if MIN_CLUSTER_POINTS <= len(cluster) <= MAX_CLUSTER_POINTS:
                centroid = cluster.mean(axis=0)
                dist = np.sqrt(centroid[0]**2 + centroid[1]**2)
                obstacles.append((dist, centroid, len(cluster)))

        obstacles.sort(key=lambda x: x[0])

        self.get_logger().debug(
            f'Detected {len(obstacles)} obstacles | '
            f'Raw points: {n_points} → Downsampled: {len(downsampled)}'
        )

        # Build marker array
        marker_array = MarkerArray()

        # First marker: DELETEALL clears every marker from the previous frame
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        for i, (dist, centroid, n) in enumerate(obstacles):
            marker = Marker()
            marker.header.frame_id = 'lidar_top'
            marker.header.stamp = msg.header.stamp
            marker.ns = 'obstacles'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            # Position at cluster centroid
            marker.pose.position.x = float(centroid[0])
            marker.pose.position.y = float(centroid[1])
            marker.pose.position.z = float(centroid[2])
            marker.pose.orientation.w = 1.0

            # Size scales with point count — bigger cluster = bigger marker
            size = min(max(n / 50.0, 0.5), 3.0)
            marker.scale.x = size
            marker.scale.y = size
            marker.scale.z = size

            # Red if within alert distance, green if beyond
            marker.color.a = 0.8
            if dist < ALERT_DISTANCE_M:
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            else:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0

            # Marker disappears after 60s if not refreshed — prevents stale markers
            marker.lifetime.sec = 60
            marker.lifetime.nanosec = 0

            marker_array.markers.append(marker)

            # Publish text alert for close obstacles
            if dist < ALERT_DISTANCE_M:
                alert = String()
                alert.data = (
                    f'OBSTACLE {dist:.1f}m '
                    f'[x={centroid[0]:.1f}, y={centroid[1]:.1f}] '
                    f'({n} pts)'
                )
                self.alert_publisher.publish(alert)
                self.get_logger().warn(alert.data)

        self.marker_publisher.publish(marker_array)


def main():
    rclpy.init()
    node = ObstacleDetector()
    rclpy.spin(node)


main()
