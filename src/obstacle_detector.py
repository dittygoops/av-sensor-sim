"""
Subscribes to /lidar/points, runs voxel downsampling + DBSCAN clustering,
and publishes detected obstacles to /obstacles/alerts.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
import numpy as np
from sklearn.cluster import DBSCAN

ALERT_DISTANCE_M = 10.0
VOXEL_SIZE = 0.3       # meters per voxel
DBSCAN_EPSILON = 0.75  # max distance between points in same cluster
DBSCAN_MIN_SAMPLES = 5 # minimum points to form a cluster
MIN_CLUSTER_POINTS = 5
MAX_CLUSTER_POINTS = 2000  # filter out ground/walls


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
        self.get_logger().info(
            f'Obstacle detector running — alerting on objects within {ALERT_DISTANCE_M}m'
        )

    def callback(self, msg):
        n_points = msg.width
        raw = np.frombuffer(msg.data, dtype=np.float32).reshape(n_points, 3).copy()

        # Filter ground plane
        raw = raw[raw[:, 2] > -1.5]

        # Filter car body / near noise
        horiz_dist = np.sqrt(raw[:, 0]**2 + raw[:, 1]**2)
        raw = raw[horiz_dist > 1.0]

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
        unique_labels.discard(-1)  # -1 = noise points

        obstacles = []
        for label in unique_labels:
            cluster = downsampled[labels == label]
            if MIN_CLUSTER_POINTS <= len(cluster) <= MAX_CLUSTER_POINTS:
                centroid = cluster.mean(axis=0)
                dist = np.sqrt(centroid[0]**2 + centroid[1]**2)
                obstacles.append((dist, centroid, len(cluster)))

        obstacles.sort(key=lambda x: x[0])

        self.get_logger().info(
            f'Detected {len(obstacles)} obstacles | '
            f'Raw points: {n_points} → Downsampled: {len(downsampled)}'
        )

        for dist, centroid, n in obstacles:
            if dist >= ALERT_DISTANCE_M:
                break
            alert = String()
            alert.data = (
                f'OBSTACLE {dist:.1f}m '
                f'[x={centroid[0]:.1f}, y={centroid[1]:.1f}] '
                f'({n} pts)'
            )
            self.alert_publisher.publish(alert)
            self.get_logger().warn(alert.data)


def main():
    rclpy.init()
    node = ObstacleDetector()
    rclpy.spin(node)


main()
