"""
Subscribes to /lidar/points and detects obstacles within a distance threshold.
Publishes warnings to /obstacles/alerts.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
import numpy as np

ALERT_DISTANCE_M = 10.0


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
        raw = np.frombuffer(msg.data, dtype=np.float32).reshape(n_points, 3)

        # Ignore ground plane (z < -1.5m relative to lidar)
        raw = raw[raw[:, 2] > -1.5]

        distances = np.sqrt(raw[:, 0]**2 + raw[:, 1]**2)  # horizontal distance only

        # Ignore points within 1m of the sensor (car body / sensor noise)
        raw = raw[distances > 1.0]
        distances = distances[distances > 1.0]
        closest_dist = distances.min()
        closest_point = raw[distances.argmin()]

        self.get_logger().info(f'Closest object: {closest_dist:.1f}m at {closest_point}')

        if closest_dist < ALERT_DISTANCE_M:
            alert = String()
            alert.data = (
                f'OBSTACLE at {closest_dist:.1f}m '
                f'[x={closest_point[0]:.1f}, y={closest_point[1]:.1f}]'
            )
            self.alert_publisher.publish(alert)
            self.get_logger().warn(alert.data)


def main():
    rclpy.init()
    node = ObstacleDetector()
    rclpy.spin(node)


main()
