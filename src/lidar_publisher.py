"""
Replays nuScenes lidar scans as a ROS2 PointCloud2 topic stream.
Reads frames sequentially from the nuScenes mini split and publishes at 10Hz.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

DATAROOT = '/data/nuscenes'
PUBLISH_HZ = 10.0


class LidarPublisher(Node):
    def __init__(self):
        super().__init__('lidar_publisher')
        self.publisher = self.create_publisher(PointCloud2, '/lidar/points', 1)

        self.get_logger().info('Loading nuScenes...')
        self.nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)
        self.frames = self._collect_lidar_frames()
        self.frame_idx = 0
        self.get_logger().info(f'Loaded {len(self.frames)} lidar frames')

        self.timer = self.create_timer(1.0 / PUBLISH_HZ, self.publish_frame)

    def _collect_lidar_frames(self):
        frames = []
        for sample in self.nusc.sample:
            token = sample['data']['LIDAR_TOP']
            sample_data = self.nusc.get('sample_data', token)
            path = self.nusc.dataroot + '/' + sample_data['filename']
            frames.append(path)
        return frames

    def publish_frame(self):
        path = self.frames[self.frame_idx % len(self.frames)]
        pc = LidarPointCloud.from_file(path)

        points = pc.points[:3, :].T.astype(np.float32)  # (N, 3) x, y, z
        n_points = points.shape[0]

        msg = PointCloud2()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'lidar_top'
        msg.height = 1
        msg.width = n_points
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * n_points
        msg.data = points.tobytes()

        self.publisher.publish(msg)
        self.get_logger().info(
            f'Published frame {self.frame_idx % len(self.frames) + 1}'
            f'/{len(self.frames)} — {n_points} points'
        )
        self.frame_idx += 1


def main():
    rclpy.init()
    node = LidarPublisher()
    rclpy.spin(node)


main()
