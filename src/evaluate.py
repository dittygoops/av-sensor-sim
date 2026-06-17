"""
Offline evaluation of obstacle detector against nuScenes ground truth.
Computes precision, recall, and F1 per object category using BEV IoU matching.

Usage:
  python evaluate.py                   # evaluate all ranges
  python evaluate.py --range 30        # evaluate only within 30m
  python evaluate.py --ranges 10 20 30 50  # print a table across multiple ranges
"""
import argparse
import numpy as np
from scipy.optimize import linear_sum_assignment
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from sklearn.cluster import DBSCAN

DATAROOT = '/Users/apgupta/Documents/Coding/new/nuScenes/v1.0-mini'
IOU_THRESHOLD = 0.5
CATEGORIES = ['vehicle.car', 'human.pedestrian.adult', 'vehicle.bicycle']

# Detection parameters — must match obstacle_detector.py
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


def cluster_points(points):
    """Run voxel downsample + DBSCAN, return list of clusters (each a numpy array)."""
    points = points[points[:, 2] > GROUND_Z_THRESHOLD]
    horiz_dist = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    points = points[horiz_dist > MIN_HORIZ_DIST]

    if len(points) == 0:
        return []

    downsampled = voxel_downsample(points, VOXEL_SIZE)
    labels = DBSCAN(eps=DBSCAN_EPSILON, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(downsampled)

    clusters = []
    for label in set(labels):
        if label == -1:
            continue
        cluster = downsampled[labels == label]
        if MIN_CLUSTER_POINTS <= len(cluster) <= MAX_CLUSTER_POINTS:
            clusters.append(cluster)
    return clusters


def cluster_to_bev_aabb(cluster):
    """Fit 2D BEV AABB around cluster points. Returns (min_xy, max_xy)."""
    return cluster[:, :2].min(axis=0), cluster[:, :2].max(axis=0)


def global_to_lidar(center_global, ego_pose, cal_sensor):
    """Transform a point from global frame → ego frame → lidar frame."""
    # Global → ego
    ego_rot = Quaternion(ego_pose['rotation']).rotation_matrix
    ego_trans = np.array(ego_pose['translation'])
    center_ego = ego_rot.T @ (center_global - ego_trans)

    # Ego → lidar
    lidar_rot = Quaternion(cal_sensor['rotation']).rotation_matrix
    lidar_trans = np.array(cal_sensor['translation'])
    center_lidar = lidar_rot.T @ (center_ego - lidar_trans)
    return center_lidar


def gt_box_to_bev_aabb(ann, ego_pose, cal_sensor):
    """
    Convert nuScenes annotation to a 2D BEV AABB in lidar frame.
    Properly rotates the 4 ground-plane corners before taking min/max.
    Returns (min_xy, max_xy) as 2-element arrays.
    """
    W, L, _ = ann['size']
    obj_rot = Quaternion(ann['rotation'])
    center_global = np.array(ann['translation'])

    # 4 corners in object frame (x=forward=length, y=left=width)
    corners_obj = np.array([
        [ L/2,  W/2, 0],
        [ L/2, -W/2, 0],
        [-L/2, -W/2, 0],
        [-L/2,  W/2, 0],
    ])

    # Rotate corners to global frame, then translate
    corners_global = np.array([obj_rot.rotate(c) for c in corners_obj]) + center_global

    # Transform each corner to lidar frame
    corners_lidar = np.array([global_to_lidar(c, ego_pose, cal_sensor) for c in corners_global])

    min_xy = corners_lidar[:, :2].min(axis=0)
    max_xy = corners_lidar[:, :2].max(axis=0)
    return min_xy, max_xy


def iou_bev(box_a, box_b):
    """Compute 2D Bird's Eye View IoU between two BEV AABBs. Each box is (min_xy, max_xy)."""
    min_a, max_a = box_a
    min_b, max_b = box_b

    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    inter_dims = np.maximum(0, inter_max - inter_min)
    inter_area = inter_dims[0] * inter_dims[1]

    if inter_area == 0:
        return 0.0

    area_a = np.prod(max_a - min_a)
    area_b = np.prod(max_b - min_b)

    return inter_area / (area_a + area_b - inter_area)


def box_center_dist(box):
    """Return distance from origin to center of a BEV box (min_xy, max_xy)."""
    center = (box[0] + box[1]) / 2.0
    return np.sqrt(center[0]**2 + center[1]**2)


def evaluate_frame(nusc, sample, category_filter=None, max_range=None):
    """
    Run detection on one frame and match against ground truth.
    max_range: if set, only consider GT boxes and detections whose center is within this distance.
    Returns (tp, fp, fn) counts.
    """
    # Load lidar point cloud
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    cal_sensor = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    path = nusc.dataroot + '/' + lidar_data['filename']
    pc = LidarPointCloud.from_file(path)
    points = pc.points[:3, :].T.astype(np.float32)

    # Get ground truth boxes for this frame (transform from global → lidar frame)
    gt_boxes = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        if category_filter and not any(ann['category_name'].startswith(c) for c in category_filter):
            continue
        gt_box = gt_box_to_bev_aabb(ann, ego_pose, cal_sensor)
        if max_range is not None and box_center_dist(gt_box) > max_range:
            continue
        gt_boxes.append(gt_box)

    # Run detection, gate detections by range too
    clusters = cluster_points(points)
    det_boxes = [cluster_to_bev_aabb(c) for c in clusters]
    if max_range is not None:
        det_boxes = [b for b in det_boxes if box_center_dist(b) <= max_range]

    if not gt_boxes or not det_boxes:
        return 0, len(det_boxes), len(gt_boxes)

    # Build IoU matrix (detections x ground truth) using BEV 2D IoU
    iou_matrix = np.zeros((len(det_boxes), len(gt_boxes)))
    for i, det in enumerate(det_boxes):
        for j, gt in enumerate(gt_boxes):
            iou_matrix[i, j] = iou_bev(det, gt)

    # Hungarian assignment — maximize total IoU
    det_indices, gt_indices = linear_sum_assignment(-iou_matrix)

    tp, fp, fn = 0, 0, 0
    matched_gt = set()
    matched_det = set()

    for d, g in zip(det_indices, gt_indices):
        if iou_matrix[d, g] >= IOU_THRESHOLD:
            tp += 1
            matched_gt.add(g)
            matched_det.add(d)

    fp = len(det_boxes) - len(matched_det)
    fn = len(gt_boxes) - len(matched_gt)

    return tp, fp, fn


def run_eval(nusc, max_range=None, verbose=True):
    """Run evaluation over all frames. Returns (precision, recall, f1, tp, fp, fn)."""
    total_tp, total_fp, total_fn = 0, 0, 0
    range_label = f'{max_range}m' if max_range else 'all ranges'

    for i, sample in enumerate(nusc.sample):
        tp, fp, fn = evaluate_frame(nusc, sample, category_filter=CATEGORIES, max_range=max_range)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        if verbose:
            print(f'Frame {i+1:3d}/{len(nusc.sample)} [{range_label}] | TP={tp} FP={fp} FN={fn}')

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, total_tp, total_fp, total_fn


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--range', type=float, metavar='M',
                       help='only evaluate detections/GT within this range in meters')
    group.add_argument('--ranges', type=float, nargs='+', metavar='M',
                       help='print a comparison table across multiple ranges (e.g. --ranges 10 20 30 50)')
    args = parser.parse_args()

    print(f'Loading nuScenes from {DATAROOT}...')
    nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)
    print(f'Categories: {CATEGORIES}  |  IoU threshold: {IOU_THRESHOLD}\n')

    if args.ranges:
        # Multi-range table — suppress per-frame output, just print summary
        print(f'{"Range":>8}  {"Precision":>9}  {"Recall":>7}  {"F1":>6}  {"TP":>6}  {"FP":>6}  {"FN":>6}')
        print('-' * 58)
        for r in sorted(args.ranges):
            p, rec, f1, tp, fp, fn = run_eval(nusc, max_range=r, verbose=False)
            print(f'{r:>7.0f}m  {p:>9.3f}  {rec:>7.3f}  {f1:>6.3f}  {tp:>6}  {fp:>6}  {fn:>6}')
        # Also print unlimited
        p, rec, f1, tp, fp, fn = run_eval(nusc, max_range=None, verbose=False)
        print(f'{"unlimited":>8}  {p:>9.3f}  {rec:>7.3f}  {f1:>6.3f}  {tp:>6}  {fp:>6}  {fn:>6}')
    else:
        max_range = getattr(args, 'range')
        p, rec, f1, tp, fp, fn = run_eval(nusc, max_range=max_range, verbose=True)
        range_label = f'{max_range}m' if max_range else 'unlimited'

        print(f'\n{"="*50}')
        print(f'Results over {len(nusc.sample)} frames  (range: {range_label})')
        print(f'{"="*50}')
        print(f'Total TP: {tp}')
        print(f'Total FP: {fp}')
        print(f'Total FN: {fn}')
        print(f'Precision: {p:.3f}')
        print(f'Recall:    {rec:.3f}')
        print(f'F1:        {f1:.3f}')


main()
