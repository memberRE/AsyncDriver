import math
from typing import List

import numpy as np
from scipy.interpolate import interp1d
from nuplan.common.actor_state.state_representation import StateSE2

from enum import Enum, auto
# from data_generation.questions import DirectionCommand
def state_se2_to_array(state_se2: StateSE2):
    return np.array([state_se2.x, state_se2.y, state_se2.heading], dtype=np.float64)
state_se2_to_array_vectorize = np.vectorize(state_se2_to_array, signature="()->(3)")
    
# from .transition_utils import DirectionDecision
    

def clip_centerline_by_distance(points, distance=120.0):
    if not isinstance(points, np.ndarray):
        points = state_se2_to_array_vectorize(points)
    points = points[:,:2]
    deltas = np.diff(points, axis=0)
    distances = np.sqrt((deltas ** 2).sum(axis=1))
    cumulative_distances = np.zeros(points.shape[0])
    cumulative_distances[1:] = np.cumsum(distances)
    idx = np.where(cumulative_distances < distance)[0].max()
    return points[:idx+1], cumulative_distances[-1]

def clip_centerline_by_ego_pose(points, relative_ego_center ,relative_ego_rear):
    if not isinstance(points, np.ndarray):
        points = state_se2_to_array_vectorize(points)
    if not isinstance(relative_ego_center, np.ndarray):
        relative_ego_center = state_se2_to_array_vectorize(relative_ego_center)
        relative_ego_rear = state_se2_to_array_vectorize(relative_ego_rear)
    deltas = abs((points - relative_ego_center)[:, :-1])
    sum_distances = np.sum(deltas, axis=-1)
    ego_idx = np.argmin(sum_distances)
    assert (relative_ego_center[0,0]-relative_ego_rear[0,0])>0, "center is not ahead of rear"
    try:
        lane_heading_x = ((points[ego_idx+1][0]-points[ego_idx][0])>0)
    except:
        return None, None, None
    if lane_heading_x:
        cliped_centerline = points[ego_idx:]
    else:
        if ego_idx==0:
            return None, None, None
        cliped_centerline = np.flip(points[:ego_idx,:], axis=0)
    return cliped_centerline, ego_idx, lane_heading_x


def adaptive_resample(points, target_num_points=10):
    if not isinstance(points, np.ndarray):
        points = state_se2_to_array_vectorize(points)
    diff1 = np.diff(points, axis=0)
    diff2 = np.diff(diff1, axis=0)
    curvature = np.linalg.norm(diff2, axis=1)


    min_interval = len(points) // target_num_points
    intervals = np.full(len(curvature), fill_value=min_interval)
    high_curvature_indices = np.where(curvature > np.percentile(curvature, 75))[0]
    intervals[high_curvature_indices] = max(1, min_interval // 2)

    resampled_points = []
    i = 0
    while i < len(points):
        resampled_points.append(points[i])
        interval = intervals[min(i, len(intervals) - 1)]
        i += interval

    return np.around(np.array(resampled_points), decimals=2)

def equidistant_interpolation(waypoint, num_point):
    points = np.array(waypoint[:,:2])
    # N = points.shape[0]
    N = num_point
    deltas = np.diff(points, axis=0)
    distances = np.sqrt((deltas ** 2).sum(axis=1))
    cumulative_distances = np.zeros(points.shape[0])
    cumulative_distances[1:] = np.cumsum(distances)
    fx = interp1d(cumulative_distances, points[:, 0], kind='linear')
    fy = interp1d(cumulative_distances, points[:, 1], kind='linear')
    new_distances = np.linspace(0, cumulative_distances[-1], N)
    new_x = fx(new_distances)
    new_y = fy(new_distances)
    new_waypoint = np.column_stack((new_x, new_y))
    if isinstance(waypoint, List):
        new_waypoint = new_waypoint.tolist()
        for i, p in enumerate(new_waypoint):
            new_waypoint[i] = (round(p[0], 2) , round(p[1], 2))
    return new_waypoint

def resample_by_distance_interval(centerline, dist_ls):
    points = np.array(centerline.waypoints[:,:2])
    deltas = np.diff(points, axis=0)
    distances = np.sqrt((deltas ** 2).sum(axis=1))
    # TODO: cumulative_distances是否单增
    cumulative_distances = np.zeros(points.shape[0])
    cumulative_distances[1:] = np.cumsum(distances)
    fx = interp1d(cumulative_distances, points[:, 0], kind='linear')
    fy = interp1d(cumulative_distances, points[:, 1], kind='linear')
    try:
        new_x = fx(dist_ls)
        new_y = fy(dist_ls)
    except:
        return None, cumulative_distances
    new_waypoint = np.column_stack((new_x, new_y))
    # new_waypoint = np.around(new_waypoint, decimals=2)
    return new_waypoint, None

def get_filter_parameters(num_scenarios_per_type=None, limit_total_scenarios=None, shuffle=None, scenario_tokens=None, log_names=None):
    # nuplan challenge
    scenario_types = None

    scenario_tokens = None              # List of scenario tokens to include
    log_names = None                     # Filter scenarios by log names
    map_names = None                     # Filter scenarios by map names

    num_scenarios_per_type               # Number of scenarios per type
    limit_total_scenarios                # Limit total scenarios (float = fraction, int = num) - this filter can be applied on top of num_scenarios_per_type
    timestamp_threshold_s = None          # Filter scenarios to ensure scenarios have more than `timestamp_threshold_s` seconds between their initial lidar timestamps
    ego_displacement_minimum_m = None    # Whether to remove scenarios where the ego moves less than a certain amount

    expand_scenarios = False           # Whether to expand multi-sample scenarios to multiple single-sample scenarios
    remove_invalid_goals = True         # Whether to remove scenarios where the mission goal is invalid
    shuffle                             # Whether to shuffle the scenarios

    ego_start_speed_threshold = None     # Limit to scenarios where the ego reaches a certain speed from below
    ego_stop_speed_threshold = None      # Limit to scenarios where the ego reaches a certain speed from above
    speed_noise_tolerance = None         # Value at or below which a speed change between two timepoints should be ignored as noise.

    return scenario_types, scenario_tokens, log_names, map_names, num_scenarios_per_type, limit_total_scenarios, timestamp_threshold_s, ego_displacement_minimum_m, \
           expand_scenarios, remove_invalid_goals, shuffle, ego_start_speed_threshold, ego_stop_speed_threshold, speed_noise_tolerance
    

if __name__ == '__main__':
    pass