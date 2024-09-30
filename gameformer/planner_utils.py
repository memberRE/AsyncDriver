import scipy
import torch
import numpy as np
import matplotlib.pyplot as plt
try:
    from common_utils import *
    from speed_planner import SpeedPlanner
    from path_planner import PathPlanner
    from smoother import MotionNonlinearSmoother
    from adapter import occupancy_adpter
except:
    from .common_utils import *
    from .speed_planner import SpeedPlanner
    from .path_planner import PathPlanner
    from .smoother import MotionNonlinearSmoother
    from .adapter import occupancy_adpter


#smoother = MotionNonlinearSmoother(int(T/DT), DT)

class TrajectoryPlanner:
    def __init__(self, device='cpu'):
        self.N = int(T/DT)
        self.ts = DT
        self._device = device
        self.path_planner = PathPlanner()
        self.speed_planenr = SpeedPlanner(device)
        
        # init smoother
        #current = np.zeros(4)
        #speed = np.zeros((self.N+1,))
        #plan = np.zeros((self.N+1, 3))
        #smoother.set_reference_trajectory(current, speed, plan)
        #smoother.solve()
    
    def plan(self, ego_state, ego_state_transformed, neighbors_state_transformed, 
             predictions, plan, pred_scores, ref_path, observation):
        # Get the plan from the prediction model
        plan = plan[0].cpu().numpy()
        dy = plan[1:, 1] - plan[:-1, 1]
        dx = plan[1:, 0] - plan[:-1, 0]
        dx = np.clip(dx, 1e-2, None)
        heading = np.arctan2(dy, dx)
        heading = np.concatenate([heading, [heading[-1]]])
        plan = np.column_stack([plan, heading])
        
        # Get the plan in the reference path
        if ref_path is not None:
            distance_to_ref = scipy.spatial.distance.cdist(plan[:, :2], ref_path[:, :2])
            i = np.argmin(distance_to_ref, axis=1)
            plan = ref_path[i, :3]
            s = np.concatenate([[0], i]) * 0.1
            speed = np.diff(s) / DT
        else:
            speed = np.diff(plan[:, :2], axis=0) / DT
            speed = np.linalg.norm(speed, axis=-1)
            speed = np.concatenate([speed, [speed[-1]]])
            
        # Speed planning
        if ref_path is not None and predictions is not None and pred_scores is not None:
            occupancy = occupancy_adpter(predictions[0], pred_scores[0, 1:], neighbors_state_transformed[0], ref_path)
            ego_plan_ds = torch.from_numpy(speed).float().unsqueeze(0).to(self._device)
            ego_plan_s = torch.from_numpy(s).float().unsqueeze(0).to(self._device)
            ego_state_transformed = ego_state_transformed.to(self._device)
            ref_path = torch.from_numpy(ref_path).unsqueeze(0).to(self._device)
            occupancy = torch.from_numpy(occupancy).unsqueeze(0).to(self._device)

            s, speed = self.speed_planenr.plan(ego_state_transformed, ego_plan_ds, ego_plan_s, occupancy, ref_path)
            s = s.squeeze(0).cpu().numpy()
            speed = speed.squeeze(0).cpu().numpy()

            # Convert to Cartesian trajectory
            ref_path = ref_path.squeeze(0).cpu().numpy()
            i = (s * 10).astype(np.int32).clip(0, len(ref_path)-1)
            plan = ref_path[i, :3]

        # Trajectory smoothing
        #current = ego_state_transformed.squeeze(0).cpu().numpy()[:4]
        #plan = np.concatenate([current[None, :3], plan], axis=0)
        #speed = np.concatenate([current[None, 3], speed], axis=0).clip(0, 20)
        #smoother.set_reference_trajectory(current, speed, plan)
        #try:
        #    solution = smoother.solve()
        #    plan = solution.value(smoother.state).T[1:, :3]
        #except:
        #    plan = plan[1:, :3]

        return plan
    
    @staticmethod
    def transform_to_Cartesian_path(path, ref_path):
        frenet_idx = np.array(path[:, 0] * 10, dtype=np.int32)
        frenet_idx = np.clip(frenet_idx, 0, len(ref_path)-1)
        ref_points = ref_path[frenet_idx]
        l = path[frenet_idx, 1]

        cartesian_x = ref_points[:, 0] - l * np.sin(ref_points[:, 2])
        cartesian_y = ref_points[:, 1] + l * np.cos(ref_points[:, 2])
        cartesian_path = np.column_stack([cartesian_x, cartesian_y])

        return cartesian_path


def annotate_occupancy(occupancy, ego_path, red_light_lane):
    ego_path_red_light = scipy.spatial.distance.cdist(ego_path[:, :2], red_light_lane)

    if len(red_light_lane) < 80:
        pass
    else:
        occupancy[np.any(ego_path_red_light < 0.5, axis=-1)] = 1

    return occupancy


def annotate_speed(ref_path, speed_limit):
    speed = np.ones(len(ref_path)) * speed_limit
    
    # get the turning point
    turning_idx = np.argmax(np.abs(ref_path[:, 3]) > 1/10)

    # set speed limit to 3 m/s for turning
    if turning_idx > 0:
        speed[turning_idx:] = 3

    return speed[:, None]


def trajectory_smoothing(trajectory):
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    h = trajectory[:, 2]

    window_length = 15
    x = scipy.signal.savgol_filter(x, window_length=window_length, polyorder=3)
    y = scipy.signal.savgol_filter(y, window_length=window_length, polyorder=3)
    h = scipy.signal.savgol_filter(h, window_length=window_length, polyorder=3)
   
    return np.column_stack([x, y, h])


def wrap_to_pi(theta):
    return (theta+np.pi) % (2*np.pi) - np.pi


def transform_to_ego_frame(path, ego_state):
    ego_x, ego_y, ego_h = ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading
    path_x, path_y = path[:, 0], path[:, 1]
    ego_path_x = np.cos(ego_h) * (path_x - ego_x) + np.sin(ego_h) * (path_y - ego_y)
    ego_path_y = -np.sin(ego_h) * (path_x - ego_x) + np.cos(ego_h) * (path_y - ego_y)
    ego_path = np.stack([ego_path_x, ego_path_y], axis=-1)

    return ego_path
