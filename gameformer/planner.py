import os
import sys

from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import math
import time
import logging
import matplotlib.pyplot as plt
from shapely import Point, LineString
from planner_utils import *
from obs_adapter import *
from predictor import GameFormer
from state_lattice_planner import LatticePlanner

from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

logging.basicConfig(level=logging.INFO)

class Planner(AbstractPlanner):
    def __init__(self, device=None, disable_refpath=False):
        self._max_path_length = MAX_LEN # [m]
        self._future_horizon = T # [s] 
        self._step_interval = DT # [s]
        self._target_speed = 13.0 # [m/s]
        self._N_points = int(T/DT)

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        self._device = device
        self.disable_refpath = disable_refpath
        logging.error(f'Using device: {self._device}')
        if self.disable_refpath:
            logging.info('disable ref path --------------------------------------------------------------')
    
    def __getstate__(self):
        # Get a dictionary of the object's state
        state = self.__dict__.copy()
        # Remove the model attribute from the state dictionary
        state.pop('_model', None)
        state.pop('_trajectory_planner', None)
        return state

    def name(self) -> str:
        return "GameFormer Planner"
    
    def observation_type(self):
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization):
        self._map_api = initialization.map_api
        self._goal = initialization.mission_goal
        self._route_roadblock_ids = initialization.route_roadblock_ids
        self._initialize_route_plan(self._route_roadblock_ids)
        self._initialize_model()
        self._trajectory_planner = TrajectoryPlanner(self._device)
        self._path_planner = LatticePlanner(self._candidate_lane_edge_ids, self._max_path_length)

    def _initialize_model(self):
        self._model = GameFormer(3, 2, 6, 20)
        self._model.load_state_dict(torch.load(current_dir+'/model_epoch_50_valADE_1.1328.pth', map_location=self._device))
        self._model.to(self._device)
        self._model.eval()
        
    def _initialize_route_plan(self, route_roadblock_ids):
        self._route_roadblocks = []
        self._route_roadblock_dict = {}

        for id_ in route_roadblock_ids:
            block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            self._route_roadblocks.append(block)
            self._route_roadblock_dict[block.id] = block

        self._candidate_lane_edge_ids = [
            edge.id for block in self._route_roadblocks if block for edge in block.interior_edges
        ]

    def _get_reference_path(self, ego_state, traffic_light_data, observation):
        # Get starting block
        starting_block = None
        min_target_speed = 3
        max_target_speed = 15
        cur_point = (ego_state.rear_axle.x, ego_state.rear_axle.y)
        closest_distance = math.inf

        for block in self._route_roadblocks:
            for edge in block.interior_edges:
                distance = edge.polygon.distance(Point(cur_point))
                if distance < closest_distance:
                    starting_block = block
                    closest_distance = distance

            if np.isclose(closest_distance, 0):
                break
            
        # In case the ego vehicle is not on the route, return None
        if closest_distance > 7:
            return None

        # Get reference path, handle exception
        try:
            ref_path = self._path_planner.plan(ego_state, starting_block, observation, traffic_light_data)
        except:
            ref_path = None

        if ref_path is None:
            logging.error('No reference path found')
            return None

        # Annotate red light to occupancy
        occupancy = np.zeros(shape=(ref_path.shape[0], 1))
        for data in traffic_light_data:
            id_ = str(data.lane_connector_id)
            if data.status == TrafficLightStatusType.RED and id_ in self._candidate_lane_edge_ids:
                lane_conn = self._map_api.get_map_object(id_, SemanticMapLayer.LANE_CONNECTOR)
                conn_path = lane_conn.baseline_path.discrete_path
                conn_path = np.array([[p.x, p.y] for p in conn_path])
                red_light_lane = transform_to_ego_frame(conn_path, ego_state)
                occupancy = annotate_occupancy(occupancy, ref_path, red_light_lane)

        # Annotate max speed along the reference path
        target_speed = starting_block.interior_edges[0].speed_limit_mps or self._target_speed
        target_speed = np.clip(target_speed, min_target_speed, max_target_speed)
        max_speed = annotate_speed(ref_path, target_speed)

        # Finalize reference path
        ref_path = np.concatenate([ref_path, max_speed, occupancy], axis=-1) # [x, y, theta, k, v_max, occupancy]
        if len(ref_path) < MAX_LEN * 10:
            ref_path = np.append(ref_path, np.repeat(ref_path[np.newaxis, -1], MAX_LEN*10-len(ref_path), axis=0), axis=0)
        
        return ref_path.astype(np.float32)

    def _get_prediction(self, features):
        predictions, plan = self._model(features)
        K = len(predictions) // 2 - 1
        final_predictions = predictions[f'level_{K}_interactions'][:, 1:]
        final_scores = predictions[f'level_{K}_scores']
        ego_current = features['ego_agent_past'][:, -1]
        neighbors_current = features['neighbor_agents_past'][:, :, -1]

        return plan, final_predictions, final_scores, ego_current, neighbors_current

    def _constant_velocity_planning(self, ego_state, ref_path):
        ego_vel_x = ego_state.dynamic_car_state.rear_axle_velocity_2d.x
        ego_vel_y = ego_state.dynamic_car_state.rear_axle_velocity_2d.y
        timesteps = np.arange(0, self._future_horizon, self._step_interval)

        if ref_path is not None:
            s = ego_vel_x * timesteps
            i = (s * 10).astype(np.int32)
            i = np.clip(i, 0, len(ref_path)-1)
            plan = ref_path[i, :3]
        else:
            plan = np.zeros((self._N_points, 3))
            plan[:, 0] = ego_vel_x * timesteps
            plan[:, 1] = ego_vel_y * timesteps
            plan[:, 2] = np.arctan2(ego_vel_y, ego_vel_x)

        return plan
    
    def _ml_planning(self, plan, ref_path):
        # Get plan
        plan = plan.cpu().numpy()
        dy = plan[1:, 1] - plan[:-1, 1]
        dx = plan[1:, 0] - plan[:-1, 0]
        dx = np.clip(dx, 1e-2, None)
        heading = np.arctan2(dy, dx)
        heading = np.concatenate([heading, [heading[-1]]])
        plan = np.column_stack([plan, heading])

        # Get plan in the reference path
        if ref_path is not None:
            distance_to_ref = scipy.spatial.distance.cdist(plan[:, :2], ref_path[:, :2])
            i = np.argmin(distance_to_ref, axis=1)
            plan = ref_path[i, :3]

        return plan
    
    def _plan(self, ego_state, history, traffic_light_data, observation):
        # start time
        start_time = time.perf_counter()
      
        # Construct input features
        features = observation_adapter(history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device)
        feature_construct_time = time.perf_counter() - start_time
        # if feature_construct_time > 0.5:
        #     logging.error(f'Feature construction time: {feature_construct_time:.3f} s')
        #     plan = self._constant_velocity_planning(ego_state, None)
        #     states = transform_predictions_to_states(plan, history.ego_states, self._future_horizon, DT)
        #     trajectory = InterpolatedTrajectory(states)
        #
        #     return trajectory

        # Get reference path
        if self.disable_refpath:
            ref_path = None
        else:
            ref_path = self._get_reference_path(ego_state, traffic_light_data, observation)
        find_path_time = time.perf_counter() - start_time
        # if find_path_time > 0.5:
        #     logging.error(f'Find path time: {find_path_time:.3f} s')
        #     plan = self._constant_velocity_planning(ego_state, ref_path)
        #     states = transform_predictions_to_states(plan, history.ego_states, self._future_horizon, DT)
        #     trajectory = InterpolatedTrajectory(states)
        #
        #     return trajectory

        # Infer prediction model
        with torch.no_grad():
            plan, predictions, pred_scores, ego_state_transformed, neighbors_state_transformed = self._get_prediction(features)

        prediction_time = time.perf_counter() - start_time
        # if prediction_time > 0.5:
        #     logging.error(f'Prediction time: {prediction_time:.3f} s')
        #     plan = self._ml_planning(plan[0], ref_path)
        #     states = transform_predictions_to_states(plan, history.ego_states, self._future_horizon, DT)
        #     trajectory = InterpolatedTrajectory(states)
        #
        #     return trajectory

        # Trajectory planning
        with torch.no_grad():
            plan = self._trajectory_planner.plan(ego_state, ego_state_transformed, neighbors_state_transformed, 
                                                 predictions, plan, pred_scores, ref_path, observation)
            
        states = transform_predictions_to_states(plan, history.ego_states, self._future_horizon, DT)
        trajectory = InterpolatedTrajectory(states)

        return trajectory
    
    def compute_planner_trajectory(self, current_input: PlannerInput):
        s = time.time()
        iteration = current_input.iteration.index
        history = current_input.history
        traffic_light_data = list(current_input.traffic_light_data)
        ego_state, observation = history.current_state
        try:
            trajectory = self._plan(ego_state, history, traffic_light_data, observation)
        except:
            plan = self._constant_velocity_planning(ego_state, None)
            states = transform_predictions_to_states(plan, history.ego_states, self._future_horizon, DT)
            trajectory = InterpolatedTrajectory(states)

        self._compute_trajectory_runtimes.append(time.time() - s)
        logging.error(f'Iteration {iteration}: {time.time() - s:.3f} s')

        return trajectory
