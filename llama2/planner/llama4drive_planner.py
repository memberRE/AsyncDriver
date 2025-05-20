import logging
import time
from typing import Dict, List

import numpy as np
import torch
import math

from shapely import Point

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan_garage.planning.simulation.planner.pdm_planner.utils.route_utils import get_current_roadblock_candidates, \
    remove_route_loops

from gameformer.planner import Planner as BaseGFPlanner
from gameformer.planner_utils import *
from gameformer.obs_adapter import *
from gameformer.state_lattice_planner import LatticePlanner

from llama2.planner.llama4drive import LLAMA2DriveModel
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import RoadBlockGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusType
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses
from nuplan_garage.planning.simulation.planner.pdm_planner.utils.graph_search.bfs_roadblock import (
    BreadthFirstSearchRoadBlock,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.simulation.planner.ml_planner.model_loader import ModelLoader

def get_navigation_by_gt(scenario: AbstractScenario,
                         map_api: AbstractMap,
                         route_roadblock_dict,
                         starting):
    # (path, path_id), path_found
    gt_path = scenario.get_ego_future_trajectory(
        iteration=0, time_horizon=15.0,
    )
    ans_path = [starting]
    ans_path_id = [starting.id]
    ans_valid = [edge.id for edge in starting.outgoing_edges]
    for state in gt_path:
        block, block_ca = get_current_roadblock_candidates(state, map_api, route_roadblock_dict)
        if block.id == ans_path_id[-1]:
            continue
        if block.id not in ans_valid:
            ca_id = []
            ca_b = []

            for edge in block_ca:
                if edge.id in ans_valid:
                    ca_id.append(edge.id)
                    ca_b.append(edge)
            if len(ca_id) == 1:
                block = ca_b[0]
            elif len(ca_id) == 0:
                b_id, dis = map_api.get_distance_to_nearest_map_object(state.rear_axle.point,
                                                                       SemanticMapLayer.ROADBLOCK)
                block = map_api.get_map_object(b_id, SemanticMapLayer.ROADBLOCK)
                bc_id, dis2 = map_api.get_distance_to_nearest_map_object(state.rear_axle.point,
                                                                         SemanticMapLayer.ROADBLOCK_CONNECTOR)
                block_c = map_api.get_map_object(bc_id, SemanticMapLayer.ROADBLOCK_CONNECTOR)
                if b_id in ans_valid:
                    block = block
                elif bc_id in ans_valid:
                    block = block_c
                else:
                    continue
            else:
                continue

        ans_path.append(block)
        ans_path_id.append(block.id)
        ans_valid = [edge.id for edge in block.outgoing_edges]
    return (ans_path, ans_path_id), True

def route_roadblock_correction(
    ego_state: EgoState,
    map_api: AbstractMap,
    route_roadblock_dict: Dict[str, RoadBlockGraphEdgeMapObject],
    search_depth_backward: int = 15,
    search_depth_forward: int = 30,
    scenario: AbstractScenario = None
) -> List[str]:
    """
    Applies several methods to correct route roadblocks.
    :param ego_state: class containing ego state
    :param map_api: map object
    :param route_roadblocks_dict: dictionary of on-route roadblocks
    :param search_depth_backward: depth of forward BFS search, defaults to 15
    :param search_depth_forward:  depth of backward BFS search, defaults to 30
    :return: list of roadblock id's of corrected route
    """
    starting_block, starting_block_candidates = get_current_roadblock_candidates(
        ego_state, map_api, route_roadblock_dict
    )
    starting_block_ids = [roadblock.id for roadblock in starting_block_candidates]

    route_roadblocks = list(route_roadblock_dict.values())
    route_roadblock_ids = list(route_roadblock_dict.keys())

    # Fix 1: when agent starts off-route
    if starting_block.id not in route_roadblock_ids:
        # Backward search if current roadblock not in route
        graph_search = BreadthFirstSearchRoadBlock(
            route_roadblock_ids[0], map_api, forward_search=False
        )
        (path, path_id), path_found = graph_search.search(
            starting_block_ids, max_depth=search_depth_backward
        )

        if path_found:
            route_roadblocks[:0] = path[:-1]
            route_roadblock_ids[:0] = path_id[:-1]

        else:
            # Forward search to any route roadblock
            graph_search = BreadthFirstSearchRoadBlock(
                starting_block.id, map_api, forward_search=True
            )
            (path, path_id), path_found = graph_search.search(
                route_roadblock_ids[:3], max_depth=search_depth_forward
            )

            if not path_found:
                (path, path_id), path_found = get_navigation_by_gt(scenario, map_api, route_roadblock_dict, starting_block)

            end_roadblock_idx = np.argmax(
                np.array(route_roadblock_ids) == path_id[-1]
            )

            route_roadblocks = route_roadblocks[end_roadblock_idx + 1 :]
            route_roadblock_ids = route_roadblock_ids[end_roadblock_idx + 1 :]

            route_roadblocks[:0] = path
            route_roadblock_ids[:0] = path_id

    # Fix 2: check if roadblocks are linked, search for links if not
    roadblocks_to_append = {}
    for i in range(len(route_roadblocks) - 1):
        next_incoming_block_ids = [
            _roadblock.id for _roadblock in route_roadblocks[i + 1].incoming_edges
        ]
        is_incoming = route_roadblock_ids[i] in next_incoming_block_ids

        if is_incoming:
            continue

        graph_search = BreadthFirstSearchRoadBlock(
            route_roadblock_ids[i], map_api, forward_search=True
        )
        (path, path_id), path_found = graph_search.search(
            route_roadblock_ids[i + 1], max_depth=search_depth_forward
        )

        if path_found and path and len(path) >= 3:
            path, path_id = path[1:-1], path_id[1:-1]
            roadblocks_to_append[i] = (path, path_id)

    # append missing intermediate roadblocks
    offset = 1
    for i, (path, path_id) in roadblocks_to_append.items():
        route_roadblocks[i + offset : i + offset] = path
        route_roadblock_ids[i + offset : i + offset] = path_id
        offset += len(path)

    # Fix 3: cut route-loops
    route_roadblocks, route_roadblock_ids = remove_route_loops(
        route_roadblocks, route_roadblock_ids
    )

    return route_roadblock_ids

class LLAMA4DrivePlanner(BaseGFPlanner):
    requires_scenario = True

    def __init__(self,
                 scenario: AbstractScenario, 
                 sub_planner: AbstractPlanner = None,
                 enable_pdm_scorer_in_multirefpath=False,
                 disable_refpath=False, 
                 ins_mode=None,
                 llm_plan=False, 
                 ins_wo_stop=False, 
                 lora_r=8,
                 finetune_model_path=None, 
                 model_name_or_path=None,
                 near_multiple_vehicles=False,
                 short_ins=-1,
                 llm_inf_step=1,
                 model_cfg=None,
                 model_urban: TorchModuleWrapper = None,
                 onnx_model_path=None,
                 tensorrt_model_path=None,
                 inference_model_type=None):
        super().__init__(disable_refpath=disable_refpath)
        if isinstance(model_cfg, list):
            model_cfg = {k:v for d in model_cfg for k,v in d.items()}
        self.ins_mode = ins_mode
        self.llm_plan = llm_plan
        self.ins_wo_stop = ins_wo_stop
        self.lora_r = lora_r
        self.finetune_model_path = finetune_model_path
        self.model_name_or_path = model_name_or_path
        self.near_multiple_vehicles = near_multiple_vehicles
        self.short_ins = short_ins
        self.onnx_model_path = onnx_model_path
        self.tensorrt_model_path = tensorrt_model_path
        self.inference_model_type = inference_model_type
        logging.error(f'Ins mode: {ins_mode}')
        if ins_mode in ['gt', 'plain_ref']:
            ins_mode = None
        model_cfg['ins_mode'] = ins_mode
        model_cfg['ins_wo_stop'] = ins_wo_stop
        model_cfg['lora_r'] = lora_r
        model_cfg['finetune_model_path'] = finetune_model_path
        model_cfg['llm_inf_step'] = llm_inf_step
        model_cfg['lora_r'] = lora_r
        model_cfg['near_multiple_vehicles'] = near_multiple_vehicles
        model_cfg['model_name_or_path'] = model_name_or_path
        model_cfg['onnx_model_path'] = onnx_model_path
        model_cfg['tensorrt_model_path'] = tensorrt_model_path
        model_cfg['inference_model_type'] = inference_model_type
        self._model_cfg = model_cfg
        self.scenario = scenario
        self.sub_planner = sub_planner
        self.enable_pdm_scorer_in_multirefpath = enable_pdm_scorer_in_multirefpath

        if enable_pdm_scorer_in_multirefpath:
            logging.info('PDM scorer is enabled in multi-refpath mode')
            assert sub_planner
        
        
    def name(self) -> str:
        return self.__class__.__name__

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_model', None)
        state.pop('_path_planner', None)
        state.pop('_trajectory_planner', None)
        state.pop('sub_planner', None)
        state.pop('model_urban', None)
        state.pop('_model_loader', None)
        return state

    def initialize(self, initialization: PlannerInitialization):
        super().initialize(initialization)
        if self.sub_planner:
            self.sub_planner.initialize(initialization)
        if self.enable_pdm_scorer_in_multirefpath:
            self._path_planner = LatticePlanner(self._candidate_lane_edge_ids, self._max_path_length, return_all_refpath=True)

    def _initialize_model(self):
        self._model = LLAMA2DriveModel(self._model_cfg)

    def _get_prediction(self, features, ref_path, cur_iter):
        # predictions, plan = self._model(features)
        output = self._model.inference(features, ref_path, cur_iter)
        predictions = output.predictions
        if self.llm_plan:
            plan = output.llm_plan
        else:
            plan = output.plan
        if predictions is not None:
            K = len(predictions) // 2 - 1
            final_predictions = predictions[f'level_{K}_interactions'][:, 1:]
            final_scores = predictions[f'level_{K}_scores']
        else:
            final_predictions = None
            final_scores = None
        try:
            ego_current = features['ego_agent_past'][:, -1]
            neighbors_current = features['neighbor_agents_past'][:, :, -1]
        except:
            ego_current = None
            neighbors_current = None
        return plan, final_predictions, final_scores, ego_current, neighbors_current
    
    def get_ego_agent_future(self, ego_state):
        current_absolute_state = ego_state

        trajectory_absolute_states = self.scenario.get_ego_future_trajectory(
            iteration=self.iteration, num_samples=80, time_horizon=8
        )
        trajectory_absolute_states = [state.rear_axle for state in trajectory_absolute_states]
        logging.error(f'now: {ego_state.rear_axle.x}')
        logging.error(f'future: {trajectory_absolute_states[0].x}')
        # assert ego_state.rear_axle.x == trajectory_absolute_states[0].x
        # assert ego_state.rear_axle.y == trajectory_absolute_states[0].y
        # Get all future poses of the ego relative to the ego coordinate system
        trajectory_relative_poses = convert_absolute_to_relative_poses(
            current_absolute_state.rear_axle, trajectory_absolute_states
        )

        return trajectory_relative_poses

    def _get_multi_refpath(self, ego_state, traffic_light_data, observation):
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
        
        ref_paths = self._path_planner.plan(ego_state, starting_block, observation, traffic_light_data)
        if ref_paths is None:
            return []
        ans = []
        for ref_path, cost in ref_paths:
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

            ans.append((ref_path.astype(np.float32), cost))
        
        return ans


    def _plan(self, ego_state, history, traffic_light_data, observation, cur_iter):
        # start time
        start_time = time.perf_counter()

        # Construct input features
        features = observation_adapter(history, traffic_light_data, self._map_api, self._route_roadblock_ids,
                                       self._device)
        feature_construct_time = time.perf_counter() - start_time

        # Get reference path
        if self.enable_pdm_scorer_in_multirefpath:
            ref_path_set = self._get_multi_refpath(ego_state, traffic_light_data, observation)
            if ref_path_set is None or len(ref_path_set) == 0:
                logging.error('No reference path found')
        else:
            ref_path = self._get_reference_path(ego_state, traffic_light_data, observation)
            if ref_path is None:
                logging.error('No reference path found')

        find_path_time = time.perf_counter() - start_time

        # Infer prediction model
        if self.ins_mode == 'gt':
            ins_path = self.get_ego_agent_future(ego_state)
        elif self.enable_pdm_scorer_in_multirefpath:
            if ref_path_set is None or len(ref_path_set) == 0:
                ins_path = None
            else:
                ins_path = ref_path_set[0][0]
        else:
            if self.short_ins != -1:
                if ref_path is not None:
                    ego_future_poses = ref_path[:,:3]
                    dis_norm = np.linalg.norm(np.diff(ego_future_poses[:, :-1], n=1, axis=0), axis=1)
                    dis_cum = np.cumsum(dis_norm, axis=0)
                    dis_cum_num = np.where(dis_cum<self.short_ins)[0][-1]
                    ins_path = ref_path[:dis_cum_num, :]
                else:
                    ins_path = ref_path
            else:
                ins_path = ref_path

        plan, predictions, pred_scores, ego_state_transformed, neighbors_state_transformed = self._get_prediction(features, ins_path, cur_iter)

        prediction_time = time.perf_counter() - start_time

        # if prediction_time > 0.5:
        #     logging.error(f'Prediction time: {prediction_time:.3f} s')
        #     plan = self._ml_planning(plan[0], ref_path)
        #     states = transform_predictions_to_states(plan, history.ego_states, self._future_horizon, DT)
        #     trajectory = InterpolatedTrajectory(states)
        #
        #     return trajectory

        if self.disable_refpath:
            print('!!!!!!! ref path is disabled !!!!!!!!!!!!')
            ref_path = None

        # Trajectory planning
        with torch.no_grad():
            if self.enable_pdm_scorer_in_multirefpath:
                # plans = []
                max_score = -1
                max_traj = None
                corr_cost = -1
                for ref_path, cost in ref_path_set:
                    plan_r = self._trajectory_planner.plan(ego_state, ego_state_transformed, neighbors_state_transformed,
                                                        predictions, plan, pred_scores, ref_path, observation)
                    states = transform_predictions_to_states(plan_r, history.ego_states, self._future_horizon, 0.1)
                    trajectory = InterpolatedTrajectory(states)
                    _, scores = self.sub_planner.compute_planner_trajectory_just_4_get_score(self.current_input, trajectory)
                    curr_score = np.mean(scores)
                    # plans.append((trajectory, cost, curr_score))
                    if curr_score > max_score or (abs(curr_score - max_score) < 1e-5 and cost < corr_cost):
                        max_score = curr_score
                        max_traj = trajectory
                        corr_cost = cost
                if max_traj is None:
                    plan = self._trajectory_planner.plan(ego_state, ego_state_transformed, neighbors_state_transformed,
                                                 predictions, plan, pred_scores, None, observation)
                    states = transform_predictions_to_states(plan, history.ego_states, self._future_horizon, 0.1)
                    trajectory = InterpolatedTrajectory(states)
                else:
                    logging.info(f"in Iter {self.iteration}:  Max score: {max_score:.3f}, cost: {corr_cost:.3f}/{ref_path_set[0][1]:.3f}")
                    trajectory = max_traj
                
            else:
                plan = self._trajectory_planner.plan(ego_state, ego_state_transformed, neighbors_state_transformed,
                                                 predictions, plan, pred_scores, ref_path, observation)
                states = transform_predictions_to_states(plan, history.ego_states, self._future_horizon, 0.1)
                trajectory = InterpolatedTrajectory(states)

        return trajectory

    def _route_roadblock_correction(self, ego_state: EgoState) -> None:
        """
        Corrects the roadblock route and reloads lane-graph dictionaries.
        :param ego_state: state of the ego vehicle.
        """
        closest_distance = math.inf

        for block in self._route_roadblocks:
            for edge in block.interior_edges:
                distance = edge.polygon.distance(Point((ego_state.rear_axle.x, ego_state.rear_axle.y)))
                if distance < closest_distance:
                    starting_block = block
                    closest_distance = distance

            if np.isclose(closest_distance, 0):
                break

        # In case the ego vehicle is not on the route, return None
        if closest_distance > 7:
            route_roadblock_ids = route_roadblock_correction(
                ego_state, self._map_api, self._route_roadblock_dict, scenario=self.scenario
            )
            # print(route_roadblock_ids)
            self._initialize_route_plan(route_roadblock_ids)

    def compute_planner_trajectory(self, current_input: PlannerInput):
        s = time.time()
        iteration = current_input.iteration.index
        self.iteration = current_input.iteration.index
        history = current_input.history
        traffic_light_data = list(current_input.traffic_light_data)
        ego_state, observation = history.current_state
        self.current_input = current_input
        if iteration == 0:
            old_ids = [b.id for b in self._route_roadblocks]
            # logging.error(f'Old route roadblocks: {old_ids}')
            self._route_roadblock_correction(ego_state)
            new_ids = [b.id for b in self._route_roadblocks]
            logging.error(f'\n New route roadblocks: {new_ids} \n Old route roadblocks: {old_ids}')

        if self.sub_planner:
            pdm_trajectory = self.sub_planner.compute_planner_trajectory(current_input)
        cur_iter = current_input.iteration
        trajectory = self._plan(ego_state, history, traffic_light_data, observation, cur_iter)

        self._compute_trajectory_runtimes.append(time.time() - s)
        logging.error(f'Iteration {iteration}: {time.time() - s:.3f} s')

        return trajectory