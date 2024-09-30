import copy
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.metrics.utils.collision_utils import CollisionType
from nuplan.planning.simulation.observation.idm.utils import (
    is_agent_ahead,
    is_agent_behind,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely import Point, creation

from nuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation import (
    PDMObservation,
)
from nuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import (
    PDMOccupancyMap,
)
from nuplan_garage.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
    ego_is_comfortable,
)
from nuplan_garage.planning.simulation.planner.pdm_planner.scoring.pdm_scorer_utils import (
    get_collision_type,
)
from nuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    coords_array_to_polygon_array,
    state_array_to_coords_array,
)
from nuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    BBCoordsIndex,
    EgoAreaIndex,
    MultiMetricIndex,
    StateIndex,
    WeightedMetricIndex,
)
from nuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

from nuplan.common.geometry.convert import absolute_to_relative_poses
from data_generation.utils import equidistant_interpolation, clip_centerline_by_ego_pose

# constants
# TODO: Add to config
WEIGHTED_METRICS_WEIGHTS = np.zeros(len(WeightedMetricIndex), dtype=np.float64)
WEIGHTED_METRICS_WEIGHTS[WeightedMetricIndex.PROGRESS] = 5.0
WEIGHTED_METRICS_WEIGHTS[WeightedMetricIndex.TTC] = 5.0
WEIGHTED_METRICS_WEIGHTS[WeightedMetricIndex.COMFORTABLE] = 2.0

# TODO: Add to config
DRIVING_DIRECTION_COMPLIANCE_THRESHOLD = 2.0  # [m] (driving direction)
DRIVING_DIRECTION_VIOLATION_THRESHOLD = 6.0  # [m] (driving direction)
STOPPED_SPEED_THRESHOLD = 5e-03  # [m/s] (ttc)
PROGRESS_DISTANCE_THRESHOLD = 0.1  # [m] (progress)

# agent_type_name
agent_type_name = {
    TrackedObjectType.VEHICLE: 'vehicle',
    TrackedObjectType.PEDESTRIAN: 'pedestrian',
    TrackedObjectType.BICYCLE: 'bicycle',
    TrackedObjectType.EGO: 'ego',
    TrackedObjectType.TRAFFIC_CONE: 'traffic cone',
    TrackedObjectType.BARRIER: 'barrier',
    TrackedObjectType.CZONE_SIGN: 'czone sign',
    TrackedObjectType.GENERIC_OBJECT: 'generic object'}


class PDMScorer4LLM:
    """Class to score proposals in PDM pipeline. Re-implements nuPlan's closed-loop metrics."""

    def __init__(self, proposal_sampling: TrajectorySampling):
        """
        Constructor of PDMScorer
        :param proposal_sampling: Sampling parameters for proposals
        """
        self._proposal_sampling = proposal_sampling

        # lazy loaded
        self._initial_ego_state: Optional[EgoState] = None
        self._observation: Optional[PDMObservation] = None
        self._centerline: Optional[PDMPath] = None
        self._route_lane_dict: Optional[Dict[str, LaneGraphEdgeMapObject]] = None
        self._drivable_area_map: Optional[PDMOccupancyMap] = None
        self._map_api: Optional[AbstractMap] = None

        self._num_proposals: Optional[int] = None
        self._states: Optional[npt.NDArray[np.float64]] = None
        self._ego_coords: Optional[npt.NDArray[np.float64]] = None
        self._ego_polygons: Optional[npt.NDArray[np.object_]] = None

        self._ego_areas: Optional[npt.NDArray[np.bool_]] = None

        self._multi_metrics: Optional[npt.NDArray[np.float64]] = None
        self._weighted_metrics: Optional[npt.NDArray[np.float64]] = None
        self._progress_raw: Optional[npt.NDArray[np.float64]] = None

        self._collision_time_idcs: Optional[npt.NDArray[np.float64]] = None
        self._ttc_time_idcs: Optional[npt.NDArray[np.float64]] = None
        
        self.error_dict: Optional[Dict] = None

    def time_to_at_fault_collision(self, proposal_idx: int) -> float:
        """
        Returns time to at-fault collision for given proposal
        :param proposal_idx: index for proposal
        :return: time to infraction
        """
        return (
            self._collision_time_idcs[proposal_idx]
            * self._proposal_sampling.interval_length
        )

    def time_to_ttc_infraction(self, proposal_idx: int) -> float:
        """
        Returns time to ttc infraction for given proposal
        :param proposal_idx: index for proposal
        :return: time to infraction
        """
        return (
            self._ttc_time_idcs[proposal_idx] * self._proposal_sampling.interval_length
        )

    def score_proposals(
        self,
        states: npt.NDArray[np.float64],
        initial_ego_state: EgoState,
        observation: PDMObservation,
        centerline: PDMPath,
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
        drivable_area_map: PDMOccupancyMap,
        map_api: AbstractMap,
    ) -> npt.NDArray[np.float64]:
        """
        Scores proposal similar to nuPlan's closed-loop metrics
        :param states: array representation of simulated proposals
        :param initial_ego_state: ego-vehicle state at current iteration
        :param observation: PDM's observation class
        :param centerline: path of the centerline
        :param route_lane_dict: dictionary containing on-route lanes
        :param drivable_area_map: Occupancy map of drivable are polygons
        :param map_api: map object
        :return: array containing score of each proposal
        """

        # initialize & lazy load class values
        self._reset(
            states,
            initial_ego_state,
            observation,
            centerline,
            route_lane_dict,
            drivable_area_map,
            map_api,
        )

        # fill value ego-area array (used across multiple metrics)
        self._calculate_ego_area()

        error_dict = dict()
        
        # 1. multiplicative metrics
        no_at_fault_collision = self._calculate_no_at_fault_collision()
        driving_direction_compliance = self._calculate_driving_direction_compliance()
        drivable_area_compliance = self._calculate_drivable_area_compliance()

        # 2. weighted metrics
        progress = self._calculate_progress()
        ttc = self._calculate_ttc()
        comfortable = self._calculate_is_comfortable()
        
        error_dict['no_at_fault_collision'] = no_at_fault_collision
        error_dict['driving_direction_compliance'] = driving_direction_compliance
        error_dict['drivable_area_compliance'] = drivable_area_compliance
        error_dict['progress'] = progress
        error_dict['ttc'] = ttc
        error_dict['comfortable'] = comfortable
        
        self.error_dict = error_dict

        return self._aggregate_scores()
    
    def to_error_prompt(self):
        pre_prompt = "After conducting simulation based on the Refined Ego Future Trajectories that you've decided, it was found that the following problems may exist:\n\n"
        
        
        error_prompt_ls = []
        
        if self.error_dict['no_at_fault_collision']['error_bool']:
            no_at_fault_collision_prompt = self._no_at_fault_collision_to_prompt()
            error_prompt_ls.append(no_at_fault_collision_prompt)
            
        if self.error_dict['driving_direction_compliance']['error_bool']:
            driving_direction_compliance_prompt = self._driving_direction_compliance_to_prompt()
            error_prompt_ls.append(driving_direction_compliance_prompt)
        
        if self.error_dict['drivable_area_compliance']['error_bool']:
            drivable_area_compliance_prompt = self._drivable_area_compliance_to_prompt()
            error_prompt_ls.append(drivable_area_compliance_prompt)
            
        # TODO: progress prompt
        if self.error_dict['progress']['error_bool']:
            progress_prompt = self._progress_to_prompt()
            error_prompt_ls.append(progress_prompt)

        if self.error_dict['ttc']['error_bool']:
            # ttc_prompt = self._ttc_to_prompt()
            ttc_prompt = ''
            error_prompt_ls.append(ttc_prompt)
        
        if self.error_dict['comfortable']['error_bool']:
            comfortable_prompt = self._comfortable_to_prompt()
            error_prompt_ls.append(comfortable_prompt)

        
        error_prompt = ''
        for idx, p in enumerate(error_prompt_ls):
            if p == '':
                continue
            error_prompt += '%s. '%str(idx+1)
            error_prompt += p
            error_prompt += '\n\n'
        if error_prompt == '':
            return "After conducting simulation based on the Refined Ego Future Trajectories that you've decided, it was found that there was no problem with the trajectory you predicted. Please consider on this basis whether you can DO reasonably accelerate. DO output the waypoint first and it should maintain the same format! (16 points, each point is represented by (x, y). The time interval between two points is 0.5s.)"
        
        pro_prompt = '\n'
        pro_prompt += 'Please make revisions to the trajactory based on the previously entered environmental information, the first prediction results, and the above issues. DO output the waypoint first and it should maintain the same format! (16 points, each point is represented by (x, y). The time interval between two points is 0.5s.)'

        # if len(error_prompt_ls) == 0:
        #     return None
        return pre_prompt + error_prompt + pro_prompt

    def _aggregate_scores(self) -> npt.NDArray[np.float64]:
        """
        Aggregates metrics with multiplicative and weighted average.
        :return: array containing score of each proposal
        """

        # accumulate multiplicative metrics
        multiplicate_metric_scores = self._multi_metrics.prod(axis=0)

        # normalize and fill progress values
        raw_progress = self._progress_raw * multiplicate_metric_scores
        max_raw_progress = np.max(raw_progress)
        if max_raw_progress > PROGRESS_DISTANCE_THRESHOLD:
            normalized_progress = raw_progress / max_raw_progress
        else:
            normalized_progress = np.ones(len(raw_progress), dtype=np.float64)
            normalized_progress[multiplicate_metric_scores == 0.0] = 0.0
        self._weighted_metrics[WeightedMetricIndex.PROGRESS] = normalized_progress

        # accumulate weighted metrics
        weighted_metric_scores = (
            self._weighted_metrics * WEIGHTED_METRICS_WEIGHTS[..., None]
        ).sum(axis=0)
        weighted_metric_scores /= WEIGHTED_METRICS_WEIGHTS.sum()

        # calculate final scores
        final_scores = multiplicate_metric_scores * weighted_metric_scores

        return final_scores

    def _reset(
        self,
        states: npt.NDArray[np.float64],
        initial_ego_state: EgoState,
        observation: PDMObservation,
        centerline: PDMPath,
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
        drivable_area_map: PDMOccupancyMap,
        map_api: AbstractMap,
    ) -> None:
        """
        Resets metric values and lazy loads input classes.
        :param states: array representation of simulated proposals
        :param initial_ego_state: ego-vehicle state at current iteration
        :param observation: PDM's observation class
        :param centerline: path of the centerline
        :param route_lane_dict: dictionary containing on-route lanes
        :param drivable_area_map: Occupancy map of drivable are polygons
        :param map_api: map object
        """
        assert states.ndim == 3
        assert states.shape[1] == self._proposal_sampling.num_poses + 1
        assert states.shape[2] == StateIndex.size()

        self._initial_ego_state = initial_ego_state
        self._observation = observation
        self._centerline = centerline
        self._route_lane_dict = route_lane_dict
        self._drivable_area_map = drivable_area_map
        self._map_api = map_api

        self._num_proposals = states.shape[0]

        # save ego state values
        self._states = states

        # calculate coordinates of ego corners and center
        self._ego_coords = state_array_to_coords_array(
            states, initial_ego_state.car_footprint.vehicle_parameters
        )

        # initialize all ego polygons from corners
        self._ego_polygons = coords_array_to_polygon_array(self._ego_coords)

        # zero initialize all remaining arrays.
        self._ego_areas = np.zeros(
            (
                self._num_proposals,
                self._proposal_sampling.num_poses + 1,
                len(EgoAreaIndex),
            ),
            dtype=np.bool_,
        )
        self._multi_metrics = np.zeros(
            (len(MultiMetricIndex), self._num_proposals), dtype=np.float64
        )
        self._weighted_metrics = np.zeros(
            (len(WeightedMetricIndex), self._num_proposals), dtype=np.float64
        )
        self._progress_raw = np.zeros(self._num_proposals, dtype=np.float64)

        # initialize infraction arrays with infinity (meaning no infraction occurs)
        self._collision_time_idcs = np.zeros(self._num_proposals, dtype=np.float64)
        self._ttc_time_idcs = np.zeros(self._num_proposals, dtype=np.float64)
        self._collision_time_idcs.fill(np.inf)
        self._ttc_time_idcs.fill(np.inf)

    def _calculate_ego_area(self) -> None:
        """
        Determines the area of proposals over time.
        Areas are (1) in multiple lanes, (2) non-drivable area, or (3) oncoming traffic
        """

        n_proposals, n_horizon, n_points, _ = self._ego_coords.shape
        coordinates = self._ego_coords.reshape(n_proposals * n_horizon * n_points, 2)

        in_polygons = self._drivable_area_map.points_in_polygons(coordinates)
        in_polygons = in_polygons.reshape(
            len(self._drivable_area_map), n_proposals, n_horizon, n_points
        ).transpose(
            1, 2, 0, 3
        )  # shape: n_proposals, n_horizon, n_polygons, n_points

        drivable_area_on_route_idcs: List[int] = [
            idx
            for idx, token in enumerate(self._drivable_area_map.tokens)
            if token in self._route_lane_dict.keys()
        ]  # index mask for on-route lanes

        corners_in_polygon = in_polygons[..., :-1]  # ignore center coordinate
        center_in_polygon = in_polygons[..., -1]  # only center

        # in_multiple_lanes: if
        # - more than one drivable polygon contains at least one corner
        # - no polygon contains all corners
        batch_multiple_lanes_mask = np.zeros((n_proposals, n_horizon), dtype=np.bool_)
        batch_multiple_lanes_mask = (corners_in_polygon.sum(axis=-1) > 0).sum(
            axis=-1
        ) > 1

        batch_not_single_lanes_mask = np.zeros((n_proposals, n_horizon), dtype=np.bool_)
        batch_not_single_lanes_mask = np.all(
            corners_in_polygon.sum(axis=-1) != 4, axis=-1
        )

        multiple_lanes_mask = np.logical_and(
            batch_multiple_lanes_mask, batch_not_single_lanes_mask
        )
        self._ego_areas[multiple_lanes_mask, EgoAreaIndex.MULTIPLE_LANES] = True

        # in_nondrivable_area: if at least one corner is not within any drivable polygon
        batch_nondrivable_area_mask = np.zeros((n_proposals, n_horizon), dtype=np.bool_)
        batch_nondrivable_area_mask = (corners_in_polygon.sum(axis=-2) > 0).sum(
            axis=-1
        ) < 4
        self._ego_areas[
            batch_nondrivable_area_mask, EgoAreaIndex.NON_DRIVABLE_AREA
        ] = True

        # in_oncoming_traffic: if center not in any drivable polygon that is on-route
        batch_oncoming_traffic_mask = np.zeros((n_proposals, n_horizon), dtype=np.bool_)
        batch_oncoming_traffic_mask = (
            center_in_polygon[..., drivable_area_on_route_idcs].sum(axis=-1) == 0
        )
        self._ego_areas[
            batch_oncoming_traffic_mask, EgoAreaIndex.ONCOMING_TRAFFIC
        ] = True

    def _calculate_no_at_fault_collision(self) -> None:
        """
        Re-implementation of nuPlan's at-fault collision metric.
        """
        
        
        no_at_fault_collision_dict = dict()
        
        no_collision_scores = np.ones(self._num_proposals, dtype=np.float64)

        proposal_collided_track_ids = {
            proposal_idx: copy.deepcopy(self._observation.collided_track_ids)
            for proposal_idx in range(self._num_proposals)
        }
        
        
        no_at_fault_collision_dict['collided_track_ids'] = proposal_collided_track_ids
        no_at_fault_collision_dict['new_collisions'] = dict()

        for time_idx in range(self._proposal_sampling.num_poses + 1):
            
            ego_polygons = self._ego_polygons[:, time_idx]
            intersecting = self._observation[time_idx].query(
                ego_polygons, predicate="intersects"
            )

            if len(intersecting) == 0:
                continue
            
            # assert len(proposal_idx)==1, 'more than one trajectories are proposed'
            
            for proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
                token = self._observation[time_idx].tokens[geometry_idx] 
                if (self._observation.red_light_token in token) or (
                    token in proposal_collided_track_ids[proposal_idx]
                ):
                    continue
                
                
                ego_in_multiple_lanes_or_nondrivable_area = (
                    self._ego_areas[proposal_idx, time_idx, EgoAreaIndex.MULTIPLE_LANES]
                    or self._ego_areas[
                        proposal_idx, time_idx, EgoAreaIndex.NON_DRIVABLE_AREA
                    ]
                )

                tracked_object = self._observation.unique_objects[token] 
                no_at_fault_collision_dict['new_collisions'][token] = dict()
                no_at_fault_collision_dict['new_collisions'][token]['object'] = tracked_object
                no_at_fault_collision_dict['new_collisions'][token]['agent_type'] = agent_type_name[tracked_object.tracked_object_type]
                no_at_fault_collision_dict['new_collisions'][token]['initial_time'] = self._time_from_t_idx(time_idx)
                centroid = self._observation[time_idx][token].centroid
                track_heading = tracked_object.box.center.heading
                track_state = StateSE2(centroid.x, centroid.y, track_heading)
                track_state = absolute_to_relative_poses([self._initial_ego_state.car_footprint.center, track_state])[1]
                no_at_fault_collision_dict['new_collisions'][token]['state'] = track_state

                # classify collision
                collision_type: CollisionType = get_collision_type(
                    self._states[proposal_idx, time_idx],
                    self._ego_polygons[proposal_idx, time_idx],
                    tracked_object,
                    self._observation[time_idx][token],
                )
                no_at_fault_collision_dict['new_collisions'][token]['collision_type'] = collision_type
                
                collisions_at_stopped_track_or_active_front: bool = collision_type in [
                    CollisionType.ACTIVE_FRONT_COLLISION,
                    CollisionType.STOPPED_TRACK_COLLISION,
                ]
                collision_at_lateral: bool = (
                    collision_type == CollisionType.ACTIVE_LATERAL_COLLISION
                )

                # 1. at fault collision
                if collisions_at_stopped_track_or_active_front or (
                    ego_in_multiple_lanes_or_nondrivable_area and collision_at_lateral
                ):
                    no_at_fault_collision_dict['new_collisions'][token]['at_fault'] = True
                    no_at_fault_collision_score = (
                        0.0
                        if tracked_object.tracked_object_type in AGENT_TYPES
                        else 0.5
                    )
                    no_collision_scores[proposal_idx] = np.minimum(
                        no_collision_scores[proposal_idx], no_at_fault_collision_score
                    )
                    self._collision_time_idcs[proposal_idx] = min(
                        time_idx, self._collision_time_idcs[proposal_idx]
                    )

                else:  # 2. no at fault collision
                    no_at_fault_collision_dict['new_collisions'][token]['at_fault'] = False
                    proposal_collided_track_ids[proposal_idx].append(token)

        if len(no_at_fault_collision_dict['new_collisions'])==0:
            no_at_fault_collision_dict['error_bool'] = False
        else:
            no_at_fault_collision_dict['error_bool'] = True
        self._multi_metrics[MultiMetricIndex.NO_COLLISION] = no_collision_scores
        
        return no_at_fault_collision_dict

    def _calculate_ttc(self):
        """
        Re-implementation of nuPlan's time-to-collision metric.
        """
        
        
        
        ttc_dict = dict()

        ttc_scores = np.ones(self._num_proposals, dtype=np.float64)
        temp_collided_track_ids = {
            proposal_idx: copy.deepcopy(self._observation.collided_track_ids)
            for proposal_idx in range(self._num_proposals)
        }

        # calculate TTC for 1s in the future with less temporal resolution.
        future_time_idcs = np.arange(0, 10, 3)
        n_future_steps = len(future_time_idcs)

        # create polygons for each ego position and 1s future projection
        coords_exterior = self._ego_coords.copy()
        coords_exterior[:, :, BBCoordsIndex.CENTER, :] = coords_exterior[
            :, :, BBCoordsIndex.FRONT_LEFT, :
        ]
        coords_exterior_time_steps = np.repeat(
            coords_exterior[:, :, None], n_future_steps, axis=2
        )

        speeds = np.hypot(
            self._states[..., StateIndex.VELOCITY_X],
            self._states[..., StateIndex.VELOCITY_Y],
        )

        dxy_per_s = np.stack(
            [
                np.cos(self._states[..., StateIndex.HEADING]) * speeds,
                np.sin(self._states[..., StateIndex.HEADING]) * speeds,
            ],
            axis=-1,
        )

        for idx, future_time_idx in enumerate(future_time_idcs):
            delta_t = float(future_time_idx) * self._proposal_sampling.interval_length
            coords_exterior_time_steps[:, :, idx] = (
                coords_exterior_time_steps[:, :, idx] + dxy_per_s[:, :, None] * delta_t
            )

        polygons = creation.polygons(coords_exterior_time_steps)
        
        ttc_dict['time_stamp'] = dict()

        # check collision for each proposal and projection
        for time_idx in range(self._proposal_sampling.num_poses + 1):
            time_stamp = self._time_from_t_idx(time_idx)
            for step_idx, future_time_idx in enumerate(future_time_idcs):
                future_dict = dict()
                
                current_time_idx = time_idx + future_time_idx
                polygons_at_time_step = polygons[:, time_idx, step_idx]
                intersecting = self._observation[current_time_idx].query(
                    polygons_at_time_step, predicate="intersects"
                )

                if len(intersecting) == 0:
                    continue
                
                ttc_dict['time_stamp'][time_stamp] = []
                for proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
                    token = self._observation[current_time_idx].tokens[geometry_idx]
                    if (
                        (self._observation.red_light_token in token)
                        or (token in temp_collided_track_ids[proposal_idx])
                        or (speeds[proposal_idx, time_idx] < STOPPED_SPEED_THRESHOLD)
                    ):
                        continue
                    
                    ego_in_multiple_lanes_or_nondrivable_area = (
                        self._ego_areas[
                            proposal_idx, time_idx, EgoAreaIndex.MULTIPLE_LANES
                        ]
                        or self._ego_areas[
                            proposal_idx, time_idx, EgoAreaIndex.NON_DRIVABLE_AREA
                        ]
                    )
                    ego_rear_axle: StateSE2 = StateSE2(
                        *self._states[proposal_idx, time_idx, StateIndex.STATE_SE2]
                    )

                    centroid = self._observation[current_time_idx][token].centroid
                    track_heading = self._observation.unique_objects[
                        token
                    ].box.center.heading
                    track_state = StateSE2(centroid.x, centroid.y, track_heading)
                    
                    future_dict['token'] = ''
                    # future_dict['token'] = token
                    future_dict['object'] = self._observation.unique_objects[token]
                    future_dict['agent_type'] = agent_type_name[self._observation.unique_objects[token].tracked_object_type]
                    track_state = absolute_to_relative_poses([self._initial_ego_state.car_footprint.center, track_state])[1]
                    future_dict['state'] = track_state
                    future_dict['collition_time'] = float(future_time_idx) * self._proposal_sampling.interval_length
                    
                    if is_agent_ahead(ego_rear_axle, track_state) or (
                        (
                            ego_in_multiple_lanes_or_nondrivable_area
                            or self._map_api.is_in_layer(
                                ego_rear_axle, layer=SemanticMapLayer.INTERSECTION
                            )
                        )
                        and not is_agent_behind(ego_rear_axle, track_state)
                    ):
                        future_dict['at_fault'] = True
                        ttc_scores[proposal_idx] = np.minimum(
                            ttc_scores[proposal_idx], 0.0
                        )
                        self._ttc_time_idcs[proposal_idx] = min(
                            time_idx, self._ttc_time_idcs[proposal_idx]
                        )
                    else:
                        future_dict['at_fault'] = False
                        temp_collided_track_ids[proposal_idx].append(token)
                    
                    ttc_dict['time_stamp'][time_stamp].append(future_dict)
        
        if len(ttc_dict['time_stamp'])==0:
            ttc_dict['error_bool'] = False
        else:
            ttc_dict['error_bool'] = True
        self._weighted_metrics[WeightedMetricIndex.TTC] = ttc_scores
        return ttc_dict

    def _calculate_progress(self) -> None:
        """
        Re-implementation of nuPlan's progress metric (non-normalized).
        Calculates progress along the centerline.
        """
        progress_dict = dict()
        
        # calculate raw progress in meter
        progress_in_meter = np.zeros(self._num_proposals, dtype=np.float64)
        assert self._num_proposals==1, 'more than one trajectories are given'
        for proposal_idx in range(self._num_proposals):
            start_point = Point(
                *self._ego_coords[proposal_idx, 0, BBCoordsIndex.CENTER]
            )
            end_point = Point(*self._ego_coords[proposal_idx, -1, BBCoordsIndex.CENTER])
            center_points = self._ego_coords[proposal_idx, :, BBCoordsIndex.CENTER]
            real_progress = np.sum(np.sqrt(np.sum((np.diff(center_points, axis=0))**2, axis=1)))
            
            centerline = self._centerline._states_se2_array
            center_states = [StateSE2.deserialize(pose) for pose in centerline]
            center_states = absolute_to_relative_poses([self._initial_ego_state.car_footprint.rear_axle] + center_states)[1:]
            relative_ego_center = absolute_to_relative_poses([self._initial_ego_state.car_footprint.rear_axle, self._initial_ego_state.car_footprint.center])[1:]
            relative_ego_rear = absolute_to_relative_poses([self._initial_ego_state.car_footprint.rear_axle, self._initial_ego_state.car_footprint.rear_axle])[1:]
            center_states, _, _ = clip_centerline_by_ego_pose(center_states, relative_ego_center, relative_ego_rear)
            if center_states is not None:
                # center_states = np.array([[stat.x, stat.y, stat.heading] for stat in center_states])   
                center_states = np.round(equidistant_interpolation(center_states, 20),2)
            
            progress = self._centerline.project([start_point, end_point])
            progress_in_meter[proposal_idx] = progress[1] - progress[0]
            
            progress_dict['progress'] = progress[1] - progress[0]
            progress_dict['real_progress'] = real_progress
            progress_dict['progress_rate'] = (progress[1] - progress[0]) / real_progress 
            progress_dict['center_points_interpolated'] = center_states
            progress_dict['error_bool'] = (progress_dict['progress_rate']<0.9)

        self._progress_raw = progress_in_meter
        
        return progress_dict

    def _calculate_is_comfortable(self) -> None:
        """
        Re-implementation of nuPlan's comfortability metric.
        """
        comfortable_dict = dict()
        
        time_point_s: npt.NDArray[np.float64] = (
            np.arange(0, self._proposal_sampling.num_poses + 1).astype(np.float64)
            * self._proposal_sampling.interval_length
        )
        is_comfortable = ego_is_comfortable(self._states, time_point_s) # [n_proposal, len(comfort_metric)]
        
        assert len(is_comfortable)==1, 'more than one trajectories are given'
        comfortable_dict['metrics'] = [
            'lon_acceleration',
            'lat_acceleration',
            'jerk_metric',
            'lon_jerk_metric',
            'yaw_accel',
            'yaw_rate',
        ]
        for proposal_idx in range(len(is_comfortable)):
            comfortable_dict['uncomfortable_metrics'] = np.array(comfortable_dict['metrics'])[~is_comfortable[proposal_idx]]
        
        self._weighted_metrics[WeightedMetricIndex.COMFORTABLE] = np.all(
            is_comfortable, axis=-1
        )
        comfortable_dict['error_bool'] = (len(comfortable_dict['uncomfortable_metrics'])!=0)
        
        return comfortable_dict

    def _calculate_drivable_area_compliance(self) -> None:
        """
        Re-implementation of nuPlan's drivable area compliance metric
        """
        drivable_area_compliance_dict = dict()
        
        drivable_area_compliance_scores = np.ones(self._num_proposals, dtype=np.float64)
        off_road_mask = self._ego_areas[:, :, EgoAreaIndex.NON_DRIVABLE_AREA].any(
            axis=-1
        ) 
        drivable_area_compliance_scores[off_road_mask] = 0.0
        
        drivable_area_compliance_dict['off_road_record'] = []
        off_road_record = self._ego_areas[:, :, EgoAreaIndex.NON_DRIVABLE_AREA]
        assert len(self._ego_areas)==1, 'more than one trajectories are given'
        for proposal_idx in range(len(self._ego_areas)):
            proposal_off_road_record = off_road_record[proposal_idx]
            if not proposal_off_road_record.all()==True:
                continue
            time_idx = np.where(np.diff(proposal_off_road_record))[0]+1
            records_split = np.split(proposal_off_road_record, time_idx)
            if len(time_idx)!=0:
                time_idx = time_idx.tolist().insert(0, 0)
            else:
                time_idx = [0]
            for rec, init_t in zip(records_split, time_idx):
                if rec.sum()!=0:
                    duration = rec.sum() * self._proposal_sampling.interval_length
                    time_initial = self._time_from_t_idx(init_t)
                    drivable_area_compliance_dict['off_road_record'].append([time_initial, duration])
        drivable_area_compliance_dict['error_bool'] = False if len(drivable_area_compliance_dict['off_road_record'])==0 else True
            
        self._multi_metrics[
            MultiMetricIndex.DRIVABLE_AREA
        ] = drivable_area_compliance_scores
        
        return drivable_area_compliance_dict

    def _calculate_driving_direction_compliance(self) -> None:
        """
        Re-implementation of nuPlan's driving direction compliance metric
        """
        driving_direction_compliance_dict = dict()
        
        center_coordinates = self._ego_coords[:, :, BBCoordsIndex.CENTER]
        cum_progress = np.zeros(
            (self._num_proposals, self._proposal_sampling.num_poses + 1),
            dtype=np.float64,
        )
        cum_progress[:, 1:] = (
            (center_coordinates[:, 1:] - center_coordinates[:, :-1]) ** 2.0
        ).sum(axis=-1) ** 0.5

        # mask out progress along the driving direction
        oncoming_traffic_masks = self._ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC]
        cum_progress[~oncoming_traffic_masks] = 0.0

        driving_direction_compliance_scores = np.ones(
            self._num_proposals, dtype=np.float64
        )

        for proposal_idx in range(self._num_proposals):
            oncoming_traffic_progress, oncoming_traffic_mask = (
                cum_progress[proposal_idx],
                oncoming_traffic_masks[proposal_idx],
            )

            # split progress whenever ego changes traffic direction
            oncoming_progress_splits = np.split(
                oncoming_traffic_progress,
                np.where(np.diff(oncoming_traffic_mask))[0] + 1,
            )

            # sum up progress of splitted intervals
            # Note: splits along the driving direction will have a sum of zero.
            max_oncoming_traffic_progress = max(
                oncoming_progress.sum()
                for oncoming_progress in oncoming_progress_splits
            )
            
            change_time_idx = np.where(np.diff(oncoming_traffic_mask))[0] + 1
            wrong_distances = [oncoming_progress.sum() for oncoming_progress in oncoming_progress_splits]
            driving_direction_compliance_dict['not_far'] = []
            driving_direction_compliance_dict['far'] = []
            for t_idx, wrong_dis in zip(change_time_idx, wrong_distances):
                if wrong_dis < DRIVING_DIRECTION_COMPLIANCE_THRESHOLD:
                    continue
                elif wrong_dis < DRIVING_DIRECTION_VIOLATION_THRESHOLD:
                    time_initial = self._time_from_t_idx(t_idx)
                    driving_direction_compliance_dict['not_far'].append([time_initial, wrong_dis])
                else:
                    time_initial = self._time_from_t_idx(t_idx)
                    driving_direction_compliance_dict['far'].append([time_initial, wrong_dis])
                    
            if len(driving_direction_compliance_dict['not_far'])==0 and len(driving_direction_compliance_dict['far'])==0:
                driving_direction_compliance_dict['error_bool'] = False
            else:
                driving_direction_compliance_dict['error_bool'] = True
            

            if max_oncoming_traffic_progress < DRIVING_DIRECTION_COMPLIANCE_THRESHOLD:
                driving_direction_compliance_scores[proposal_idx] = 1.0
            elif max_oncoming_traffic_progress < DRIVING_DIRECTION_VIOLATION_THRESHOLD:
                driving_direction_compliance_scores[proposal_idx] = 0.5
            else:
                driving_direction_compliance_scores[proposal_idx] = 0.0

        self._multi_metrics[
            MultiMetricIndex.DRIVING_DIRECTION
        ] = driving_direction_compliance_scores
        
        return driving_direction_compliance_dict
        
    def _time_from_t_idx(self, time_id):
        return np.round(time_id * self._proposal_sampling.interval_length, 2)
    
    def _no_at_fault_collision_to_prompt(self):
        no_at_fault_collision_prompt = ''
        new_collision_info = self.error_dict['no_at_fault_collision']['new_collisions']
        
        for collision_token in new_collision_info.keys():
            token_info = new_collision_info[collision_token]
            # collision info prompt
            collision_token = ''
            no_at_fault_collision_prompt += 'At %s s, the ego will collide with %s %s at the coordinate [%s, %s]. ' \
                    %(str(token_info['initial_time']), token_info['agent_type'], collision_token, np.round(token_info['state'].x,2), np.round(token_info['state'].y,2))
            
            # collision type prompt
            collision_type = token_info['collision_type']
            if collision_type==0:
                # STOPPED_EGO_COLLISION
                no_at_fault_collision_prompt += 'It is a collision that happens when ego is stopped. '
            elif collision_type==1:
                # STOPPED_TRACK_COLLISION
                no_at_fault_collision_prompt += 'It is a collision that happens when the %s is stopped. '%(token_info['agent_type'])
            elif collision_type==2:
                # ACTIVE_FRONT_COLLISION
                no_at_fault_collision_prompt += 'It is a collision that happens when front bumper of ego hits the %s. '%(token_info['agent_type'])
            elif collision_type==3:
                # ACTIVE_REAR_COLLISION
                no_at_fault_collision_prompt += 'It is a collision that happens when the %s hits ego in the rear. '%(token_info['agent_type'])
            elif collision_type==4:
                # ACTIVE_LATERAL_COLLISION
                no_at_fault_collision_prompt += 'It is a collision that happens when the %s and ego hit on the sides. '%(token_info['agent_type'])
            
            # at fault promt
            if token_info['at_fault']:
                no_at_fault_collision_prompt += 'This collision is at your fault. '
            else:
                no_at_fault_collision_prompt += 'This collision is not at your fault. '
        
        return no_at_fault_collision_prompt
    
    def _driving_direction_compliance_to_prompt(self):
        driving_direction_compliance_prompt = ''
        direction_compliance_info = self.error_dict['driving_direction_compliance']
        
        # not far prompt
        for dir_info in direction_compliance_info['not_far']:
            driving_direction_compliance_prompt += 'The ego will deflect out of the driving direction at %s s and deflect %s m. '\
                %(str(dir_info[0]), str(dir_info[1]))
        
        # far prompt
        for dir_info_far in direction_compliance_info['far']:
            driving_direction_compliance_prompt += 'The ego will be seriously deflected out of the driving direction at %s s, deflected by %s m.'\
                %(str(dir_info_far[0]), str(dir_info_far[1]))
        
        return driving_direction_compliance_prompt
    
    def _drivable_area_compliance_to_prompt(self):
        drivable_area_compliance_prompt = ''
        area_compliance_info = self.error_dict['drivable_area_compliance']['off_road_record']
        
        for off_road_record in area_compliance_info:
            drivable_area_compliance_prompt += 'Starting from %s s, the ego car continues to drive in the non drivable area for %s s. '\
                %(str(np.round(off_road_record[0],2)), str(np.round(off_road_record[1],2)))
                
        return drivable_area_compliance_prompt
    
    def _progress_to_prompt(self):
        progress_info = self.error_dict['progress']
        if progress_info['center_points_interpolated'] is not None:
            progress_prompt = 'The Ego car actually moved forward %sm, but only moved %sm along the centerline, the progress rate is only %s. Please driving follow the centerline which is:\n%s. ' \
                %(str(np.round(progress_info['real_progress'], 2)), str(np.round(progress_info['progress'],2)), str(np.round(progress_info['progress_rate'],2)), str(progress_info['center_points_interpolated']))
        else:
            progress_prompt = 'The Ego car actually moved forward %sm, but only moved %sm along the centerline, the progress rate is only %s. Please driving follow the centerline. ' \
                %(str(np.round(progress_info['real_progress'], 2)), str(np.round(progress_info['progress'],2)), str(np.round(progress_info['progress_rate'],2)))
            
        return progress_prompt
    
    def _ttc_to_prompt(self):
        ttc_prompt = ''
        ttc_info = self.error_dict['ttc']['time_stamp']
        
        for time_stamp in ttc_info.keys():
            future_dict_at_timestamp = ttc_info[time_stamp]
            for future_dict in future_dict_at_timestamp:
                ttc_prompt += 'If the ego car continues to drive in the direction at %s s, it may cause a collision with %s %s at coordinate [%s, %s] at %s s later, '\
                    %(str(time_stamp), future_dict['agent_type'], future_dict['token'], np.round(future_dict['state'].x,2), np.round(future_dict['state'].y,2), future_dict['collition_time'])
                if future_dict['at_fault']:
                    ttc_prompt += 'which is at your fault. '
                else:
                    ttc_prompt += 'which is not at your fault. '
                    
        return ttc_prompt
    
    def _comfortable_to_prompt(self):
        comfortable_prompt = ''
        comfortable_info = self.error_dict['comfortable']['uncomfortable_metrics']
        
        comfortable_prompt += 'The '
        for metric in comfortable_info:
            if metric == 'lon_acceleration':
                comfortable_prompt += 'longitudinal acceleration, '
            elif metric == 'lat_acceleration':
                comfortable_prompt += 'lateral acceleration, '
            # elif metric == 'jerk_metric':
            #     comfortable_prompt += ''
            # elif metric == 'lon_jerk_metric':
            #     comfortable_prompt += ''
            elif metric == 'yaw_accel':
                comfortable_prompt += 'yaw acceleration, '
            elif metric == 'yaw_rate':
                comfortable_prompt += 'yaw rate, '
        comfortable_prompt += 'can lead to lack of comfort. Please adjust appropriately. '
        
        return comfortable_prompt
                
        
        
        
        