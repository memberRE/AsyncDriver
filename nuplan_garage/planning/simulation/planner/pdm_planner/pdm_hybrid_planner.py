import gc
import logging
import warnings
from typing import List, Optional, Type, cast

import numpy as np
import torch
from nuplan_garage.planning.simulation.planner.pdm_planner.scoring.llm_scorer import PDMScorer4LLM

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimeDuration, TimePoint, StateSE2, StateVector2D
from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    Observation,
)
from nuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    BBCoordsIndex,
    SE2Index,
    StateIndex,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
    transform_predictions_to_states,
)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.lightning_module_wrapper import (
    LightningModuleWrapper,
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.utils.serialization.scene import Trajectory

from nuplan_garage.planning.simulation.planner.pdm_planner.abstract_pdm_closed_planner import (
    AbstractPDMClosedPlanner,
)
from nuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation_utils import (
    get_drivable_area_map,
)
from nuplan_garage.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import (
    BatchIDMPolicy,
)
from nuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_feature_utils import (
    create_pdm_feature,
)
from nuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses

from shapely.geometry import Point
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.common.actor_state.oriented_box import OrientedBox

warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)

def _convert_trajectory_to_proposal(trajectory: InterpolatedTrajectory, sampling: TrajectorySampling):
    # convert to numpy
    start_time = trajectory.start_time
    relative_time_points = [
        TimePoint(start_time.time_us + int(sampling.interval_length * i * 1e6))
        for i in range(sampling.num_poses + 1)]
    new_trajectory = trajectory.get_state_at_times(relative_time_points)
    pro_array = np.zeros((sampling.num_poses + 1, StateIndex.size()))
    for i in range(pro_array.shape[0]):
        state:EgoState = new_trajectory[i]
        pro_array[i][StateIndex.STATE_SE2] = state.rear_axle.serialize()
        pro_array[i][StateIndex.VELOCITY_2D] = state.dynamic_car_state.rear_axle_velocity_2d.array
        pro_array[i][StateIndex.ACCELERATION_2D] = state.dynamic_car_state.rear_axle_acceleration_2d.array
        pro_array[i][StateIndex.STEERING_ANGLE] = state.tire_steering_angle
        pro_array[i][StateIndex.ANGULAR_VELOCITY] = state.dynamic_car_state.angular_velocity
        pro_array[i][StateIndex.ANGULAR_ACCELERATION] = state.dynamic_car_state.angular_acceleration
        pro_array[i][StateIndex.STEERING_RATE] = state.dynamic_car_state.tire_steering_rate

    pro_array = pro_array[np.newaxis, ...]
    return pro_array



class PDMHybridPlanner(AbstractPDMClosedPlanner):
    """PDM-Closed planner class."""

    # Inherited property, see superclass.
    requires_scenario: bool = False

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        proposal_sampling: TrajectorySampling,
        idm_policies: BatchIDMPolicy,
        lateral_offsets: Optional[List[float]],
        map_radius: float,
        model: TorchModuleWrapper,
        correction_horizon: float,
        use_better_anchor: bool,
        checkpoint_path: str,
        leading_agent_update_rate: int = 2,
    ):
        """
        Constructor for PDM-Hybrid.
        :param trajectory_sampling: sampling parameters for final trajectory
        :param proposal_sampling: sampling parameters for proposals
        :param idm_policies: BatchIDMPolicy class
        :param lateral_offsets: centerline offsets for proposals (optional)
        :param map_radius: radius around ego to consider
        :param model: torch model
        :param correction_horizon: time to apply open-loop correction [s]
        :param checkpoint_path: path to checkpoint for model as string
        """

        super(PDMHybridPlanner, self).__init__(
            trajectory_sampling,
            proposal_sampling,
            idm_policies,
            lateral_offsets,
            map_radius,
            leading_agent_update_rate,
        )
        self.use_better_anchor = use_better_anchor

        self._device = "cpu"

        self._model = LightningModuleWrapper.load_from_checkpoint(
            checkpoint_path,
            model=model,
            map_location=self._device,
        ).model
        self._model.eval()
        torch.set_grad_enabled(False)

        self._correction_horizon: float = correction_horizon  # [s]
        self.lateral_offsets = lateral_offsets
        self.score_4_LLM = PDMScorer4LLM(proposal_sampling)

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        self._iteration = 0
        self._map_api = initialization.map_api
        self._load_route_dicts(initialization.route_roadblock_ids)
        gc.collect()

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(
        self, current_input: PlannerInput
    ) -> AbstractTrajectory:
        """Inherited, see superclass."""
        gc.disable()
        ego_state, _ = current_input.history.current_state

        # Apply route correction on first iteration (ego_state required)
        if self._iteration == 0:
            self._route_roadblock_correction(ego_state)

        # Update/Create drivable area polygon map
        self._drivable_area_map = get_drivable_area_map(
            self._map_api, ego_state, self._map_radius
        )

        # get ego future for check
        if self.use_better_anchor:
            # get instruction from gt
            trajectory_absolute_states = current_input.scenario.get_ego_future_trajectory(
                iteration=self._iteration, num_samples=20, time_horizon=10
            )
            ege_future_states = [state for state in trajectory_absolute_states]
            trajectory_abs_states = [state.rear_axle for state in ege_future_states]
            trajectory_relative_poses = convert_absolute_to_relative_poses(
                ego_state.rear_axle, [state.rear_axle for state in ege_future_states]
            )
            instruction, prompt = self.get_instruction(trajectory_relative_poses, ege_future_states, threshold=0.5)
            
            # get better anchor from instruction
            current_lane = self._get_starting_lane(ego_state)
            self._centerline = PDMPath(self._get_discrete_centerline(current_lane))
            # if self._iteration>100:
            traj, traj_se2, progress_ls = self.instruction_2_traj(instruction, self._centerline, ego_state, num_poses=20, time_horizon=10)
            anchor = PDMPath(traj_se2)
            ### for debug
            # ege_future_states = [ego_state]+[state for state in trajectory_absolute_states]
            # trajectory_abs_states = [ego_state.rear_axle]+[state.rear_axle for state in ege_future_states]
            # anchor = PDMPath(trajectory_abs_states)
            # trajactories = InterpolatedTrajectory(ege_future_states)
            # uncorrected_states = trajactories.get_sampled_trajectory()
            ###
            closed_loop_trajectory = self._get_closed_loop_trajectory(current_input)
            uncorrected_states = closed_loop_trajectory.get_sampled_trajectory()
            # trajectory of PDM-Offset
            pdm_feature = create_pdm_feature(
                self._model,
                current_input,
                anchor,
                None,
                self._device,
            )
        else:
            # Create centerline
            current_lane = self._get_starting_lane(ego_state)
            # end_lane = self._get_starting_lane(ege_future_states[-1])
            # same_route = (end_lane.id == current_lane.id)
            self._centerline = PDMPath(self._get_discrete_centerline(current_lane))

            # trajectory of PDM-Closed
            closed_loop_trajectory = self._get_closed_loop_trajectory(current_input)
            if not self.lateral_offsets:
                # calculate free trajectory
                return closed_loop_trajectory
            uncorrected_states = closed_loop_trajectory.get_sampled_trajectory()

            # trajectory of PDM-Offset
            pdm_feature = create_pdm_feature(
                self._model,
                current_input,
                self._centerline,
                closed_loop_trajectory,
                self._device,
            )
        predictions = self._model.forward({"pdm_features": pdm_feature})
        trajectory_data = (
            cast(Trajectory, predictions["trajectory"]).data.cpu().detach().numpy()[0]
        )
        corrected_states = transform_predictions_to_states(
            trajectory_data,
            current_input.history.ego_states,
            self._model.trajectory_sampling.time_horizon,
            self._model.trajectory_sampling.step_time,
        )

        # apply correction by fusing
        trajectory = self._apply_trajectory_correction(
            uncorrected_states, corrected_states
        )
        self._iteration += 1
        return trajectory
    
    def get_instruction(self, ego_future_poses, ege_future_states, threshold=0.5, return_prompt=False):
        dis_norm = np.linalg.norm(np.diff(np.concatenate([ego_future_poses[:1,:-1], ego_future_poses[:,:-1]], axis=0), n=1, axis=0), axis=1)
        dis_cum = np.cumsum(dis_norm, axis=0)
        
        cur_cmd = None
        cur_dis = 0
        instruction = ''
        cmd_ls = []
        dis_ls = []
        time_ls = []
        v_cmd_ls = []
        tmp_v_ls = []
        for heading, (idx, dis) in zip(ego_future_poses[:,2], enumerate(dis_cum)):
            if heading > threshold :
                cmd = 'turn right in '
            elif heading < -threshold:
                cmd = 'turn left in '
            elif dis-cur_dis>0.1:
                cmd = 'go straight in '
            else:
                cmd = 'stop. '
            if cmd!=cur_cmd:
                if return_prompt:
                    instruction += cmd
                    if cmd != 'stop. ':
                        instruction += (str(np.round(dis-cur_dis, 2)) + ' meters. ')
                # if idx>1:
                #     delta_dis = (dis_cum[idx]-dis_cum[idx-1]) - (dis_cum[idx-1]-dis_cum[idx-2])
                #     if delta_dis>0.1:
                #         v_cmd_ls.append('accelerate')
                #     elif delta_dis<-0.1:
                #         v_cmd_ls.append('deccelerate')
                #     else:
                #         v_cmd_ls.append('keep')
                # else:
                #     v_cmd_ls.append('keep')
                cmd_ls.append(cmd)
                dis_ls.append(np.round(dis-cur_dis, 2))
                time_ls.append(idx*0.5)
                ax_abs = ege_future_states[idx]._dynamic_car_state.rear_axle_acceleration_2d.x
                ay_abs = ege_future_states[idx]._dynamic_car_state.rear_axle_acceleration_2d.y
                vx_abs = ege_future_states[idx]._dynamic_car_state.rear_axle_velocity_2d.x
                vy_abs = ege_future_states[idx]._dynamic_car_state.rear_axle_velocity_2d.y
                if np.sqrt(ax_abs**2+ay_abs**2) > 0.5:
                    if (ax_abs*vx_abs+ay_abs*vy_abs) > 0:
                        v_cmd = 'accelerate'
                    else:
                        v_cmd = 'deccelerate'
                else:
                    v_cmd = 'keep'
                tmp_v_ls.append(v_cmd)
                v_cmd_ls.append(tmp_v_ls)
                cur_dis = dis
                cur_cmd = cmd
                tmp_v_ls = []
            else:
                ax_abs = ege_future_states[idx]._dynamic_car_state.rear_axle_acceleration_2d.x
                ay_abs = ege_future_states[idx]._dynamic_car_state.rear_axle_acceleration_2d.y
                vx_abs = ege_future_states[idx]._dynamic_car_state.rear_axle_velocity_2d.x
                vy_abs = ege_future_states[idx]._dynamic_car_state.rear_axle_velocity_2d.y
                if np.sqrt(ax_abs**2+ay_abs**2) > 0.5:
                    if (ax_abs*vx_abs+ay_abs*vy_abs) > 0:
                        v_cmd = 'accelerate'
                    else:
                        v_cmd = 'deccelerate'
                else:
                    v_cmd = 'keep'
                tmp_v_ls.append(v_cmd)
        
        dis_ls.append(np.round(dis-cur_dis, 2))
        time_ls.append(idx*0.5)
        time_ls = time_ls[1:]
        dis_ls = dis_ls[1:]
        # delta_dis = (dis_cum[idx]-dis_cum[idx-1]) - (dis_cum[idx-1]-dis_cum[idx-2])
        # if delta_dis>0.1:
        #     v_cmd_ls.append('accelerate')
        # elif delta_dis<-0.1:
        #     v_cmd_ls.append('deccelerate')
        # else:
        #     v_cmd_ls.append('keep')
        tmp_v_ls.append(v_cmd)
        v_cmd_ls.append(tmp_v_ls)
        v_cmd_ls = v_cmd_ls[1:]
        
        return [cmd_ls, time_ls, dis_ls, v_cmd_ls], instruction
    
    def instruction_2_traj(self, instruction, centerline, ego_state, num_poses=None, time_horizon=None):
        delta_t = time_horizon/num_poses
        init_t = ego_state._time_point
        init_v_2d = ego_state._dynamic_car_state.rear_axle_velocity_2d
        init_a_2d = ego_state._dynamic_car_state.rear_axle_acceleration_2d
        
        init_v = np.sqrt(init_v_2d.x**2+init_v_2d.y**2)
        init_a = np.sqrt(init_a_2d.x**2+init_a_2d.y**2)
        
        v = init_v
        if (init_v_2d.x*init_a_2d.x+init_v_2d.y*init_a_2d.y) > 0:
            a = init_a
        else:
            a = -init_a
        current_progress: float = centerline.project(
            Point(*ego_state.rear_axle.array)
        ) #62.86323685184562
        progress_ls = [current_progress]
        
        cmd_ls = instruction[0]
        time_ls = instruction[1]
        dis_ls = instruction[2]
        v_cmd_ls = instruction[3]
        for cmd, t, v_cmd in zip(cmd_ls, time_ls, v_cmd_ls):
            if 'stop' in cmd:
                for _ in range(int(t//delta_t)):
                    progress = progress_ls[-1]
                    progress_ls.append(progress)
                    if t>0.5:
                        v = 0
            else:
                for v_c in v_cmd:
                    if v_c == 'keep':
                        # for _ in range(int(t//delta_t)):
                        progress = progress_ls[-1]+v*0.5
                        progress_ls.append(progress)
                    if v_c == 'accelerate':
                        # for _ in range(int(t//delta_t)):
                        if a>0:
                            progress = progress_ls[-1]+v*0.5+0.5*a*0.5*0.5
                        else:
                            progress = progress_ls[-1]+v*0.5+0.5*0.5*0.5
                            a = 1
                        progress_ls.append(progress)
                        v += 0.5
                    if v_c == 'deccelerate':
                        # for _ in range(int(t//delta_t)):
                        if a<0:
                            progress = progress_ls[-1]+v*0.5+0.5*a*0.5*0.5
                        else:
                            progress = progress_ls[-1]+v*0.5-0.5*0.5*0.5
                            a = -1
                        progress_ls.append(progress)
                        if v-0.5>=0.25:
                            v -= 0.5
        progress_ls = progress_ls[1:]
        if len(progress_ls)>num_poses:
            progress_ls = progress_ls[:num_poses]
        anchor = centerline.interpolate(progress_ls, as_array=True)
        anchor_state = []
        anchor_se2 = []
        for t_id, p in enumerate(anchor):
            new_state = Waypoint(time_point=TimePoint(int(init_t.time_us+t_id*500000)), \
                oriented_box=OrientedBox(center=StateSE2(p[0], p[1], p[2]), \
                    length=ego_state.car_footprint.vehicle_parameters.rear_length, \
                        width=ego_state.car_footprint.vehicle_parameters.width, \
                            height=ego_state.car_footprint.vehicle_parameters.height))
            # new_state = ego_state.build_from_rear_axle(rear_axle_pose=StateSE2(p[0], p[1], p[2]),
            #                                             rear_axle_velocity_2d=StateVector2D(0,0),
            #                                             rear_axle_acceleration_2d=StateVector2D(0,0),
            #                                             tire_steering_angle=0,
            #                                             time_point=TimePoint(int(init_t.time_us+t_id*500000)),
            #                                             vehicle_parameters=ego_state.car_footprint.vehicle_parameters,)
            anchor_state.append(new_state)
            anchor_se2.append(StateSE2(p[0], p[1], p[2]))
                    
        return anchor_state, anchor_se2, progress_ls

    def compute_planner_trajectory_just_4_get_score(self, current_input: PlannerInput, llm_trajectory: InterpolatedTrajectory):
        """Inherited, see superclass."""
        gc.disable()
        ego_state, observation = current_input.history.current_state

        # Update/Create drivable area polygon map
        self._drivable_area_map = get_drivable_area_map(
            self._map_api, ego_state, self._map_radius
        )

        # Create centerline
        current_lane = self._get_starting_lane(ego_state)
        self._centerline = PDMPath(self._get_discrete_centerline(current_lane))

        # 1. Environment forecast and observation update
        self._observation.update(
            ego_state,
            observation,
            current_input.traffic_light_data,
            self._route_lane_dict,
        )

        # 2. Centerline extraction and proposal update
        self._update_proposal_manager(ego_state, self.use_better_anchor)

        # 3. Generate/Unroll proposals
        proposals_array = _convert_trajectory_to_proposal(llm_trajectory, self._proposal_sampling)
        # proposals_array = self._generator.generate_proposals(
        #     ego_state, self._observation, self._proposal_manager
        # )


        # 4. Simulate proposals
        simulated_proposals_array = self._simulator.simulate_proposals(
            proposals_array, ego_state
        )

        # 5. Score proposals
        proposal_scores = self.score_4_LLM.score_proposals(
            simulated_proposals_array,
            ego_state,
            self._observation,
            self._centerline,
            self._route_lane_dict,
            self._drivable_area_map,
            self._map_api,
        )
        
        prompt = self.score_4_LLM.to_error_prompt()
        # print(prompt)
        return prompt, proposal_scores


    def _apply_trajectory_correction(
        self,
        uncorrected_states: List[EgoState],
        corrected_states: List[EgoState],
    ) -> InterpolatedTrajectory:
        """
        Applies open-loop correction and fuses to a single trajectory.
        :param uncorrected_states: ego vehicles states of PDM-Closed trajectory
        :param corrected_states: ego-vehicles states of PDM-Offset trajectory
        :return: trajectory after applying correction.
        """

        # split trajectory
        uncorrected_duration: TimeDuration = TimeDuration.from_s(
            self._correction_horizon
        )
        cutting_time_point: TimePoint = (
            uncorrected_states[0].time_point + uncorrected_duration
        )

        uncorrected_split = [
            ego_state
            for ego_state in uncorrected_states
            if ego_state.time_point <= cutting_time_point
        ]

        corrected_split = [
            ego_state
            for ego_state in corrected_states
            if ego_state.time_point > cutting_time_point
        ]

        return InterpolatedTrajectory(uncorrected_split + corrected_split)