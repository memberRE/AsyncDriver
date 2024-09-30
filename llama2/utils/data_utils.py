import numpy as np

from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.nuplan_map.lane import NuPlanLane
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
from nuplan.planning.metrics.utils.state_extractors import extract_ego_center, extract_ego_time_point
from nuplan.planning.metrics.utils.route_extractor import (
    CornersGraphEdgeMapObject,
    extract_corners_route,
    get_common_or_connected_route_objs_of_corners,
    get_outgoing_edges_obj_dict,
    get_route,
    get_timestamps_in_common_or_connected_route_objs,
)
from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import find_lane_changes

def state_se2_to_array(state_se2: StateSE2):
    return np.array([state_se2.x, state_se2.y, state_se2.heading], dtype=np.float64)
state_se2_to_array_vectorize = np.vectorize(state_se2_to_array, signature="()->(3)")


def find_current_lane(map_api, ego_pose):
    # find all lane candidates
    layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    lane_candidates = []
    finding_radius = 0.01
    while lane_candidates==[]:
        roadblock_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=finding_radius, layers=layers     # 0.01在lane和lane connector之间的的时候会找不到，调整成0.1就可以了
        )
        lane_candidates = roadblock_dict[SemanticMapLayer.LANE] + roadblock_dict[SemanticMapLayer.LANE_CONNECTOR]
        if len(lane_candidates) == 0:
            if finding_radius >= 1:
                finding_radius += 0.1
            else:
                finding_radius *= 10
            if finding_radius > 3:
                # logger.info("!!!!!!!!!!!! No lane found in ego location. Even No lane connector found in the map.")
                return None
            continue
    
    # find current lane
    if len(lane_candidates) == 1:
        return lane_candidates[0]
    lane_discrete_path = [lane.baseline_path.get_nearest_pose_from_position(ego_pose.point) for lane in lane_candidates]
    # find the lane with the smallest heading error
    lane_discrete_path_np = state_se2_to_array_vectorize(
        np.array(lane_discrete_path, dtype=np.object_)
    )
    lane_heading = lane_discrete_path_np[:, 2]
    heading_error = np.abs(np.unwrap(lane_heading - ego_pose.heading))
    idx = np.argmin(heading_error)
    return lane_candidates[idx]

def encode_traffic_light(lane, traffic_light, ego_pose):
    ego_lane_flag = True
    distance = []
    if isinstance(lane, NuPlanLane):
        traffic_light_lanes = lane.outgoing_edges
        ego_lane_flag = False
    else:  # lane connector
        traffic_light_lanes = [lane]
    traffic_light_for_lanes = []
    for traffic_light_lane in traffic_light_lanes:
        if traffic_light_lane.has_traffic_lights():
            relevant_status = [
                t
                for t in traffic_light
                if t.lane_connector_id == int(traffic_light_lane.id)
            ]
            if len(relevant_status) > 0:
                traffic_light_for_lanes.append(relevant_status[0].status.serialize())
            else:
                traffic_light_for_lanes.append('UNKNOWN')
        else:
            traffic_light_for_lanes.append('GREEN')
        distance.append(ego_pose.distance_to(
            traffic_light_lane.baseline_path.get_nearest_pose_from_position(ego_pose.point).point))
    
    if len(traffic_light_for_lanes)==0:
        traffic_light_for_lanes.append('UNKNOWN')
        
    traffic_light_type = TrafficLightStatusType[traffic_light_for_lanes[0]].value
    traffic_light_type_one_hot = np.array([0,0,0,0])
    traffic_light_type_one_hot[traffic_light_type] = 1
    return traffic_light_for_lanes, ego_lane_flag, distance


def find_lane_change(traj, map_api):
    """
    Returns the lane chane metric
    :param history: History from a simulation engine
    :param scenario: Scenario running this metric
    :return the estimated lane change duration in micro seconds and status.
    """
    # Extract xy coordinates of center of ego from history.
    # ego_states = traj.extract_ego_state
    ego_states = traj
    ego_poses = extract_ego_center(ego_states)

    # Get the list of lane or lane_connectors associated to ego at each time instance, and store to use in other metrics
    # ego_driven_route = get_route(map_api, ego_poses)

    # Extract ego timepoints
    ego_timestamps = extract_ego_time_point(ego_states)

    # Extract corners of ego's footprint
    ego_footprint_list = [ego_state.car_footprint for ego_state in ego_states]

    # Extract corner lanes/lane connectors
    corners_route = extract_corners_route(map_api, ego_footprint_list)
    # Store to load in high level metrics
    corners_route = corners_route

    common_or_connected_route_objs = get_common_or_connected_route_objs_of_corners(corners_route)

    # Extract ego timepoints where its corners are in common_or_connected_route objs
    timestamps_in_common_or_connected_route_objs = get_timestamps_in_common_or_connected_route_objs(
        common_or_connected_route_objs, ego_timestamps
    )

    # Store to load in high level metrics
    timestamps_in_common_or_connected_route_objs = timestamps_in_common_or_connected_route_objs

    # Extract lane changes in the history
    lane_changes = find_lane_changes(ego_timestamps, common_or_connected_route_objs)
    
    return lane_changes