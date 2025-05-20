import sys
import argparse
import os
from pathlib import Path
import tempfile
from nuplan.planning.script.run_simulation import main as main_simulation
import hydra
import warnings
warnings.filterwarnings("ignore", "invalid value encountered in.*", RuntimeWarning)
import os

case_type = [
    'starting_left_turn',
    'starting_right_turn',
    'starting_straight_traffic_light_intersection_traversal',
    'stopping_with_lead',
    'high_lateral_acceleration',
    'high_magnitude_speed',
    'low_magnitude_speed',
    'traversing_pickup_dropoff',
    'waiting_for_pedestrian_to_cross',
    'behind_long_vehicle',
    'stationary_in_traffic',
    'near_multiple_vehicles',
    'changing_lane',
    'following_lane_with_lead'
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--openloop', '-o', action='store_true')
    parser.add_argument('--planner', '-p', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--planner_type', type=str, default=None)
    parser.add_argument('--ref', type=str, default=None)
    parser.add_argument('--type', type=int, default=None)
    parser.add_argument('--llm_inf_step', type=int, default=1)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--short_ins', type=int, default=-1)
    parser.add_argument('--disable_refpath', action='store_true')
    parser.add_argument('--ins_wo_stop', action='store_true')
    parser.add_argument('--refine', action='store_true')
    parser.add_argument('--base_model', type=str, default=None)
    parser.add_argument('--simulation_root_path', type=str, default=None)
    # 新增用于ONNX和TensorRT模型路径的参数
    parser.add_argument('--onnx_model_path', type=str, default=None, help='Path to the ONNX model')
    parser.add_argument('--tensorrt_model_path', type=str, default=None, help='Path to the TensorRT model')
    parser.add_argument('--inference_model_type', type=str, default=None, help='Type of inference model to use (torch onnx tensorrt)')
    return parser.parse_args()

args = parse_args()


# Location of path with all simulation configs
CONFIG_PATH = '../../nuplan/planning/script/config/simulation'
CONFIG_NAME = 'default_simulation'
sim_path = args.simulation_root_path
if args.type is not None:
    if not os.path.exists(Path(sim_path) / 'simulation' / args.save_dir):
        os.makedirs(Path(sim_path) / 'simulation' / args.save_dir, exist_ok=True)
    SAVE_DIR = Path(sim_path) / 'simulation' / args.save_dir / case_type[args.type]
else:
    SAVE_DIR = Path(sim_path) / 'simulation' / args.save_dir
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

PLANNER = args.planner_type
scenario_filter_type = 'test_scenarios_hard20'


CHALLENGE = 'open_loop_boxes' if args.openloop else 'closed_loop_reactive_agents'  # [open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents]

DATASET_PARAMS = [
    'scenario_builder=nuplan_challenge',
    f'scenario_filter={scenario_filter_type}',
    f'+planner.{PLANNER}.ins_mode={args.ref}',
    f'+planner.{PLANNER}.ins_wo_stop={args.ins_wo_stop}',
    f'planner.{PLANNER}.disable_refpath={args.disable_refpath}',
    f'+planner.{PLANNER}.finetune_model_path={args.planner}',
    f'+planner.{PLANNER}.model_name_or_path={args.base_model}',
    f'+planner.{PLANNER}.enable_pdm_scorer_in_multirefpath={args.refine}',
    f'+planner.{PLANNER}.lora_r={args.lora_r}',
    f'+planner.{PLANNER}.llm_inf_step={args.llm_inf_step}',
    f'+planner.{PLANNER}.short_ins={args.short_ins}',
    f'scenario_filter.scenario_types=[{case_type[args.type]}]',
    'scenario_filter.num_scenarios_per_type=20',
    "hydra.searchpath=[pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]",
    #"hydra.searchpath=[file:///abspath/to/asyncdriver/nuplan/planning/script/config/common, file:///abspath/to/asyncdriver/nuplan/planning/script/experiments]",
]
print("--------",args.inference_model_type)
# 新增两个参数写入config
if args.onnx_model_path is not None:
    DATASET_PARAMS.append(f'+planner.{PLANNER}.onnx_model_path={args.onnx_model_path}')
if args.tensorrt_model_path is not None:
    DATASET_PARAMS.append(f'+planner.{PLANNER}.tensorrt_model_path={args.tensorrt_model_path}')
if args.inference_model_type is not None:
    DATASET_PARAMS.append(f'+planner.{PLANNER}.inference_model_type={args.inference_model_type}')


# Name of the experiment
EXPERIMENT = 'llama4drive_experiment'

# Initialize configuration management system
hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
hydra.initialize(config_path=CONFIG_PATH)
# Compose the configuration
cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
    f'experiment_name={EXPERIMENT}',
    f'group={SAVE_DIR}',
    f'planner={PLANNER}',
    f'+simulation={CHALLENGE}',
    'worker=sequential',
    *DATASET_PARAMS,
])

# Run the simulation loop (real-time visualization not yet supported, see next section for visualization)
main_simulation(cfg)

# Simple simulation folder for visualization in nuBoard
simple_simulation_folder = cfg.output_dir
