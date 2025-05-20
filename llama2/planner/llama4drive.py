import logging
import threading
import sys, os
from functools import wraps
from tqdm import tqdm
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    HfArgumentParser,
    TrainingArguments,
    BitsAndBytesConfig,
    is_torch_tpu_available,
    set_seed,
)
from peft import (  # noqa: E402
    LoraConfig,
    PeftModel,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from llama2.model_llama4drive import LlamaForCausalLM
from llama2.model_llama4drive import LlamaForCausalLM, ModelWithLoRA
import torch
import numpy as np


DEBUG = False
RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"

def get_model(model_name_or_path=None,
              finetune_model_path=None,
              add_special_tokens='<map>,</map>',
              resize_token_embeddings=True,
              devices=None,
              **kwargs):
    if len(os.listdir(finetune_model_path))==1:
            finetune_model_path = os.path.join(finetune_model_path, os.listdir(finetune_model_path)[0])
    tokenizer = AutoTokenizer.from_pretrained(finetune_model_path, use_fast=False)

    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"
    config_kwargs = {
        "cache_dir": None,
        "revision": 'main',
        "use_auth_token": None,
    }
    config = AutoConfig.from_pretrained(model_name_or_path, **config_kwargs)
    config.feature_len = kwargs.get('feature_len', 80)
    config.map_former = kwargs.get('map_former', False)
    config.mapEncoder_pretrain_weight = kwargs.get('mapEncoder_pretrain_weight', None)
    config.enable_lora = kwargs.get('enable_lora', False)
    config.pool_mode = kwargs.get('pool_mode', 'all_mean')
    config.use_all_tokens = kwargs.get('use_all_tokens', False)
    config.map_insize = kwargs.get('map_insize', 256)

    config.use_all_tokens = kwargs.get('use_all_tokens', False)
    config.adapter_fusion = kwargs.get('adapter_fusion', False)

    config.llm_inf_step = kwargs.get('llm_inf_step', 1)
    config.lora_r = kwargs.get('lora_r', 16)
    config.onnx_model_path = kwargs.get('onnx_model_path', None)
    config.tensorrt_model_path = kwargs.get('tensorrt_model_path', None)
    config.inference_model_type = kwargs.get('inference_model_type', None)

    if add_special_tokens is not None:
        additional_special_tokens = add_special_tokens.split(',')
        special_token_ids = tokenizer.convert_tokens_to_ids(additional_special_tokens)
        special_token_dict = dict(zip(additional_special_tokens, special_token_ids))
        config.special_token_dict = special_token_dict

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype=torch.float16,
        device_map=devices if devices else 'auto',
        quantization_config=bnb_config
    )

    if config.enable_lora:
        # if model_args.layers_to_transform is not None:
        #     model_args.layers_to_transform = [int(num) for num in model_args.layers_to_transform.strip().split(',')]
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=32,
            target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj'],
            fan_in_fan_out = False,
            lora_dropout=0.05,
            inference_mode=False,
            bias="none",
            task_type="CAUSAL_LM",
            layers_to_transform=None
        )
        print('\n\n================== Lora Cfg =================')
        print(lora_config)
        print('\n\n')
        model = ModelWithLoRA(model, lora_config)
    else:
        lora_config = None

    embedding_size = model.get_input_embeddings().weight.shape[0]

    try:
        if len(tokenizer) > embedding_size and resize_token_embeddings:
            print('resize_token_embeddings from {} to {}'.format(embedding_size, len(tokenizer)))
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=2)
        model = prepare_model_for_kbit_training(model)
        model.load_weights(finetune_model_path)
        if not config.enable_lora:
            model.resume_from_checkpoint(finetune_model_path)
    except:
        if len(tokenizer) > embedding_size and resize_token_embeddings:
            print('resize_token_embeddings from {} to {}'.format(embedding_size, len(tokenizer)))
            model.resize_token_embeddings(len(tokenizer))
        model = prepare_model_for_kbit_training(model)
        model.load_weights(finetune_model_path)
        if not config.enable_lora:
            model.resume_from_checkpoint(finetune_model_path)

    model = model.eval()
    return model, tokenizer

def padding_token(input_ids_list, padding_id, padding_side='left'):
    max_length = max([len(i) for i in input_ids_list])
    for i in range(len(input_ids_list)):
        if padding_side == 'left':
            input_ids_list[i] = [padding_id] * (max_length - len(input_ids_list[i])) + input_ids_list[i].tolist()
        elif padding_side == 'right':
            input_ids_list[i] = input_ids_list[i].tolist() + [padding_id] * (max_length - len(input_ids_list[i]))
        else:
            raise ValueError('padding_side must be left or right!')
    return torch.tensor(input_ids_list)

class LLAMA2DriveModel:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, model_config):
        with cls._lock:
            logging.error('!!!!!!!!!!!!!!!! instantiate LLAMA2DriveModel !!!!!!!!!!')
            if cls._instance is None:
                cls._instance = super(LLAMA2DriveModel, cls).__new__(cls)
                cls._initialize_model(model_config)
                cls._instance.infer_locker = threading.Lock()
                cls.ins_mode = model_config['ins_mode']
                cls.ins_wo_stop = model_config['ins_wo_stop']
                cls.lora_r = model_config['lora_r']
            instance = cls._instance
        return instance

    @classmethod
    def _initialize_model(cls, config):
        print("Loading LLM...")
        if isinstance(config, list):
            config = {k:v for d in config for k,v in d.items()}
        cls.model, cls.tokenizer = get_model(
            **config
        )
        cls.model_loaded = True
        cls.diversity = config.get('diversity_ins', False)

    def generate_prompt(self, lane, return_navi=False):
        if isinstance(lane, torch.Tensor):
            lane = lane.cpu().numpy()
        if self.ins_mode == 'None':
            if lane is None:
                instruction = 'Keep going.'
                logging.info('No lane information, use default instruction')
            else:
                cmd_ds, instruction = self.get_instruction(lane)
        elif self.ins_mode == 'limit':
            if lane is None:
                instruction = 'Keep going'
                logging.info('No lane information, use default instruction')
            else:
                cmd_ds, instruction = self.get_instruction(lane, limit=4)
        if return_navi:
            try:
                nav_inst = cmd_ds[0][0]
                if 'go straight in' in nav_inst:
                    cmd = np.array([1, 0, 0, 0])
                elif 'turn left in' in nav_inst:
                    cmd = np.array([0, 1, 0, 0])
                elif 'turn right in' in nav_inst:
                    cmd = np.array([0, 0, 1, 0])
                elif 'stop' in nav_inst:
                    cmd = np.array([0, 0, 0, 1])
                else:
                    import ipdb; ipdb.set_trace()
            except:
                cmd = np.array([1, 0, 0, 0])
            logging.info(cmd)
        logging.info(instruction)
        messages = f"""
    Role: You are now an autonomous driving driver, and I will provide you with the environment information including Ego Car Information, Agents Information and Map Information.\n\nEnvironment: <map>\n\nNevigation instructions: {instruction}. \n\nPlease predict the future waypoints of the ego car based on the given environmental information and nevigation instrucions.\n\nFinal Answer:\n
    """
        if return_navi:
            return messages, cmd
        return messages

    def get_instruction(self, ego_future_poses, threshold=0.5, return_prompt=True, limit=99999999, diversity=False):
        # dis_norm = np.linalg.norm(np.diff(np.concatenate([ego_future_poses[:1,:-1], ego_future_poses[:,:-1]], axis=0), n=1, axis=0), axis=1)
        ego_future_poses = ego_future_poses[:, :3]
        dis_norm = np.linalg.norm(np.diff(ego_future_poses[:, :-1], n=1, axis=0), axis=1)
        dis_cum = np.cumsum(dis_norm, axis=0)

        cur_cmd = None
        cur_dis = 0
        instruction = ''
        cmd_ls = []
        dis_ls = []
        tmp_dis_ls = []
        time_ls = []
        if self.ins_wo_stop:
            if dis_cum[-1]<0.5:
                cur_cmd = 'stop. '
                cur_dis = 0
            else:
                for heading, (idx, dis), d_n in zip(ego_future_poses[1:, 2], enumerate(dis_cum), dis_norm):
                    if heading > threshold:
                        cmd = 'turn left in ' if not self.diversity else 'Veer left in '
                    elif heading < -threshold:
                        cmd = 'turn right in ' if not self.diversity else 'Make a right-hand turn in '
                    else:
                        cmd = 'go straight in ' if not self.diversity else 'Stay on this route for '
                    if cur_cmd == None:
                        cur_cmd = cmd
                    elif cmd != cur_cmd:
                        cmd_ls.append(cur_cmd)
                        dis_ls.append(np.round(dis_cum[idx - 1] - cur_dis, 2))
                        cur_dis = dis_cum[idx - 1]
                        cur_cmd = cmd
        else:
            for heading, (idx, dis), d_n in zip(ego_future_poses[1:, 2], enumerate(dis_cum), dis_norm):
                if heading > threshold:
                    cmd = 'turn left in ' if not self.diversity else 'Veer left in '
                elif heading < -threshold:
                    cmd = 'turn right in ' if not self.diversity else 'Make a right-hand turn in '
                elif d_n > 0.1:
                    cmd = 'go straight in ' if not self.diversity else 'Stay on this route for '
                else:
                    cmd = 'stop. ' if not self.diversity else 'Bring it to a halt.'
                if cur_cmd == None:
                    cur_cmd = cmd
                elif cmd != cur_cmd:
                    # if return_prompt:
                    #     instruction += cmd
                    #     if cmd != 'stop. ':
                    #         instruction += (str(np.round(dis-cur_dis, 2)) + ' meters. ')
                    cmd_ls.append(cur_cmd)
                    # if 'go straight' in cur_cmd:
                    #     import pdb; pdb.set_trace()
                    dis_ls.append(np.round(dis_cum[idx - 1] - cur_dis, 2))
                    # time_ls.append(idx*0.5)
                    if 'stop' not in cur_cmd and 'halt' not in cur_cmd:
                        cur_dis = dis_cum[idx - 1]
                    cur_cmd = cmd


        cmd_ls.append(cur_cmd)
        dis_ls.append(np.round(dis - cur_dis, 2))
        num = 0
        if return_prompt:
            for c, d in zip(cmd_ls, dis_ls):
                if num >= limit:
                    break
                instruction += c
                if 'stop' not in c:
                    instruction += (str(np.round(d, 2)) + ' meters. ')
                num += 1
                
        # print('whole route is %s meters.'%str(dis_cum[-1]))
        return [cmd_ls, dis_ls], instruction

    def inference(self, data, ref_path, cur_iter):
        if not hasattr(self, 'model_loaded') or not self.model_loaded:
            raise RuntimeError("Model not loaded properly.")
        tokenizer = self.tokenizer
        messages = self.generate_prompt(ref_path)
        messages = messages.replace('<map>', '<map></map>')
        map_info = data
        input_dict = {
            'ego_agent_past': map_info.get('ego_agent_past', None),
            'neighbor_agents_past': map_info.get('neighbor_agents_past', None),
            'route_lanes': map_info.get('route_lanes', None),
            'map_lanes': map_info.get('map_lanes', None),
            'map_crosswalks': map_info.get('map_crosswalks', None),
            'ego_future': map_info.get('ego_agent_future', None),
            'neighbors_future': map_info.get('neighbor_agents_future', None),
            'cur_iter': cur_iter,
        }
        input_ids = tokenizer([messages], return_tensors="pt", add_special_tokens=False).input_ids[0]
        input_ids = padding_token([input_ids], tokenizer.pad_token_id, padding_side='left').cuda()
        input_ids = torch.cat([torch.zeros((input_ids.shape[0], 1), dtype=torch.int64)+1, input_ids.cpu(), torch.ones((input_ids.shape[0], 1),  dtype=torch.int64)+1], dim=1).cuda()
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            with self.infer_locker:
                output = self.model(input_ids=input_ids, attention_mask=attention_mask, inference=True, **input_dict)
                # logging.error(f'{torch.cuda.memory_allocated() / 1024 / 1024}')
                # torch.cuda.empty_cache()
        return output

    def debug_inference(self, input_dict):
        tokenizer = self.tokenizer
        messages = input_dict.pop('messages')
        input_ids = tokenizer([messages], return_tensors="pt", add_special_tokens=False).input_ids[0]
        input_ids = padding_token([input_ids], tokenizer.pad_token_id, padding_side='left').cuda()
        input_ids = torch.cat([torch.zeros((input_ids.shape[0], 1), dtype=torch.int64)+1, input_ids.cpu(), torch.ones((input_ids.shape[0], 1),  dtype=torch.int64)+1], dim=1).cuda()
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **input_dict)