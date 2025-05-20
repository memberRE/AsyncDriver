import threading
import numpy as np
import torch
import pycuda.driver as cuda
import tensorrt as trt
import onnxruntime as ort

class TRTInferSingleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, engine_path, device_id=0):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(TRTInferSingleton, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, engine_path, device_id=0):
        if self._initialized:
            return
        self.device_id = device_id
        self.engine_path = engine_path

        # CUDA上下文只能创建一次
        cuda.init()
        self._device = cuda.Device(self.device_id)
        self._context = self._device.make_context()
        self._context.pop()
        self._engine_cache = {}
        self._context_cache = {}
        self._trt_logger = trt.Logger(trt.Logger.ERROR) if trt else None

        self._initialized = True

    def get_engine_and_context(self, engine_path):
        if engine_path not in self._engine_cache:
            # 加载engine
            with open(engine_path, "rb") as f:
                engine = trt.Runtime(self._trt_logger).deserialize_cuda_engine(f.read())
            self._engine_cache[engine_path] = engine
        engine = self._engine_cache[engine_path]
        if engine_path not in self._context_cache:
            context = engine.create_execution_context()
            self._context_cache[engine_path] = context
        context = self._context_cache[engine_path]
        return engine, context

    def infer_hidden_states(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        self._context.push()
        stream = cuda.Stream()
        with torch.no_grad():
            bs, seq_len, hidden_size = inputs_embeds.shape
            size_emb = bs * seq_len * hidden_size
            size_mask = bs * seq_len
            size_pos  = bs * seq_len
            host = {
                "emb":   cuda.pagelocked_empty(size_emb,   np.float32),
                "mask":  cuda.pagelocked_empty(size_mask,  np.int64),
                "pos":   cuda.pagelocked_empty(size_pos,   np.int64),
                "out":   cuda.pagelocked_empty(size_emb,   np.float32),
            }
            dev = {k: cuda.mem_alloc(buf.nbytes) for k, buf in host.items()}

            # 获取engine和context（缓存）
            engine, context = self.get_engine_and_context(self.engine_path)

            host["emb"][:]  = inputs_embeds.detach().cpu().numpy().ravel()
            host["mask"][:] = attention_mask.detach().cpu().numpy().ravel()
            host["pos"][:]  = position_ids.detach().cpu().numpy().ravel()

            cuda.memcpy_htod_async(dev["emb"],  host["emb"],  stream)
            cuda.memcpy_htod_async(dev["mask"], host["mask"], stream)
            cuda.memcpy_htod_async(dev["pos"],  host["pos"],  stream)

            names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
            context.set_tensor_address(names[0], int(dev["emb"]))
            context.set_tensor_address(names[1], int(dev["mask"]))
            context.set_tensor_address(names[2], int(dev["pos"]))
            context.set_tensor_address(names[3], int(dev["out"]))
            context.set_input_shape(names[0], (bs, seq_len, hidden_size))
            context.set_input_shape(names[1], (bs, seq_len))
            context.set_input_shape(names[2], (bs, seq_len))

            context.execute_async_v3(stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(host["out"], dev["out"], stream)
            stream.synchronize()

            hs_gpu = torch.tensor(
                host["out"].reshape(bs, seq_len, hidden_size),
                device=inputs_embeds.device
            )
        self._context.pop()
        return hs_gpu

    def clear_cache(self):
        """手动清空engine/context缓存与CUDA上下文"""
        self._engine_cache.clear()
        self._context_cache.clear()
        try:
            self.context.pop()
        except Exception:
            pass

    def __del__(self):
        self.clear_cache()



class ONNXInferSingleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, onnx_path, providers=['CUDAExecutionProvider']):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ONNXInferSingleton, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, onnx_path, providers=['CUDAExecutionProvider']):
        if self._initialized:
            return
        self.onnx_path = onnx_path
        self.providers = providers
        self._session_cache = {}
        self._initialized = True

    def get_cached_session(self, path):
        if path not in self._session_cache:
            self._session_cache[path] = ort.InferenceSession(path, providers=self.providers)
        return self._session_cache[path]

    def infer_hidden_states(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        # ONNX 输入准备
        onnx_inputs = {
            "inputs_embeds": inputs_embeds.detach().cpu().numpy().astype(np.float16),
            "attention_mask": attention_mask.detach().cpu().numpy().astype(np.int64),
            "position_ids": position_ids.detach().cpu().numpy().astype(np.int64)
        }
        # 执行推理
        ort_session = self.get_cached_session(self.onnx_path)
        hidden_states = ort_session.run(None, onnx_inputs)[0]
        hidden_states = torch.from_numpy(hidden_states).to(torch.float32).to(inputs_embeds.device)
        return hidden_states

    def clear_cache(self):
        self._session_cache.clear()

    def __del__(self):
        self.clear_cache()