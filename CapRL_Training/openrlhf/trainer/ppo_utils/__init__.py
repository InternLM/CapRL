from .kl_controller import AdaptiveKLController, FixedKLController
from .replay_buffer import NaiveReplayBuffer
from .data_processor import BaseDataProcessor, Qwen2VLDataProcessor

InternVLDataProcessor = None
try:
    from transformers import InternVLProcessor
except:
    InternVLProcessor = None
from transformers import Qwen2VLProcessor, Qwen2_5_VLProcessor, Qwen3VLProcessor

DATA_PROCESSOR_MAP = {
    Qwen2VLProcessor: Qwen2VLDataProcessor,
    Qwen2_5_VLProcessor: Qwen2VLDataProcessor,
    InternVLProcessor: InternVLDataProcessor,
    Qwen3VLProcessor: Qwen2VLDataProcessor,
}

__all__ = [
    "AdaptiveKLController",
    "FixedKLController",
    "NaiveReplayBuffer",
]
