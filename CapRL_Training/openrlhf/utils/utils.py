from typing import List

import torch
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoProcessor, AutoConfig
import time
from contextlib import contextmanager

import logging

# (私有) 全局变量，用于保存原始的日志级别
_original_ray_log_level = None


def control_ray_logs(mode: str):
    """
    一个开关函数，用于开启或关闭 Ray 的日志。

    Args:
        mode (str): "on" 或 "off"
    """
    global _original_ray_log_level

    # 1. 获取 "ray" 根 logger，这将控制所有 ray.* 子模块
    ray_logger = logging.getLogger("ray")

    if mode.lower() == "off":
        # --- 关闭日志 ---

        # 仅在日志尚未被关闭时 (即 _original_ray_log_level 为 None)
        # 才保存当前级别。这防止你连续调用 "off" 时覆盖了原始设置。
        if _original_ray_log_level is None:
            _original_ray_log_level = ray_logger.level

        # 设置为 CRITICAL，这实际上“静音”了所有
        # 低于 CRITICAL 级别 (INFO, WARNING, ERROR) 的日志。
        ray_logger.setLevel(logging.CRITICAL)
        # print("--- Ray logs set to OFF (CRITICAL) ---") # 你可以取消注释这行来确认状态

    elif mode.lower() == "on":
        # --- 开启日志 ---

        # 仅在日志确实被关闭过 (即 _original_ray_log_level 不是 None)
        # 才恢复它。
        if _original_ray_log_level is not None:
            ray_logger.setLevel(_original_ray_log_level)
            # print(f"--- Ray logs set to ON (Restored) ---") # 你可以取消注释这行

            # 恢复后，重置状态，以便下次“关闭”
            _original_ray_log_level = None
        # else:
        # 如果 _original_ray_log_level 是 None，说明日志本来就是开的
        # print("--- Ray logs are already ON ---") # 你可以取消注释这行

    else:
        print(f"Error: Unknown mode '{mode}' for control_ray_logs. Use 'on' or 'off'.")

@contextmanager
def timing(description: str = "Code block", logger=None) -> None:
    """
    一个上下文管理器，用于计算并打印某段代码的执行时间。

    用法:
    with timing("数据处理"):
        # 你的代码
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        if logger:
            logger.info(f"✨ {description} completed in {elapsed_time:.4f} seconds")
        else:
            print(f"✨ {description} completed in {elapsed_time:.4f} seconds")


def get_strategy(args):
    from openrlhf.utils.deepspeed import DeepspeedStrategy

    strategy = DeepspeedStrategy(
        seed=getattr(args, "seed", 42),
        full_determinism=getattr(args, "full_determinism", False),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
        train_batch_size=getattr(args, "train_batch_size", 128),
        zero_stage=args.zero_stage,
        bf16=getattr(args, "bf16", True),
        args=args,
    )
    return strategy


def get_vl_processor(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    # TODO: Maybe better max_pixels set methods for other vl model
    min_pixels = int(os.getenv("MIN_PIXELS", 4*28*28))
    max_pixels = int(os.getenv("MAX_PIXELS", 1000 * 1000))
    cfg = AutoConfig.from_pretrained(pretrain)
    processor = AutoProcessor.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast, min_pixels=min_pixels, max_pixels=max_pixels)
    processor.image_token_id = cfg.image_token_id
    tokenizer = AutoTokenizer.from_pretrained(pretrain, use_fast=use_fast)
    # tokenizer = processor.tokenizer
    tokenizer.padding_side = padding_side
    processor.tokenizer = tokenizer
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
    return processor


def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if model is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def convert_token_to_id(token, tokenizer):
    if isinstance(token, str):
        token = tokenizer.encode(token, add_special_tokens=False)
        assert len(token) == 1
        return token[0]
    else:
        raise ValueError("token should be int or str")


def zero_pad_sequences(
    sequences: List[torch.Tensor], side: str = "left", value: int = 0, stack: bool = False
) -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    if stack:
        return torch.stack(padded_sequences, dim=0)
    else:
        return torch.cat(padded_sequences, dim=0)


def remove_pad_token(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Remove the pad token. Return tensors and not lists.

    Args:
        input_ids shape: [bs, seq_length]
        attention_mask shape: [bs, seq_length]
    Returns:
        no_padding_batch(List[Tensor[int]]): contains the rmpad token ids per query.
    """
    no_padding_batch = []
    for ids, mask in zip(input_ids, attention_mask):
        # Fix for both left and right padding
        no_padding_batch.append((ids[mask.bool()]))
    return no_padding_batch
