# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

import logging
import os
import json
from typing import Any
from uuid import uuid4
from datetime import datetime

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length

        tool_config_path = getattr(self.config.data, 'tool_config_path', None)
        if tool_config_path:
            tool_list = initialize_tools_from_config(tool_config_path)
            self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        else:
            self.tool_schemas = []
        
        # ========== 初始化日志目录 ==========
        # 命令行 data.caption_log_dir=None 在 Hydra 中会变成字符串 "None"，需视为未设置
        raw = getattr(self.config.data, 'caption_log_dir', None)
        if raw is None or str(raw).strip().lower() in ("none", "null", ""):
            self.caption_log_dir = None
            self.log_file = None
        else:
            self.caption_log_dir = str(raw).strip()
            os.makedirs(self.caption_log_dir, exist_ok=True)
            self.log_file = os.path.join(
                self.caption_log_dir,
                f"captions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            )

    def _log_caption(self, media_path: str, caption: str, prompt_text: str, extra_info: dict = None):
        """记录生成的 caption 到本地文件"""
        if not self.log_file:
            return
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "media_path": media_path,  # 改名：video_path -> media_path
            "prompt": prompt_text,
            "caption": caption,
            **(extra_info or {})
        }
        
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log caption: {e}")

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # 1. extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # 2. apply chat template and tokenize
        prompt_ids_for_training, prompt_ids_for_vllm = await self.apply_chat_template(
            messages,
            tools=self.tool_schemas,
            images=images,
            videos=videos,
        )

        # 3. generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids_for_vllm,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        
        metrics["num_preempted"] = getattr(output, 'num_preempted', -1) or -1
        response_mask = [1] * len(output.token_ids)

        # ========== 记录生成的 caption ==========
        if self.log_file:
            # 解码生成的 caption
            caption_text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
            
            # 获取媒体路径（图片或视频）
            media_path = None
            media_type = None
            
            # 尝试从 raw_prompt 中获取图片或视频路径
            for msg in kwargs.get("raw_prompt", []):
                if isinstance(msg.get("content"), list):
                    for item in msg["content"]:
                        if isinstance(item, dict):
                            if item.get("type") == "image":
                                media_path = item.get("image")
                                media_type = "image"
                                break
                            elif item.get("type") == "video":
                                media_path = item.get("video")
                                media_type = "video"
                                break
                    if media_path:
                        break
            
            # 获取 prompt 文本
            prompt_text = ""
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        prompt_text = content
                    elif isinstance(content, list):
                        prompt_text = " ".join([
                            item.get("text", "") for item in content 
                            if isinstance(item, dict) and item.get("type") == "text"
                        ])
                    break
            
            # 记录
            self._log_caption(
                media_path=str(media_path) if media_path else "unknown",
                caption=caption_text,
                prompt_text=prompt_text[:500],
                extra_info={
                    "media_type": media_type or "unknown",  # 新增：记录媒体类型
                    "prompt_length": len(prompt_ids_for_training),
                    "response_length": len(output.token_ids),
                }
            )
        # ========== 记录结束 ==========

        result = AgentLoopOutput(
            prompt_ids=prompt_ids_for_training,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids_for_training) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
        )
        
        return result