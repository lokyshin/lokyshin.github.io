"""
title: DeepSeek R1
author: zgccrui
description: 在OpenWebUI中显示DeepSeek R1模型的思维链 - 仅支持0.5.6及以上版本
version: 1.2.5
licence: MIT
"""

import json
import httpx
import re
from typing import AsyncGenerator, Callable, Awaitable
from pydantic import BaseModel, Field
import asyncio

# 关键：导入 Pipeline 基类
from pipelines.base import Pipeline

class DeepSeekR1(Pipeline):  # 关键：继承 Pipeline（或BasePipeline）
    class Valves(BaseModel):
        DEEPSEEK_API_BASE_URL: str = Field(
            default="https://api.deepseek.com/v1",
            description="DeepSeek API的基础请求地址",
        )
        DEEPSEEK_API_KEY: str = Field(
            default="", description="用于身份验证的DeepSeek API密钥，可从控制台获取"
        )
        DEEPSEEK_API_MODEL: str = Field(
            default="deepseek-reasoner",
            description="API请求的模型名称，默认为 deepseek-reasoner",
        )

    def __init__(self):
        super().__init__()
        # 关键：Pipeline 需要有一个 name/title/description 等标识
        self.name = "deepseek_r1_thinking"
        self.description = "DeepSeek R1 demo pipeline for OpenWebUI"

        self.valves = self.Valves()
        self.data_prefix = "data: "
        self.thinking = -1  # -1:未开始 0:思考中 1:已回答
        self.emitter = None

    def pipes(self):
        """可选：如果你想在前端显示可用模型列表，保留此方法"""
        return [
            {
                "id": self.valves.DEEPSEEK_API_MODEL,
                "name": self.valves.DEEPSEEK_API_MODEL,
            }
        ]

    # 关键：把你的主要执行逻辑放在 run(...) 方法里
    async def run(
        self, 
        body: dict, 
        __event_emitter__: Callable[[dict], Awaitable[None]] = None
    ) -> AsyncGenerator[str, None]:
        """主处理管道（将原先 pipe(...) 的逻辑迁移到这里）"""

        self.thinking = -1
        self.emitter = __event_emitter__

        # 验证配置
        if not self.valves.DEEPSEEK_API_KEY:
            yield json.dumps({"error": "未配置API密钥"}, ensure_ascii=False)
            return

        headers = {
            "Authorization": f"Bearer {self.valves.DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }

        try:
            # 模型ID提取
            model_id = body["model"].split(".", 1)[-1]
            payload = {**body, "model": model_id}

            # 处理消息以防止连续相同角色（你的逻辑原样保留）
            messages = payload["messages"]
            i = 0
            while i < len(messages) - 1:
                if messages[i]["role"] == messages[i + 1]["role"]:
                    alternate_role = (
                        "assistant" if messages[i]["role"] == "user" else "user"
                    )
                    messages.insert(
                        i + 1,
                        {"role": alternate_role, "content": "[Unfinished thinking]"},
                    )
                i += 1

            # 发起API请求
            async with httpx.AsyncClient(http2=True) as client:
                async with client.stream(
                    "POST",
                    f"{self.valves.DEEPSEEK_API_BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=300,
                ) as response:
                    if response.status_code != 200:
                        error = await response.aread()
                        yield self._format_error(response.status_code, error)
                        return

                    # 流式处理响应
                    async for line in response.aiter_lines():
                        if not line.startswith(self.data_prefix):
                            continue

                        data = json.loads(line[len(self.data_prefix):])
                        choice = data.get("choices", [{}])[0]

                        # 结束条件判断
                        if choice.get("finish_reason"):
                            return

                        # 状态机处理
                        state_output = await self._update_thinking_state(
                            choice.get("delta", {})
                        )
                        if state_output:
                            yield state_output  # 发送状态标记
                            if state_output == "<think>":
                                yield "\n"

                        # 内容处理并立即发送
                        content = self._process_content(choice["delta"])
                        if content:
                            if content.startswith("<think>"):
                                # 这里保留原先的正则处理
                                match = re.match(r"^<think>", content)
                                if match:
                                    content = re.sub(r"^<think>", "", content)
                                    yield "<think>"
                                    await asyncio.sleep(0.1)
                                    yield "\n"
                            elif content.startswith("</think>"):
                                match = re.match(r"^</think>", content)
                                if match:
                                    content = re.sub(r"^</think>", "", content)
                                    yield "</think>"
                                    await asyncio.sleep(0.1)
                                    yield "\n"

                            yield content

        except Exception as e:
            yield self._format_exception(e)

    async def _update_thinking_state(self, delta: dict) -> str:
        """更新思考状态机（简化版）"""
        state_output = ""

        # 状态转换：未开始 -> 思考中
        if self.thinking == -1 and delta.get("reasoning_content"):
            self.thinking = 0
            state_output = "<think>"

        # 状态转换：思考中 -> 已回答
        elif self.thinking == 0 and not delta.get("reasoning_content"):
            self.thinking = 1
            state_output = "\n</think>\n\n"

        return state_output

    def _process_content(self, delta: dict) -> str:
        """直接返回处理后的内容"""
        return delta.get("reasoning_content", "") or delta.get("content", "")

    def _format_error(self, status_code: int, error: bytes) -> str:
        """错误格式化保持不变"""
        try:
            err_msg = json.loads(error).get("message", error.decode(errors="ignore"))[:200]
        except:
            err_msg = error.decode(errors="ignore")[:200]
        return json.dumps(
            {"error": f"HTTP {status_code}: {err_msg}"}, ensure_ascii=False
        )

    def _format_exception(self, e: Exception) -> str:
        """异常格式化保持不变"""
        err_type = type(e).__name__
        return json.dumps({"error": f"{err_type}: {str(e)}"}, ensure_ascii=False)
