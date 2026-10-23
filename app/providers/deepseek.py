import json
import time
import httpx

from app.core.config import ProviderConfig, ModelConfig
from app.core.exceptions import ProviderError, AllKeysFailedError
from app.providers.base import BaseProvider, KeyPool
from app.providers.streaming import (
    decode_sse_payload,
    normalize_stream_error,
    parse_stream_chunk,
)
from app.schemas.chat import (
    ChatRequest, ChatResponse, ChatChoice, ChatMessage,
    StreamChunk, Delta, StreamChoice, Usage,
)


class DeepSeekProvider(BaseProvider):
    def __init__(self, config: ProviderConfig, key_pool: KeyPool):
        super().__init__(config, key_pool)
        self.client = httpx.AsyncClient(timeout=None if config.timeout <= 0 else config.timeout)

    def _headers(self, api_key: str) -> dict:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _build_request(self, request: ChatRequest, model_config: ModelConfig) -> dict:
        data = {
            "model": model_config.name,
            "messages": [m.model_dump() for m in request.messages],
        }
        for attr in ("temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty", "stop"):
            val = getattr(request, attr, None)
            if val is not None:
                data[attr] = val
        return data

    def _parse_response(self, body: dict, model: str) -> ChatResponse:
        choice = body["choices"][0]
        msg = choice["message"]
        usage_data = body.get("usage")
        return ChatResponse(
            id=body["id"],
            created=body.get("created", 0),
            model=model,
            choices=[
                ChatChoice(
                    index=choice.get("index", 0),
                    message=ChatMessage(
                        role=msg["role"],
                        content=msg.get("content", ""),
                    ),
                    finish_reason=choice.get("finish_reason"),
                )
            ],
            usage=Usage(**usage_data) if usage_data else None,
        )

    def _parse_reasoner_response(self, body: dict, model: str) -> ChatResponse:
        choice = body["choices"][0]
        msg = choice["message"]
        reasoning = msg.get("reasoning_content", "")
        content = msg.get("content", "")
        final = (f"【reasoning】\n{reasoning}\n\n【answer】\n{content}") if reasoning else content
        usage_data = body.get("usage")
        return ChatResponse(
            id=body["id"],
            created=body.get("created", 0),
            model=model,
            choices=[
                ChatChoice(
                    message=ChatMessage(role="assistant", content=final),
                    finish_reason=choice.get("finish_reason"),
                )
            ],
            usage=Usage(**usage_data) if usage_data else None,
        )

    def _parse_chunk(self, data: dict, model: str) -> StreamChunk | None:
        if not data.get("choices"):
            usage_data = data.get("usage")
            if not usage_data:
                return None
            return StreamChunk(
                id=data.get("id", ""),
                created=data.get("created", 0),
                model=model,
                choices=[],
                usage=Usage(**usage_data),
            )
        choice = data["choices"][0]
        delta = choice.get("delta", {})
        return StreamChunk(
            id=data.get("id", ""),
            created=data.get("created", 0),
            model=model,
            choices=[
                StreamChoice(
                    index=choice.get("index", 0),
                    delta=Delta(
                        role=delta.get("role"),
                        content=delta.get("reasoning_content") or delta.get("content"),
                    ),
                    finish_reason=choice.get("finish_reason"),
                )
            ],
            usage=Usage(**data["usage"]) if data.get("usage") else None,
        )

    def _decode_json_response(self, resp: httpx.Response) -> dict:
        try:
            return resp.json()
        except json.JSONDecodeError:
            content_type = resp.headers.get("content-type", "")
            text = resp.text[:300]
            raise ProviderError("deepseek", resp.status_code, f"non-json upstream response ({content_type}): {text}")

    async def chat_completions(
        self, request: ChatRequest, model_config: ModelConfig
    ) -> ChatResponse:
        self.raise_if_unhealthy()
        started_at = time.monotonic()
        payload = self._build_request(request, model_config)

        last_error = None
        for _ in range(len(self.key_pool.keys)):
            acquired = self.key_pool.acquire()
            if not acquired:
                continue
            key, kidx = acquired

            try:
                resp = await self._try_send(payload, key.key)
                body = self._decode_json_response(resp)
                is_reasoner = model_config.name == "deepseek-v4-pro"
                result = (self._parse_reasoner_response if is_reasoner else self._parse_response)(body, model_config.name)
                tokens = result.usage.total_tokens if result.usage else 0
                self.key_pool.release(kidx, tokens)
                self.record_success(started_at)
                return result
            except ProviderError as e:
                last_error = e
                self.mark_key_failed(kidx, e)
            except httpx.RequestError as e:
                last_error = ProviderError("deepseek", 0, f"request error: {e}")
                self.mark_key_failed(kidx, last_error)

        if last_error:
            self.record_provider_failure(last_error)
        raise last_error or AllKeysFailedError("deepseek")

    async def chat_completions_stream(
        self, request: ChatRequest, model_config: ModelConfig
    ):
        self.raise_if_unhealthy()
        started_at = time.monotonic()
        payload = self._build_request(request, model_config)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}

        last_error = None
        for _ in range(len(self.key_pool.keys)):
            acquired = self.key_pool.acquire()
            if not acquired:
                continue
            key, kidx = acquired

            pending_chunks = []
            output_started = False
            completed = False
            total_tokens = 0
            try:
                async with self.client.stream(
                    "POST", f"{self.config.base_url}/chat/completions",
                    json=payload,
                    headers=self._headers(key.key),
                ) as resp:
                    if resp.status_code != 200:
                        text = await resp.aread()
                        raise ProviderError("deepseek", resp.status_code, text.decode(errors="replace")[:300])

                    event = ""
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("event: "):
                            event = line[7:].strip()
                        elif line.startswith("data: "):
                            raw = line[6:]
                            if raw == "[DONE]":
                                completed = True
                                break
                            data = decode_sse_payload(raw, "deepseek", event)
                            event = ""
                            chunk = parse_stream_chunk(
                                self._parse_chunk,
                                data,
                                model_config.name,
                                "deepseek",
                            )
                            if chunk:
                                if chunk.usage:
                                    total_tokens = chunk.usage.total_tokens
                                has_content = any(
                                    choice.delta.content
                                    for choice in chunk.choices
                                )
                                if has_content:
                                    output_started = True
                                    for pending in pending_chunks:
                                        yield pending
                                    pending_chunks.clear()
                                    yield chunk
                                elif output_started:
                                    yield chunk
                                else:
                                    pending_chunks.append(chunk)

                    if not completed:
                        raise ProviderError("deepseek", 502, "stream ended before [DONE]")

                    for pending in pending_chunks:
                        yield pending

                    self.key_pool.release(kidx, total_tokens)
                    self.record_success(started_at)
                    return
            except ProviderError as e:
                self.mark_key_failed(kidx, e)
                if output_started:
                    self.record_provider_failure(e)
                    raise
                last_error = e
            except (httpx.HTTPError, UnicodeError) as e:
                last_error = normalize_stream_error(e, "deepseek")
                self.mark_key_failed(kidx, last_error)
                if output_started:
                    self.record_provider_failure(last_error)
                    raise last_error from e

        if last_error:
            self.record_provider_failure(last_error)
        raise last_error or AllKeysFailedError("deepseek")

    async def _try_send(self, payload: dict, api_key: str) -> httpx.Response:
        headers = self._headers(api_key)
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = await self.client.post(
                    f"{self.config.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                if resp.status_code in (429,) or resp.status_code >= 500:
                    if attempt < self.config.max_retries:
                        continue
                    raise ProviderError("deepseek", resp.status_code, resp.text[:200])
                resp.raise_for_status()
                return resp
            except httpx.TimeoutException:
                if attempt < self.config.max_retries:
                    continue
                raise ProviderError("deepseek", 0, "timeout")
            except httpx.HTTPStatusError as e:
                raise ProviderError("deepseek", e.response.status_code, e.response.text[:200])
            except httpx.RequestError as e:
                if attempt < self.config.max_retries:
                    continue
                raise ProviderError("deepseek", 0, f"request error: {e}")
        raise ProviderError("deepseek", 0, "max retries exceeded")
