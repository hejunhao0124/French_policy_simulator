"""
LLM 调用模块 - 使用 OpenAI 兼容接口（SiliconFlow）

优化：
1. 滑动窗口限流（RPM/TPM 双维度控制，避免 429 限速错误）
2. 响应缓存（相同 prompt 直接命中，避免重复调用）
3. 纯 asyncio 异步 I/O（替代 ThreadPoolExecutor，避免 GIL 开销）
4. 限速错误不重试（直接返回，不浪费配额）
5. 使用 threading.Event 替代 time.sleep 轮询
6. 区分错误类型（RateLimitError / APIError / 未知异常），记录 traceback
"""

import asyncio
import hashlib
import logging
import os
import sqlite3
import threading
import time
import traceback
from typing import List, Dict, Any

from openai import AsyncOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

CACHE_VERSION = 2  # 版本号，变更后旧缓存自动失效


def is_rate_limit_error(error) -> bool:
    """检测是否为限速错误"""
    if isinstance(error, RateLimitError):
        return True
    if hasattr(error, "status_code") and getattr(error, "status_code", None) == 429:
        return True
    rate_limit_keywords = [
        "rate limit", "too many requests", "quota exceeded",
        "tpm", "rpm", "overloaded",
        "RateLimit", "rate_limit", "insufficient_quota",
    ]
    error_msg = str(error).lower()
    return any(kw.lower() in error_msg for kw in rate_limit_keywords)


class RateLimiter:
    """滑动窗口限流器，同时控制 RPM 和 TPM"""

    def __init__(self, rpm: int, tpm: int):
        self.rpm = rpm
        self.tpm = tpm
        self._requests: list[tuple[float, int]] = []
        self._lock = asyncio.Lock()
        self._wait_interval = 0.5

    async def wait_for_slot(self, estimated_tokens: int = 700):
        """阻塞等待，直到有可用配额"""
        while True:
            async with self._lock:
                now = time.time()
                cutoff = now - 60.0
                self._requests = [(t, tok) for t, tok in self._requests if t > cutoff]
                current_rpm = len(self._requests)
                current_tpm = sum(tok for _, tok in self._requests)
                if current_rpm < self.rpm and (current_tpm + estimated_tokens) <= self.tpm:
                    return
            await asyncio.sleep(self._wait_interval)

    async def record(self, tokens_used: int):
        """记录一次请求的实际用量"""
        async with self._lock:
            self._requests.append((time.time(), tokens_used))


class LLMCache:
    """基于 SQLite 的 LLM 响应缓存"""

    def __init__(self, cache_dir=None):
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache"
            )
        os.makedirs(cache_dir, exist_ok=True)
        self.db_path = os.path.join(cache_dir, "llm_cache.db")
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache ("
                "  prompt_hash TEXT PRIMARY KEY,"
                "  model TEXT NOT NULL,"
                "  version INTEGER NOT NULL DEFAULT 0,"
                "  response TEXT NOT NULL"
                ")"
            )
            try:
                conn.execute(
                    "ALTER TABLE cache ADD COLUMN version INTEGER NOT NULL DEFAULT 0"
                )
            except Exception:
                pass
            conn.commit()

    def get(self, prompt: str, model: str) -> str | None:
        h = hashlib.md5(f"{model}:{prompt}".encode()).hexdigest()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT response FROM cache WHERE prompt_hash = ? AND model = ? AND version = ?",
                (h, model, CACHE_VERSION),
            ).fetchone()
        if row:
            resp = row[0]
            try:
                resp.encode('utf-8')
                return resp
            except (UnicodeEncodeError, UnicodeDecodeError):
                return None
        return None

    def put(self, prompt: str, model: str, response: str):
        h = hashlib.md5(f"{model}:{prompt}".encode()).hexdigest()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (prompt_hash, model, version, response) "
                "VALUES (?, ?, ?, ?)",
                (h, model, CACHE_VERSION, response),
            )
            conn.commit()

    def stats(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM cache WHERE version = ?", (CACHE_VERSION,)
            ).fetchone()[0]
            size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        return {"count": count, "size_bytes": size}


class LLMClient:
    """LLM 客户端"""

    def __init__(self, api_key, model="Qwen/Qwen2.5-7B-Instruct",
                 base_url="https://api.siliconflow.cn/v1",
                 rpm=3000, tpm=500000, temperature=0.7):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.temperature = temperature

        estimated_tokens_per_req = 700
        tpm_limited_rpm = tpm // estimated_tokens_per_req
        effective_rpm = min(rpm, tpm_limited_rpm)
        self.max_concurrent = max(20, min(300, effective_rpm * 3 // 60))

        self.cache = LLMCache()
        self.rate_limiter = RateLimiter(rpm, tpm)

        timeout_config = {"timeout": 30.0}
        self.client = OpenAI(api_key=api_key, base_url=base_url, **timeout_config)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url, **timeout_config)

    # ── 异步单次调用 ──

    async def _call_single_llm(self, persona_id: str, prompt: str,
                               max_retries: int = 3) -> Dict[str, Any]:
        """调用单个 LLM（带缓存 + 智能重试）

        错误处理策略:
        - 限速错误（429）: 直接返回失败，不重试，不浪费配额
        - 超时/连接错误: 指数退避重试
        - 其他错误: 记录日志后重试
        """
        cached = self.cache.get(prompt, self.model)
        if cached is not None:
            return {
                "persona_id": persona_id,
                "success": True,
                "response": cached,
                "cached": True,
                "tokens_used": 0,
            }

        last_error = None
        for attempt in range(max_retries):
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=self.temperature,
                )
                result = response.choices[0].message.content.strip()
                tokens_used = 0
                if hasattr(response, 'usage') and response.usage:
                    tokens_used = getattr(response.usage, 'total_tokens', 0)
                self.cache.put(prompt, self.model, result)
                return {
                    "persona_id": persona_id,
                    "success": True,
                    "response": result,
                    "cached": False,
                    "tokens_used": tokens_used,
                }
            except Exception as e:
                last_error = e
                # 限速错误直接返回，不重试
                if is_rate_limit_error(e):
                    return {
                        "persona_id": persona_id,
                        "success": False,
                        "error": str(e),
                        "error_type": "rate_limit",
                        "cached": False,
                        "tokens_used": 0,
                    }
                # 其他错误重试
                if attempt < max_retries - 1:
                    wait = min(2 ** attempt, 10)
                    logger.warning(
                        "LLM call failed for %s (attempt %d/%d, will retry in %ds): %s",
                        persona_id, attempt + 1, max_retries, wait, str(e)[:200],
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        "LLM call failed for %s (final attempt): %s",
                        persona_id, traceback.format_exc(),
                    )

        return {
            "persona_id": persona_id,
            "success": False,
            "error": str(last_error),
            "error_type": "unknown",
            "cached": False,
            "tokens_used": 0,
        }

    # ── 异步批量调用 ──

    async def _call_with_counter(self, prompts, counter):
        """异步批量调用，滑动窗口限流 + 并发控制

        使用原子操作更新进度计数器，移除锁竞争。
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def limited_call(idx, prompt_data):
            async with semaphore:
                await self.rate_limiter.wait_for_slot()
                result = await self._call_single_llm(
                    prompt_data["persona_id"], prompt_data["prompt"])
                if result.get("success") and result.get("tokens_used", 0) > 0:
                    await self.rate_limiter.record(result["tokens_used"])
                return idx, result

        coros = [limited_call(i, p) for i, p in enumerate(prompts)]

        for completed in asyncio.as_completed(coros):
            idx, result = await completed
            counter["results"][idx] = result
            counter["done"] += 1  # 原子操作，无需锁

    # ── 同步批量包装 ──

    def call_batch_sync(self, prompts, progress_callback=None):
        """同步批量调用，使用 threading.Event 替代 time.sleep 轮询"""
        results = [None] * len(prompts)
        error_holder = [None]
        done_event = threading.Event()

        counter = {
            "done": 0,
            "total": len(prompts),
            "results": results,
        }

        def _run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._call_with_counter(prompts, counter))
            except Exception as e:
                error_holder[0] = e
                logger.error("Batch call failed: %s\n%s", e, traceback.format_exc())
            finally:
                loop.close()
                done_event.set()

        thread = threading.Thread(target=_run_async, daemon=True)
        thread.start()

        # 主线程用 Event 等待，同时回调进度
        last_reported = -1
        while not done_event.wait(0.5):
            current_done = counter["done"]
            if current_done != last_reported and progress_callback:
                progress_callback(current_done, counter["total"])
                last_reported = current_done

        # 最终回调
        if progress_callback:
            progress_callback(counter["done"], counter["total"])

        thread.join()

        # 填充未完成或异常的结果
        for i, result in enumerate(results):
            if result is None:
                error_msg = str(error_holder[0]) if error_holder[0] else "请求未完成"
                results[i] = {
                    "persona_id": prompts[i]["persona_id"],
                    "success": False,
                    "error": error_msg,
                    "cached": False,
                }

        return results

    # ── 同步单次调用 ──

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def call_single_with_id(self, persona_id: str, prompt: str) -> Dict[str, Any]:
        """带 persona_id 的单次调用（带重试）"""
        cached = self.cache.get(prompt, self.model)
        if cached is not None:
            return {"persona_id": persona_id, "success": True, "response": cached}
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=self.temperature,
            )
            result = response.choices[0].message.content.strip()
            self.cache.put(prompt, self.model, result)
            return {"persona_id": persona_id, "success": True, "response": result}
        except Exception as e:
            error_type = "rate_limit" if is_rate_limit_error(e) else "api_error"
            return {"persona_id": persona_id, "success": False,
                    "error": str(e), "error_type": error_type}

    def call_single(self, prompt: str) -> Dict[str, Any]:
        """单次同步调用（简单场景）"""
        cached = self.cache.get(prompt, self.model)
        if cached is not None:
            return {"success": True, "response": cached}
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=self.temperature,
            )
            result = response.choices[0].message.content.strip()
            self.cache.put(prompt, self.model, result)
            return {"success": True, "response": result}
        except Exception as e:
            error_type = "rate_limit" if is_rate_limit_error(e) else "api_error"
            return {"success": False, "error": str(e), "error_type": error_type}
