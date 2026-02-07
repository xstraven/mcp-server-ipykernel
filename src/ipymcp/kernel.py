"""Jupyter kernel connection management."""

from __future__ import annotations

import asyncio
import functools
from dataclasses import dataclass, field

from jupyter_client import BlockingKernelClient
from jupyter_client.connect import find_connection_file


@dataclass
class ExecutionResult:
    """Result of executing code in a kernel."""

    status: str = "ok"
    stdout: str = ""
    stderr: str = ""
    result: str | None = None
    error_name: str | None = None
    error_value: str | None = None
    traceback: list[str] | None = None
    images: list[str] = field(default_factory=list)  # base64-encoded PNGs


class KernelConnection:
    """Manages connection to a running Jupyter kernel."""

    def __init__(self, connection_file: str | None = None) -> None:
        self._connection_file_arg = connection_file
        self._connection_file: str | None = None
        self._client: BlockingKernelClient | None = None

    async def connect(self) -> None:
        """Find the connection file and connect to the kernel."""
        loop = asyncio.get_event_loop()
        self._connection_file = await loop.run_in_executor(
            None, self._resolve_connection_file
        )
        self._client = BlockingKernelClient(connection_file=self._connection_file)
        self._client.load_connection_file()
        self._client.start_channels()
        await loop.run_in_executor(
            None, functools.partial(self._client.wait_for_ready, timeout=30)
        )

    def _resolve_connection_file(self) -> str:
        """Resolve the connection file path."""
        if self._connection_file_arg:
            return find_connection_file(self._connection_file_arg)
        return find_connection_file("kernel-*.json")

    def disconnect(self) -> None:
        """Disconnect from the kernel."""
        if self._client is not None:
            self._client.stop_channels()
            self._client = None

    @property
    def is_alive(self) -> bool:
        """Check if the kernel is still alive via heartbeat."""
        if self._client is None:
            return False
        return self._client.is_alive()

    @property
    def connection_file(self) -> str | None:
        return self._connection_file

    async def execute(self, code: str, timeout: float = 30.0) -> ExecutionResult:
        """Execute code in the kernel and collect all output."""
        if self._client is None:
            return ExecutionResult(
                status="error",
                error_name="ConnectionError",
                error_value="Not connected to a kernel",
            )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, functools.partial(self._execute_sync, code, timeout)
        )

    def _execute_sync(self, code: str, timeout: float) -> ExecutionResult:
        """Synchronous execution with output collection."""
        assert self._client is not None

        result = ExecutionResult()
        outputs: list[dict] = []

        def output_hook(msg: dict) -> None:
            outputs.append(msg)

        try:
            reply = self._client.execute_interactive(
                code,
                silent=False,
                store_history=False,
                timeout=timeout,
                output_hook=output_hook,
            )
        except TimeoutError:
            return ExecutionResult(
                status="error",
                error_name="TimeoutError",
                error_value=f"Execution timed out after {timeout} seconds",
            )
        except Exception as e:
            return ExecutionResult(
                status="error",
                error_name=type(e).__name__,
                error_value=str(e),
            )

        result.status = reply["content"].get("status", "ok")

        for msg in outputs:
            msg_type = msg.get("msg_type") or msg.get("header", {}).get("msg_type")
            content = msg.get("content", {})

            if msg_type == "stream":
                if content.get("name") == "stderr":
                    result.stderr += content.get("text", "")
                else:
                    result.stdout += content.get("text", "")

            elif msg_type == "execute_result":
                data = content.get("data", {})
                result.result = data.get("text/plain")

            elif msg_type == "display_data":
                data = content.get("data", {})
                if "image/png" in data:
                    result.images.append(data["image/png"])
                elif "text/plain" in data:
                    result.stdout += data["text/plain"] + "\n"

            elif msg_type == "error":
                result.status = "error"
                result.error_name = content.get("ename")
                result.error_value = content.get("evalue")
                result.traceback = content.get("traceback")

        return result

    async def execute_silent(self, code: str, timeout: float = 10.0) -> str:
        """Execute code silently and return only stdout.

        Used for introspection queries where we only care about
        printed output (e.g., JSON dumps of variable info).
        """
        result = await self.execute(code, timeout=timeout)
        if result.status == "error":
            error_msg = f"{result.error_name}: {result.error_value}"
            if result.traceback:
                error_msg += "\n" + "\n".join(result.traceback)
            raise RuntimeError(error_msg)
        return result.stdout.strip()
