"""Integration tests for ipymcp.

Tests two layers:
- TestKernelConnection: exercises KernelConnection directly against a live kernel
- TestMCPServer: exercises the full MCP server via stdio subprocess using the MCP client SDK
"""

from __future__ import annotations

import sys

import pytest
import pytest_asyncio
from jupyter_client import KernelManager
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

from ipymcp.kernel import KernelConnection

pytestmark = pytest.mark.asyncio(loop_scope="session")


# ---------------------------------------------------------------------------
# KernelConnection tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kc_kernel_manager():
    km = KernelManager(kernel_name="python3")
    km.start_kernel()
    yield km
    km.shutdown_kernel(now=True)


@pytest_asyncio.fixture(scope="module")
async def kernel(kc_kernel_manager: KernelManager):
    kc = KernelConnection(kc_kernel_manager.connection_file)
    await kc.connect()
    yield kc
    kc.disconnect()


class TestKernelConnection:
    async def test_connect_and_alive(self, kernel: KernelConnection):
        assert kernel.is_alive

    async def test_execute_print(self, kernel: KernelConnection):
        result = await kernel.execute("print('hello')")
        assert result.status == "ok"
        assert "hello" in result.stdout

    async def test_execute_expression(self, kernel: KernelConnection):
        result = await kernel.execute("1 + 2")
        assert result.status == "ok"
        assert result.result == "3"

    async def test_execute_error(self, kernel: KernelConnection):
        result = await kernel.execute("raise ValueError('boom')")
        assert result.status == "error"
        assert result.error_name == "ValueError"
        assert "boom" in result.error_value

    async def test_execute_silent(self, kernel: KernelConnection):
        out = await kernel.execute_silent("print(42)")
        assert out == "42"

    async def test_execute_silent_raises_on_error(self, kernel: KernelConnection):
        with pytest.raises(RuntimeError, match="NameError"):
            await kernel.execute_silent("nonexistent_var_xyz")

    async def test_execute_assigns_variable(self, kernel: KernelConnection):
        await kernel.execute("__test_var = 99")
        result = await kernel.execute("__test_var")
        assert result.result == "99"
        await kernel.execute("del __test_var")

    async def test_execute_image_capture(self, kernel: KernelConnection):
        await kernel.execute("%matplotlib inline")
        result = await kernel.execute(
            "import matplotlib.pyplot as plt\n"
            "plt.figure(); plt.plot([1,2]); plt.show()"
        )
        assert result.status == "ok"
        assert len(result.images) >= 1
        assert result.images[0].startswith("iVBOR")

    async def test_execute_timeout(self, kernel: KernelConnection):
        result = await kernel.execute("import time; time.sleep(10)", timeout=1.0)
        assert result.status == "error"
        assert "timeout" in result.error_value.lower() or "Timeout" in result.error_name


# ---------------------------------------------------------------------------
# MCP server tests (full protocol via stdio subprocess)
#
# Each test spins up its own MCP server subprocess to avoid anyio/pytest
# teardown conflicts with long-lived fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mcp_connection_file():
    """A separate kernel used exclusively by MCP server tests."""
    km = KernelManager(kernel_name="python3")
    km.start_kernel()
    yield km.connection_file
    km.shutdown_kernel(now=True)


def _server_params(connection_file: str) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "ipymcp", connection_file],
    )


class TestMCPServer:
    async def test_list_tools(self, mcp_connection_file: str):
        async with stdio_client(_server_params(mcp_connection_file)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = {t.name for t in tools.tools}
                assert names == {
                    "list_variables", "get_variable",
                    "describe_dataframe", "execute_code",
                }

    async def test_execute_code_print(self, mcp_connection_file: str):
        async with stdio_client(_server_params(mcp_connection_file)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("execute_code", {"code": "print('hi from mcp')"})
                assert "hi from mcp" in _extract_text(result)

    async def test_execute_code_expression(self, mcp_connection_file: str):
        async with stdio_client(_server_params(mcp_connection_file)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("execute_code", {"code": "2 ** 10"})
                assert "1024" in _extract_text(result)

    async def test_execute_code_error(self, mcp_connection_file: str):
        async with stdio_client(_server_params(mcp_connection_file)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("execute_code", {"code": "1/0"})
                assert "ZeroDivisionError" in _extract_text(result)

    async def test_list_variables(self, mcp_connection_file: str):
        async with stdio_client(_server_params(mcp_connection_file)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await session.call_tool("execute_code", {"code": "test_x = 42; test_y = [1,2,3]"})
                result = await session.call_tool("list_variables", {})
                text = _extract_text(result)
                assert "test_x" in text
                assert "test_y" in text

    async def test_get_variable_simple(self, mcp_connection_file: str):
        async with stdio_client(_server_params(mcp_connection_file)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await session.call_tool("execute_code", {"code": "test_val = 'hello world'"})
                result = await session.call_tool("get_variable", {"name": "test_val"})
                text = _extract_text(result)
                assert "hello world" in text
                assert "str" in text

    async def test_get_variable_dataframe(self, mcp_connection_file: str):
        async with stdio_client(_server_params(mcp_connection_file)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await session.call_tool(
                    "execute_code",
                    {"code": "import pandas as pd; test_df = pd.DataFrame({'a': [1,2], 'b': [3,4]})"},
                )
                result = await session.call_tool("get_variable", {"name": "test_df"})
                text = _extract_text(result)
                assert "DataFrame" in text
                assert "(2, 2)" in text

    async def test_get_variable_not_found(self, mcp_connection_file: str):
        async with stdio_client(_server_params(mcp_connection_file)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("get_variable", {"name": "no_such_var_xyz"})
                assert "does not exist" in _extract_text(result)

    async def test_describe_dataframe(self, mcp_connection_file: str):
        async with stdio_client(_server_params(mcp_connection_file)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await session.call_tool(
                    "execute_code",
                    {
                        "code": (
                            "import pandas as pd\n"
                            "test_desc_df = pd.DataFrame({"
                            "'name': ['Alice','Bob','Charlie'],"
                            "'score': [88.5, 92.3, 76.1]})"
                        )
                    },
                )
                result = await session.call_tool("describe_dataframe", {"name": "test_desc_df"})
                text = _extract_text(result)
                assert "Shape" in text
                assert "(3, 2)" in text
                assert "Statistics" in text
                assert "Null Counts" in text
                assert "Memory" in text

    async def test_describe_dataframe_wrong_type(self, mcp_connection_file: str):
        async with stdio_client(_server_params(mcp_connection_file)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await session.call_tool("execute_code", {"code": "test_not_df = 42"})
                result = await session.call_tool("describe_dataframe", {"name": "test_not_df"})
                assert "not a DataFrame" in _extract_text(result)

    async def test_describe_dataframe_not_found(self, mcp_connection_file: str):
        async with stdio_client(_server_params(mcp_connection_file)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("describe_dataframe", {"name": "no_such_df_xyz"})
                assert "does not exist" in _extract_text(result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_text(result) -> str:
    """Pull all text content from a CallToolResult."""
    parts = []
    for block in result.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
    return "\n".join(parts)
