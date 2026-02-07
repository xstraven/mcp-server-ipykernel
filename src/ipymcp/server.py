"""MCP server exposing Jupyter kernel state to language models."""

from __future__ import annotations

import json
import textwrap
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ImageContent, TextContent

from .kernel import KernelConnection

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@dataclass
class AppContext:
    """Lifespan context holding the kernel connection."""

    kernel: KernelConnection


def create_server(connection_file: str | None = None) -> FastMCP:
    """Create and configure the FastMCP instance."""

    @asynccontextmanager
    async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
        kernel = KernelConnection(connection_file)
        await kernel.connect()
        try:
            yield AppContext(kernel=kernel)
        finally:
            kernel.disconnect()

    mcp = FastMCP("ipymcp", lifespan=app_lifespan)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _get_kernel(ctx: Context) -> KernelConnection:
        app_ctx: AppContext = ctx.request_context.lifespan_context
        return app_ctx.kernel

    # -----------------------------------------------------------------------
    # Tool: list_variables
    # -----------------------------------------------------------------------

    @mcp.tool()
    async def list_variables(ctx: Context) -> str:
        """List all user-defined variables in the kernel namespace with their types and shapes."""
        kernel = _get_kernel(ctx)

        code = textwrap.dedent("""\
            import json as __json__
            __ipython_builtins__ = {
                'In', 'Out', 'get_ipython', 'exit', 'quit', 'open',
            }
            __result__ = {}
            for __name__ in dir():
                if __name__.startswith('_') or __name__ in __ipython_builtins__:
                    continue
                try:
                    __obj__ = eval(__name__)
                except Exception:
                    continue
                import types as __types__
                if isinstance(__obj__, __types__.ModuleType):
                    continue
                del __types__
                if callable(__obj__) and not hasattr(__obj__, '__len__') and not hasattr(__obj__, 'shape'):
                    continue
                __type__ = type(__obj__).__module__ + '.' + type(__obj__).__name__
                __info__ = {"type": __type__}
                try:
                    if hasattr(__obj__, 'shape') and not callable(__obj__.shape):
                        __info__["shape"] = str(__obj__.shape)
                    if hasattr(__obj__, 'dtypes') and hasattr(__obj__, 'columns'):
                        __info__["columns"] = len(__obj__.columns)
                        __info__["rows"] = len(__obj__)
                    elif hasattr(__obj__, '__len__') and not isinstance(__obj__, (str, bytes)):
                        __info__["len"] = len(__obj__)
                except Exception:
                    pass
                __result__[__name__] = __info__
            print(__json__.dumps(__result__))
            del __json__, __result__, __ipython_builtins__
            try:
                del __name__, __obj__, __type__, __info__
            except NameError:
                pass
        """)

        raw = await kernel.execute_silent(code)
        if not raw:
            return "No user-defined variables in the kernel namespace."

        variables = json.loads(raw)
        if not variables:
            return "No user-defined variables in the kernel namespace."

        lines = [f"{'Variable':<25} {'Type':<30} {'Info'}"]
        lines.append("-" * 80)
        for name, info in variables.items():
            type_str = info["type"]
            # Shorten common module prefixes
            for prefix in ("builtins.", "pandas.core.frame.", "pandas.core.series.",
                           "polars.dataframe.frame.", "polars.series.series.",
                           "numpy."):
                type_str = type_str.replace(prefix, "")

            detail_parts = []
            if "shape" in info:
                detail_parts.append(f"shape={info['shape']}")
            if "columns" in info:
                detail_parts.append(f"{info['rows']} rows x {info['columns']} cols")
            if "len" in info:
                detail_parts.append(f"len={info['len']}")
            detail = ", ".join(detail_parts)
            lines.append(f"{name:<25} {type_str:<30} {detail}")

        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Tool: get_variable
    # -----------------------------------------------------------------------

    @mcp.tool()
    async def get_variable(name: str, max_length: int = 5000, ctx: Context = None) -> str:
        """Get the value of a variable from the kernel. Provides smart formatting for DataFrames, arrays, and other common types."""
        kernel = _get_kernel(ctx)

        code = textwrap.dedent(f"""\
            import json as __json__
            try:
                __obj__ = eval({name!r})
            except NameError:
                print(__json__.dumps({{"error": "Variable {name!s} does not exist"}}))
            else:
                __type_name__ = type(__obj__).__name__
                __module__ = type(__obj__).__module__
                __result__ = {{"type": __type_name__, "module": __module__}}

                # Pandas DataFrame
                if __type_name__ == 'DataFrame' and 'pandas' in __module__:
                    __result__["shape"] = str(__obj__.shape)
                    __result__["dtypes"] = __obj__.dtypes.to_string()
                    __result__["head"] = __obj__.head(20).to_string()

                # Polars DataFrame
                elif __type_name__ == 'DataFrame' and 'polars' in __module__:
                    __result__["shape"] = str(__obj__.shape)
                    __result__["schema"] = str(__obj__.schema)
                    __result__["head"] = str(__obj__.head(20))

                # Pandas Series
                elif __type_name__ == 'Series' and 'pandas' in __module__:
                    __result__["shape"] = str(__obj__.shape)
                    __result__["dtype"] = str(__obj__.dtype)
                    __result__["head"] = __obj__.head(20).to_string()

                # NumPy ndarray
                elif __type_name__ == 'ndarray':
                    __result__["shape"] = str(__obj__.shape)
                    __result__["dtype"] = str(__obj__.dtype)
                    import numpy as __np__
                    __flat__ = __obj__.flatten()
                    if len(__flat__) > 50:
                        __result__["sample"] = str(__flat__[:50]) + " ... (truncated)"
                    else:
                        __result__["value"] = str(__obj__)
                    del __np__, __flat__

                # Everything else
                else:
                    __repr__ = repr(__obj__)
                    if len(__repr__) > {max_length}:
                        __repr__ = __repr__[:{max_length}] + "\\n... (truncated)"
                    __result__["value"] = __repr__

                print(__json__.dumps(__result__))
                del __obj__, __type_name__, __module__, __result__
            del __json__
            try:
                del __repr__
            except NameError:
                pass
        """)

        raw = await kernel.execute_silent(code)
        data = json.loads(raw)

        if "error" in data:
            return data["error"]

        parts = [f"Variable: {name}", f"Type: {data.get('module', '')}.{data['type']}"]

        if "shape" in data:
            parts.append(f"Shape: {data['shape']}")
        if "dtype" in data:
            parts.append(f"Dtype: {data['dtype']}")
        if "dtypes" in data:
            parts.append(f"\nColumn dtypes:\n{data['dtypes']}")
        if "schema" in data:
            parts.append(f"\nSchema:\n{data['schema']}")
        if "head" in data:
            parts.append(f"\nData (head):\n{data['head']}")
        if "sample" in data:
            parts.append(f"\nValues: {data['sample']}")
        if "value" in data:
            parts.append(f"\nValue:\n{data['value']}")

        return "\n".join(parts)

    # -----------------------------------------------------------------------
    # Tool: describe_dataframe
    # -----------------------------------------------------------------------

    @mcp.tool()
    async def describe_dataframe(name: str, head_rows: int = 5, ctx: Context = None) -> str:
        """Get a comprehensive summary of a pandas or polars DataFrame: shape, dtypes, head, statistics, null counts, and memory usage."""
        kernel = _get_kernel(ctx)

        code = textwrap.dedent(f"""\
            import json as __json__
            try:
                __obj__ = eval({name!r})
            except NameError:
                print(__json__.dumps({{"error": "Variable {name!s} does not exist"}}))
            else:
                __type_name__ = type(__obj__).__name__
                __module__ = type(__obj__).__module__

                if __type_name__ != 'DataFrame':
                    print(__json__.dumps({{"error": "Variable {name!s} is a " + __type_name__ + ", not a DataFrame"}}))
                elif 'pandas' in __module__:
                    __result__ = {{
                        "lib": "pandas",
                        "shape": str(__obj__.shape),
                        "dtypes": __obj__.dtypes.to_string(),
                        "head": __obj__.head({head_rows}).to_string(),
                        "describe": __obj__.describe(include='all').to_string(),
                        "nulls": __obj__.isnull().sum().to_string(),
                        "memory": str(round(__obj__.memory_usage(deep=True).sum() / 1024, 2)) + " KB",
                    }}
                    print(__json__.dumps(__result__))
                    del __result__
                elif 'polars' in __module__:
                    __result__ = {{
                        "lib": "polars",
                        "shape": str(__obj__.shape),
                        "schema": str(__obj__.schema),
                        "head": str(__obj__.head({head_rows})),
                        "describe": str(__obj__.describe()),
                        "nulls": str(__obj__.null_count()),
                        "memory": str(round(__obj__.estimated_size('kb'), 2)) + " KB",
                    }}
                    print(__json__.dumps(__result__))
                    del __result__
                else:
                    print(__json__.dumps({{"error": "Variable {name!s} is a DataFrame from " + __module__ + " which is not supported (pandas and polars are supported)"}}))
                del __type_name__, __module__, __obj__
            del __json__
        """)

        raw = await kernel.execute_silent(code)
        data = json.loads(raw)

        if "error" in data:
            return data["error"]

        lib = data.get("lib", "unknown")
        sections = [
            f"DataFrame: {name} ({lib})",
            f"Shape: {data['shape']}",
            f"Memory: {data['memory']}",
        ]

        if "dtypes" in data:
            sections.append(f"\n--- Column Types ---\n{data['dtypes']}")
        if "schema" in data:
            sections.append(f"\n--- Schema ---\n{data['schema']}")

        sections.append(f"\n--- Head ({head_rows} rows) ---\n{data['head']}")
        sections.append(f"\n--- Statistics ---\n{data['describe']}")
        sections.append(f"\n--- Null Counts ---\n{data['nulls']}")

        return "\n".join(sections)

    # -----------------------------------------------------------------------
    # Tool: execute_code
    # -----------------------------------------------------------------------

    @mcp.tool()
    async def execute_code(code: str, timeout: int = 30, ctx: Context = None) -> list[TextContent | ImageContent]:
        """Execute Python code in the Jupyter kernel and return the output. Supports text output, expression results, errors, and matplotlib/seaborn plot images."""
        kernel = _get_kernel(ctx)
        result = await kernel.execute(code, timeout=float(timeout))

        contents: list[TextContent | ImageContent] = []

        text_parts = []
        if result.stdout:
            text_parts.append(result.stdout)
        if result.stderr:
            text_parts.append(f"[stderr]\n{result.stderr}")
        if result.result:
            text_parts.append(result.result)
        if result.status == "error":
            error_text = f"{result.error_name}: {result.error_value}"
            if result.traceback:
                # Strip ANSI escape codes from traceback
                import re
                clean_tb = [re.sub(r'\x1b\[[0-9;]*m', '', line) for line in result.traceback]
                error_text = "\n".join(clean_tb)
            text_parts.append(error_text)

        if text_parts:
            contents.append(TextContent(type="text", text="\n".join(text_parts)))

        for img_b64 in result.images:
            contents.append(ImageContent(
                type="image",
                data=img_b64,
                mimeType="image/png",
            ))

        if not contents:
            contents.append(TextContent(type="text", text="(no output)"))

        return contents

    return mcp
