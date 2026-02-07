"""Entry point for python -m ipymcp."""

import argparse

from .server import create_server


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="MCP server for Jupyter kernel introspection",
    )
    parser.add_argument(
        "connection_file",
        nargs="?",
        default=None,
        help=(
            "Path or name of kernel connection file. "
            "If omitted, connects to the most recently active kernel."
        ),
    )
    args = parser.parse_args()
    mcp = create_server(args.connection_file)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    cli()
