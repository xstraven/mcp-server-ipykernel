# ipymcp

`ipymcp` is an MCP server for Jupyter kernel introspection and execution.

It connects to a running Python kernel and exposes tools for:
- listing variables in the kernel namespace
- inspecting a specific variable
- describing pandas/polars DataFrames
- executing Python code (including image outputs like matplotlib plots)

## Requirements

- Python 3.12+
- A running Jupyter kernel (`ipykernel`)

## Install

```bash
pip install -e .
```

Or with `uv`:

```bash
uv sync
```

## Run

```bash
ipymcp [connection_file]
```

Equivalent:

```bash
python -m ipymcp [connection_file]
```

If `connection_file` is omitted, `ipymcp` will try to connect to the most recently active kernel (`kernel-*.json`).

## MCP Tools

- `list_variables()`
- `get_variable(name, max_length=5000)`
- `describe_dataframe(name, head_rows=5)`
- `execute_code(code, timeout=30)`

## Development

Run tests:

```bash
uv run pytest
```

## License

MIT. See `LICENSE`.
