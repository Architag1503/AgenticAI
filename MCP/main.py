# arith_server.py
from __future__ import annotations
from fastmcp import FastMCP

mcp = FastMCP("arith")

def _as_number(x):
    # Accept int / float or numeric strings ; raise clean error otherwise
    if isinstance(x , (int , float)):
        return float(x)
    if isinstance(x , str):
        return float(x.strip())
    raise TypeError("Expected a number (int/float or numeric string)")

@mcp.tool()
async def add(a: float , b: float) -> float:
    """Return a + b"""
    return _as_number(a) + _as_number(b)

@mcp.tool()
async def sub(a: float , b: float) -> float:
    """Return a - b"""
    return _as_number(a) - _as_number(b)

@mcp.tool()
async def multiply(a: float , b: float) -> float:
    """Return a * b"""
    return _as_number(a) * _as_number(b)

@mcp.tool()
async def div(a: float , b: float)-> float:
    """Return a/b . Raises on division by zero"""
    a = _as_number(a)
    b = _as_number(b)

    if b == 0:
        raise ZeroDivisionError("Division by zero")
    return a/b


@mcp.tool()
async def power(a: float , b: float)-> float:
    """Return a ** b"""
    return _as_number(a)**_as_number(b)

@mcp.tool()
async def modulus(a: float, b: float) -> float:
    """Return a % b"""
    a = _as_number(a)
    b = _as_number(b)

    if b == 0:
        raise ZeroDivisionError("Modulo by zero")

    return a % b

def main():
    # Starts the MCP server over STDIO
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
