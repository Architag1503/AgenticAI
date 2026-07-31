# arith_server.py
from __future__ import annotations
from fastmcp import FastMCP

# WHY: We use FastMCP to quickly create a Model Context Protocol (MCP) server.
# This allows us to expose these Python functions as "Tools" that any MCP-compatible 
# LLM client (like LangChain or LangGraph) can discover and execute dynamically over a transport layer (like STDIO).
mcp = FastMCP("arith")

def _as_number(x):
    """
    Helper function to safely convert inputs to floats.
    WHY: LLMs sometimes pass arguments as strings instead of numbers. 
    This ensures our math operations don't crash due to TypeErrors.
    """
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        return float(x.strip())
    raise TypeError("Expected a number (int/float or numeric string)")

# WHY @mcp.tool(): This decorator tells the FastMCP server to expose this function 
# to the outside world as an accessible tool. The docstrings and type hints are 
# automatically converted into a JSON schema so the LLM knows EXACTLY how and when to use it.
@mcp.tool()
async def add(a: float, b: float) -> float:
    """Return a + b"""
    return _as_number(a) + _as_number(b)

@mcp.tool()
async def sub(a: float, b: float) -> float:
    """Return a - b"""
    return _as_number(a) - _as_number(b)

@mcp.tool()
async def multiply(a: float, b: float) -> float:
    """Return a * b"""
    return _as_number(a) * _as_number(b)

@mcp.tool()
async def div(a: float, b: float)-> float:
    """Return a/b . Raises on division by zero"""
    a = _as_number(a)
    b = _as_number(b)

    # WHY: We explicitly handle the ZeroDivisionError so the server doesn't 
    # crash ungracefully. The LLM will receive this error message and can correct itself.
    if b == 0:
        raise ZeroDivisionError("Division by zero")
    return a/b


@mcp.tool()
async def power(a: float, b: float)-> float:
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
    # Starts the MCP server over STDIO (Standard Input/Output).
    # WHY STDIO: This is the simplest transport layer. The client launches this script as a subprocess
    # and communicates with it by sending JSON-RPC messages through standard input and reading responses from standard output.
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
