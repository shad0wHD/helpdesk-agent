"""Unit tests for individual tools — no LLM calls, no DB required."""

import pytest

from app.tools.hr_lookup import lookup_employee


@pytest.mark.asyncio
async def test_lookup_known_employee():
    result = await lookup_employee.ainvoke({"name": "John Smith"})
    assert "EMP-1042" in result
    assert "engineering-vpn" in result


@pytest.mark.asyncio
async def test_lookup_partial_name():
    result = await lookup_employee.ainvoke({"name": "john"})
    assert "John Smith" in result


@pytest.mark.asyncio
async def test_lookup_unknown_employee():
    result = await lookup_employee.ainvoke({"name": "Nobody Here"})
    assert "not found" in result.lower()
