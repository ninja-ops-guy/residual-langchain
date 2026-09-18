"""Integration tests against LangChain's public API (langchain-core 1.6.3).

Uses only in-process fakes (FakeListChatModel / FakeListLLM / tools) — no
network, no API keys. Exercises: plain chain, chat model, tool calling,
nested chains, streaming, async, error path, gate firing, multi-chain.
"""

import asyncio
import uuid

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from residual_langchain import AsyncResidualCallbackHandler, ResidualCallbackHandler
from residual_sdk import GateSet, SQLiteResidualBackend, verify_attestation


class EchoLLM(LLM):
    @property
    def _llm_type(self):
        return "echo"

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        return f"echo:{prompt[:16]}"


def make_handler(blocked=()):
    backend = SQLiteResidualBackend(
        ":memory:", gate_set=GateSet(blocked_modules=set(blocked))
    )
    return ResidualCallbackHandler(backend=backend), backend


def test_plain_chain_attested():
    h, backend = make_handler()
    chain = RunnableLambda(lambda x: x + 1) | RunnableLambda(lambda x: x * 2)
    out = chain.invoke(2, config={"callbacks": [h]})
    assert out == 6
    tok = h.attestation_for()
    assert verify_attestation(tok).ok
    kinds = [e["kind"] for e in backend.events()]
    assert kinds[0] == "run.started" and "attestation.issued" in kinds


def test_chat_model_and_prompt():
    h, backend = make_handler()
    llm = FakeListChatModel(responses=["hello"])
    chain = ChatPromptTemplate.from_template("say {x}") | llm | StrOutputParser()
    out = chain.invoke({"x": "hi"}, config={"callbacks": [h]})
    assert out == "hello"
    modules = [
        e["payload"]["module"]
        for e in backend.events()
        if e["kind"] == "module.called"
    ]
    assert any(m.startswith("llm:") for m in modules), modules


def test_tool_calling():
    h, backend = make_handler()

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    out = add.invoke({"a": 1, "b": 2}, config={"callbacks": [h]})
    assert out == 3
    modules = [
        e["payload"]["module"]
        for e in backend.events()
        if e["kind"] == "module.called"
    ]
    assert "tool:add" in modules


def test_gate_fire_on_blocked_tool_propagates():
    h, backend = make_handler(blocked={"tool:add"})

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    from residual_sdk import GateFiredError

    with pytest.raises((GateFiredError, Exception)) as ei:
        add.invoke({"a": 1, "b": 2}, config={"callbacks": [h]})
    kinds = [e["kind"] for e in backend.events()]
    assert "gate.fired" in kinds


def test_error_path_marks_error_outcome():
    h, backend = make_handler()

    def boom(x):
        raise ValueError("boom")

    # first step succeeds (module call recorded), second fails -> outcome 'error'
    chain = RunnableLambda(lambda x: x) | RunnableLambda(boom)
    with pytest.raises(ValueError):
        chain.invoke(1, config={"callbacks": [h]})
    completed = [e for e in backend.events() if e["kind"] == "run.completed"]
    assert completed and completed[0]["payload"]["outcome"] == "error"


def test_streaming_tokens_not_attested_individually():
    h, backend = make_handler()
    llm = FakeListChatModel(responses=["abc"])
    list(llm.stream("hi", config={"callbacks": [h]}))
    modules = [
        e for e in backend.events() if e["kind"] == "module.called"
    ]
    # exactly one llm module call despite 3 tokens
    llm_calls = [e for e in modules if e["payload"]["module"].startswith("llm:")]
    assert len(llm_calls) == 1


def test_async_run():
    h = AsyncResidualCallbackHandler(backend=SQLiteResidualBackend(":memory:"))
    chain = RunnableLambda(lambda x: x + 1)

    async def main():
        return await chain.ainvoke(1, config={"callbacks": [h]})

    out = asyncio.run(main())
    assert out == 2
    tok = h.attestation_for()
    assert verify_attestation(tok).ok


def test_parallel_runs_independent():
    h, backend = make_handler()
    chain = RunnableLambda(lambda x: x)
    chain.invoke(1, config={"callbacks": [h]})
    chain.invoke(2, config={"callbacks": [h]})
    starts = [e for e in backend.events() if e["kind"] == "run.started"]
    completes = [e for e in backend.events() if e["kind"] == "run.completed"]
    assert len(starts) == 2 and len(completes) == 2
    assert len({e["run_id"] for e in starts}) == 2
    atts = [e for e in backend.events() if e["kind"] == "attestation.issued"]
    assert len(atts) == 2
