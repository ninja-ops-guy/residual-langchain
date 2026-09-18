"""ResidualCallbackHandler — LangChain callback adapter for RESIDUAL.

Lifecycle mapping (LangChain -> RESIDUAL):
  * root on_chain_start (parent_run_id is None)      -> on_run_start
  * on_chat_model_start / on_llm_start               -> on_module_call("llm:<model>")
  * on_tool_start                                    -> on_module_call("tool:<name>")
  * root on_chain_end                                -> on_run_complete("success")
  * on_chain_error / on_llm_error / on_tool_error    -> on_run_complete("error") at root
  * on_llm_new_token                                 -> ignored (no per-token evidence;
    streaming is attested at the module-call level, not token level)

Edge cases:
  * nested chains/tools map to module calls on the root run (multi-chain);
  * parallel root runs (async/concurrent) get independent run_ids keyed by the
    LangChain run UUID;
  * gate firing raises GateFiredError out of the callback (fail-closed;
    this handler sets raise_error = True so LangChain propagates it);
  * handler exceptions never silently pass: if the backend write fails, the
    LedgerWriteError propagates.
"""

from __future__ import annotations

import uuid
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from residual_sdk import ResidualBackend, SQLiteResidualBackend
from residual_sdk.attestation import canonical_json, sha256_hex


class ResidualCallbackHandler(BaseCallbackHandler):
    """Drop-in LangChain callback handler that attests agent runs."""

    raise_error = True       # langchain-core >=0.3/1.x: propagate handler errors
    raise_exceptions = True  # legacy attribute name (older langchain versions)

    def __init__(self, backend: ResidualBackend | None = None,
                 spec_id: str = "langchain-agent@1.0.0"):
        self.backend = backend or SQLiteResidualBackend(":memory:")
        self.spec_id = spec_id
        self._roots: dict[UUID, str] = {}        # lc root run_id -> residual run_id
        self._residual_of: dict[UUID, str] = {}  # any lc run_id -> residual run_id
        self._attestations: dict[str, dict] = {}
        self._done: set[str] = set()             # residual run_ids with a terminal event

    # -- helpers -------------------------------------------------------------

    def _residual_run(self, run_id: UUID, parent_run_id: UUID | None) -> str:
        if parent_run_id is not None and parent_run_id in self._residual_of:
            rid = self._residual_of[parent_run_id]
            self._residual_of[run_id] = rid
            return rid
        rid = self._residual_of.get(run_id)
        if rid is None:
            rid = str(uuid.uuid4())
            self._roots[run_id] = rid
            self._residual_of[run_id] = rid
            self.backend.on_run_start(rid, self.spec_id,
                                      {"framework": "langchain"})
        return rid

    def _module(self, run_id: UUID, parent_run_id: UUID | None, module: str,
                inputs: Any) -> None:
        rid = self._residual_run(run_id, parent_run_id)
        self._module_nested(rid, module, inputs)

    def _is_root(self, run_id: UUID) -> bool:
        return run_id in self._roots

    def _finish_root(self, residual_run_id: str, outcome: str) -> None:
        if residual_run_id in self._done:
            from residual_sdk import LedgerWriteError

            raise LedgerWriteError(
                f"run {residual_run_id} already has a terminal event (exactly-one-terminal)"
            )
        self._done.add(residual_run_id)
        self.backend.on_run_complete(residual_run_id, outcome)
        try:
            self._attestations[residual_run_id] = self.backend.get_attestation(
                residual_run_id
            )
        except Exception:
            pass

    # -- chain lifecycle ------------------------------------------------------

    def on_chain_start(self, serialized: dict[str, Any] | None,
                       inputs: dict[str, Any], *, run_id: UUID,
                       parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        name = (serialized or {}).get("name") or "chain"
        if parent_run_id is None or parent_run_id not in self._residual_of:
            self._residual_run(run_id, parent_run_id)
        else:
            rid = self._residual_run(run_id, parent_run_id)
            self._module_nested(rid, f"chain:{name}", inputs)

    def _module_nested(self, residual_run_id: str, module: str, inputs: Any) -> None:
        verdict = self.backend.on_module_call(
            residual_run_id, module, str(uuid.uuid4()),
            sha256_hex(canonical_json(_jsonable(inputs))),
        )
        if verdict != "PASS":
            from residual_sdk import GateFiredError

            self._finish_root(residual_run_id, "aborted")
            raise GateFiredError("G0-BLOCKED-MODULE", verdict, module)

    def on_chain_end(self, outputs: Any, *, run_id: UUID,
                     parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        rid = self._roots.pop(run_id, None)
        if rid is None and run_id in self._residual_of:
            candidate = self._residual_of[run_id]
            if candidate in self._done:
                # repeated terminal on an already-completed root: fail closed
                self._finish_root(candidate, "success")
            return
        if rid is not None:
            self._finish_root(rid, "success")

    def on_chain_error(self, error: BaseException, *, run_id: UUID,
                       parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        rid = self._roots.pop(run_id, None)
        if rid is None and run_id in self._residual_of:
            candidate = self._residual_of[run_id]
            if candidate in self._done:
                self._finish_root(candidate, "error")
            return
        if rid is not None:
            self._finish_root(rid, "error")

    # -- LLM lifecycle ---------------------------------------------------------

    def on_chat_model_start(self, serialized: dict[str, Any] | None,
                            messages: list, *, run_id: UUID,
                            parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        model = _model_name(serialized, kwargs)
        self._module(run_id, parent_run_id, f"llm:{model}",
                     {"n_messages": len(messages)})

    def on_llm_start(self, serialized: dict[str, Any] | None,
                     prompts: list[str], *, run_id: UUID,
                     parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        model = _model_name(serialized, kwargs)
        self._module(run_id, parent_run_id, f"llm:{model}",
                     {"n_prompts": len(prompts)})

    def on_llm_error(self, error: BaseException, *, run_id: UUID,
                     parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        rid = self._residual_of.get(run_id)
        if rid is not None:
            self._finish_root(rid, "error")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        # Streaming: deliberately no per-token evidence. The module call is
        # attested at on_chat_model_start/on_llm_start granularity.
        return None

    # -- tool lifecycle ----------------------------------------------------------

    def on_tool_start(self, serialized: dict[str, Any] | None,
                      input_str: str, *, run_id: UUID,
                      parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        name = (serialized or {}).get("name") or "unknown"
        self._module(run_id, parent_run_id, f"tool:{name}", {"input": input_str})

    def on_tool_error(self, error: BaseException, *, run_id: UUID,
                      parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        rid = self._residual_of.get(run_id)
        if rid is not None:
            self._finish_root(rid, "error")

    # -- results ----------------------------------------------------------------

    def attestation_for(self, lc_run_id: UUID | None = None) -> dict:
        """Return the attestation for a finished root run.

        With no argument, returns the most recent attestation.
        """
        if lc_run_id is not None:
            return self._attestations[self._residual_of[lc_run_id]]
        if not self._attestations:
            raise KeyError("no attestation yet")
        return list(self._attestations.values())[-1]


def _model_name(serialized: dict | None, kwargs: dict) -> str:
    if serialized and serialized.get("name"):
        return serialized["name"]
    inv = kwargs.get("invocation_params") or {}
    return inv.get("model_name") or inv.get("model") or "unknown"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return repr(obj)


# Async handler: LangChain dispatches async runs through AsyncCallbackHandler.
try:  # pragma: no cover - import guard
    from langchain_core.callbacks import AsyncCallbackHandler

    class AsyncResidualCallbackHandler(AsyncCallbackHandler):
        """Async variant: identical mapping, async methods."""

        raise_error = True
        raise_exceptions = True

        def __init__(self, backend: ResidualBackend | None = None,
                     spec_id: str = "langchain-agent@1.0.0"):
            self._sync = ResidualCallbackHandler(backend=backend, spec_id=spec_id)

        @property
        def backend(self):
            return self._sync.backend

        async def on_chain_start(self, serialized, inputs, *, run_id,
                                 parent_run_id=None, **kwargs):
            self._sync.on_chain_start(serialized, inputs, run_id=run_id,
                                      parent_run_id=parent_run_id, **kwargs)

        async def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
            self._sync.on_chain_end(outputs, run_id=run_id,
                                    parent_run_id=parent_run_id, **kwargs)

        async def on_chain_error(self, error, *, run_id, parent_run_id=None, **kwargs):
            self._sync.on_chain_error(error, run_id=run_id,
                                      parent_run_id=parent_run_id, **kwargs)

        async def on_chat_model_start(self, serialized, messages, *, run_id,
                                      parent_run_id=None, **kwargs):
            self._sync.on_chat_model_start(serialized, messages, run_id=run_id,
                                           parent_run_id=parent_run_id, **kwargs)

        async def on_llm_start(self, serialized, prompts, *, run_id,
                               parent_run_id=None, **kwargs):
            self._sync.on_llm_start(serialized, prompts, run_id=run_id,
                                    parent_run_id=parent_run_id, **kwargs)

        async def on_llm_new_token(self, token, **kwargs):
            return None

        async def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
            self._sync.on_llm_error(error, run_id=run_id,
                                    parent_run_id=parent_run_id, **kwargs)

        async def on_tool_start(self, serialized, input_str, *, run_id,
                                parent_run_id=None, **kwargs):
            self._sync.on_tool_start(serialized, input_str, run_id=run_id,
                                     parent_run_id=parent_run_id, **kwargs)

        async def on_tool_error(self, error, *, run_id, parent_run_id=None, **kwargs):
            self._sync.on_tool_error(error, run_id=run_id,
                                     parent_run_id=parent_run_id, **kwargs)

        def attestation_for(self, lc_run_id=None):
            return self._sync.attestation_for(lc_run_id)

except ImportError:  # pragma: no cover
    AsyncResidualCallbackHandler = None  # type: ignore[assignment]
