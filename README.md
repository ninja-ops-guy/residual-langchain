# residual-langchain

LangChain callback adapter for RESIDUAL run attestation. Drop
`ResidualCallbackHandler` into any chain/model/tool `config={"callbacks": [...]}`
and get a spec-compliant attestation token per root run.

## Quickstart (<20 lines to first attestation)

```python
from langchain_core.runnables import RunnableLambda
from residual_langchain import ResidualCallbackHandler
from residual_sdk import SQLiteResidualBackend, verify_attestation

handler = ResidualCallbackHandler(
    backend=SQLiteResidualBackend("evidence.db"),
    spec_id="my-chain@1.0.0",
)
chain = RunnableLambda(lambda x: x.upper())
chain.invoke("hi", config={"callbacks": [handler]})

token = handler.attestation_for()
assert verify_attestation(token).ok
```

Async: use `AsyncResidualCallbackHandler` with `ainvoke`/`astream`.

## Lifecycle mapping

| LangChain | RESIDUAL |
|---|---|
| root `on_chain_start` (no parent) | `on_run_start` |
| `on_chat_model_start` / `on_llm_start` | `on_module_call("llm:<model>")` |
| nested `on_chain_start` | `on_module_call("chain:<name>")` |
| `on_tool_start` | `on_module_call("tool:<name>")` |
| root `on_chain_end` | `on_run_complete("success")` |
| `on_chain_error` / `on_llm_error` / `on_tool_error` | `on_run_complete("error")` |
| `on_llm_new_token` (streaming) | ignored — streaming is attested at module-call granularity, not per-token |

Edge cases handled: nested/multi-chain runs map to one residual run via
`parent_run_id` tracking; parallel root runs get independent run ids; async
runs use the async variant; the handler sets `raise_error = True`
(langchain-core ≥0.3/1.x) so backend/gate failures propagate instead of being
logged and swallowed.

## What the attestation proves / does not prove

See residual-sdk README. Additionally: per-token streaming content is NOT
attested (by design); a run whose LLM call fails mid-stream attests the
module call and the `error` outcome, not partial tokens.

## Troubleshooting

- Gate fire raises `GateFiredError` out of `invoke` — inspect
  `handler.attestation_for()["verdicts"]` (requires langchain-core honoring
  `raise_error`; on exotic old versions that swallow callback errors, check
  the backend ledger directly).
- `KeyError` from `attestation_for()`: no root run has completed yet.
- Callback exceptions appearing as warnings: upgrade langchain-core ≥0.3.

## Migration from direct core use

Replace direct ledger writes around your chain calls with the handler: delete
your manual run bookkeeping, pass the handler in `config["callbacks"]`, and
read the token from `attestation_for(lc_run_id)`. Gate configuration moves to
`GateSet(blocked_modules={...})` on the backend.

## Tests

```
pip install -e . pytest
python3 -m pytest tests/ -q   # 19 passed (framework integration + ADAPTERS-CONFORMANCE binding + mutant rejection)
```

Verified against langchain-core 1.6.3. CI YAML omitted (workflow-scope token
limitation); see RELEASES.md.
