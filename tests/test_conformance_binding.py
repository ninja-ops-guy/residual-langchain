"""Runs the canonical adapter conformance suite against ResidualCallbackHandler.

The handle drives the handler's public callback methods directly (the same
methods LangChain's callback manager invokes), so the adapter's mapping code
is what is exercised.

Vendored suite: tests/vendor/adapter_conformance_suite.py
SHA-256: b8e14c9babc0eb4c7e07b04768afdda6e49c7a404e6e4ba1fc4f700f276b290e
"""

import sys
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "vendor"))

from adapter_conformance_suite import AdapterConformanceTests, AdapterHandle

from residual_langchain import ResidualCallbackHandler
from residual_sdk import (
    CoreUnreachableError,
    GateFiredError,
    GateSet,
    LedgerWriteError,
    SQLiteResidualBackend,
)


class _ChaosTransport:
    def __init__(self):
        self.reachable = True
        self.writable = True

    def ping(self):
        if not self.reachable:
            raise ConnectionRefusedError("core unreachable")
        return True

    def commit(self):
        if not self.writable:
            raise OSError("ledger commit failed")


class LangChainHandle(AdapterHandle):
    core_unreachable_exc = CoreUnreachableError
    ledger_write_exc = LedgerWriteError
    gate_fired_exc = GateFiredError

    def __init__(self):
        self.transport = _ChaosTransport()
        self.blocked: set[str] = set()
        self._new()
        self._lc_ids = {}  # residual run_id -> lc root uuid
        self._rid = {}     # suite run_id -> handler residual run_id

    def _new(self):
        self.backend = SQLiteResidualBackend(
            ":memory:",
            gate_set=GateSet(blocked_modules=self.blocked),
            transport=self.transport,
        )
        self.handler = ResidualCallbackHandler(backend=self.backend)

    def _root(self, run_id):
        if run_id not in self._lc_ids:
            self._lc_ids[run_id] = uuid.uuid4()
        return self._lc_ids[run_id]

    def start_run(self, run_id, spec_id):
        self.handler.spec_id = spec_id
        lc = self._root(run_id)
        self.handler.on_chain_start({"name": "root"}, {}, run_id=lc,
                                    parent_run_id=None)
        # handler assigns its own residual run id; record the mapping
        self._rid[run_id] = self.handler._roots.get(lc)

    def _r(self, run_id):
        return self._rid.get(run_id, run_id)

    def module_call(self, run_id, module):
        kind, _, name = module.partition(":")
        try:
            if kind == "tool":
                self.handler.on_tool_start(
                    {"name": name}, "input", run_id=uuid.uuid4(),
                    parent_run_id=self._root(run_id))
            else:
                self.handler.on_chat_model_start(
                    {"name": name}, [], run_id=uuid.uuid4(),
                    parent_run_id=self._root(run_id))
            return "PASS"
        except GateFiredError:
            raise

    def complete_run(self, run_id, outcome="success"):
        root = self._root(run_id)
        if outcome == "success":
            self.handler.on_chain_end({}, run_id=root, parent_run_id=None)
        else:
            self.handler.on_chain_error(RuntimeError(outcome), run_id=root,
                                        parent_run_id=None)

    def events(self, run_id):
        return self.backend.events(self._r(run_id))

    def get_attestation(self, run_id):
        return self.backend.get_attestation(self._r(run_id))

    def break_core(self):
        self.transport.reachable = False

    def heal_core(self):
        self.transport.reachable = True

    def break_ledger(self):
        self.transport.writable = False

    def heal_ledger(self):
        self.transport.writable = True

    def block_module(self, module):
        self.blocked.add(module)
        self._new()

    def try_mutate_ledger(self):
        self.backend._conn.execute("UPDATE events SET kind='x'")


class TestLangChainConformance(AdapterConformanceTests):
    def make_handle(self):
        return LangChainHandle()


class MutantHandler(ResidualCallbackHandler):
    """Deliberately non-compliant: swallows gate fires (returns PASS)."""

    def _module(self, run_id, parent_run_id, module, inputs):
        try:
            super()._module(run_id, parent_run_id, module, inputs)
        except GateFiredError:
            pass


class MutantHandle(LangChainHandle):
    def _new(self):
        self.backend = SQLiteResidualBackend(
            ":memory:",
            gate_set=GateSet(blocked_modules=self.blocked),
            transport=self.transport,
        )
        self.handler = MutantHandler(backend=self.backend)


class TestMutantRejected(unittest.TestCase):
    def test_gate_swallowing_mutant_detected(self):
        class Bound(AdapterConformanceTests):
            def make_handle(self):
                return MutantHandle()

        result = unittest.TestResult()
        unittest.TestLoader().loadTestsFromTestCase(Bound).run(result)
        failed = {t._testMethodName for t, _ in result.failures + result.errors}
        self.assertGreater(result.testsRun, 0)
        self.assertTrue(
            {"test_gate_firing_propagates", "test_verdict_not_coerced"} & failed,
            f"mutant not detected; failures={failed}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
