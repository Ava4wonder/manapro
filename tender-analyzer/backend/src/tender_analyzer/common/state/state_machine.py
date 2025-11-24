from typing import Mapping

from tender_analyzer.common.state.enums import TenderState


class TenderStateMachine:
    transitions: Mapping[TenderState, tuple[TenderState, ...]] = {
        TenderState.INGESTING: (TenderState.INGESTED, TenderState.FAILED),
        TenderState.INGESTED: (TenderState.SUMMARY_RUNNING, TenderState.SUMMARY_READY, TenderState.FAILED),
        TenderState.SUMMARY_RUNNING: (TenderState.SUMMARY_READY, TenderState.FAILED),
        TenderState.SUMMARY_READY: (TenderState.FULL_RUNNING, TenderState.FAILED),
        TenderState.FULL_RUNNING: (TenderState.FULL_READY, TenderState.FAILED),
        TenderState.FULL_READY: (TenderState.EVAL_RUNNING, TenderState.FAILED),
        TenderState.EVAL_RUNNING: (TenderState.EVAL_READY, TenderState.FAILED),
        TenderState.EVAL_READY: (TenderState.FAILED,),
        TenderState.FAILED: (),
    }

    def can_transition(self, current: TenderState, target: TenderState) -> bool:
        allowed = self.transitions.get(current, ())
        return target in allowed

    def transition(self, current: TenderState, target: TenderState) -> TenderState:
        if not self.can_transition(current, target):
            raise ValueError(f"invalid transition from {current} to {target}")
        return target
