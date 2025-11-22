from typing import Optional

from tender_analyzer.apps.evaluation.service import EvaluationService
from tender_analyzer.apps.qa_engine.service import AnalysisService
from tender_analyzer.common.state.enums import TenderState
from tender_analyzer.common.state.state_machine import TenderStateMachine
from tender_analyzer.domain.repositories import tender_repo


class OrchestrationService:
    def __init__(
        self,
        repository=None,
        analysis_service: Optional[AnalysisService] = None,
        evaluation_service: Optional[EvaluationService] = None,
    ) -> None:
        self.repository = repository or tender_repo
        self.analysis = analysis_service or AnalysisService()
        self.evaluation = evaluation_service or EvaluationService()
        self.state_machine = TenderStateMachine()

    def _set_state(self, tender_id: str, target: TenderState) -> None:
        tender = self.repository.get(tender_id)
        if not tender:
            return
        try:
            self.state_machine.transition(tender.state, target)
        except ValueError:
            self.repository.set_state(tender_id, TenderState.FAILED)
            raise
        self.repository.set_state(tender_id, target)

    def start_analysis(self, tender_id: str) -> None:
        tender = self.repository.get(tender_id)
        if not tender:
            raise ValueError("tender not found")
        if tender.state == TenderState.EVAL_READY:
            return

        self._set_state(tender_id, TenderState.SUMMARY_RUNNING)
        highlight_answers = self.analysis.run_highlight_qa(tender)
        self.repository.update_highlight_answers(tender_id, highlight_answers)
        self._set_state(tender_id, TenderState.SUMMARY_READY)

        self._set_state(tender_id, TenderState.FULL_RUNNING)
        full_answers = self.analysis.run_full_qa(tender, highlight_answers)
        self.repository.update_full_answers(tender_id, full_answers)
        self._set_state(tender_id, TenderState.FULL_READY)

        self._set_state(tender_id, TenderState.EVAL_RUNNING)
        evaluation = self.evaluation.run_evaluation(tender)
        self.repository.update_evaluation(tender_id, evaluation)
        self._set_state(tender_id, TenderState.EVAL_READY)
