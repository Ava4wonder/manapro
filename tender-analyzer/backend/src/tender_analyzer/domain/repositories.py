from typing import Dict, Iterable, List, Optional

from tender_analyzer.common.state.enums import TenderState
from tender_analyzer.domain.models import Evaluation, QuestionAnswer, Tender


class TenderRepository:
    def __init__(self) -> None:
        self._store: Dict[str, Tender] = {}

    def create(self, tender: Tender) -> Tender:
        self._store[tender.id] = tender
        return tender

    def get(self, tender_id: str) -> Optional[Tender]:
        return self._store.get(tender_id)

    def set_state(self, tender_id: str, state: TenderState) -> None:
        tender = self.get(tender_id)
        if not tender:
            return
        tender.state = state

    def update_highlight_answers(
        self, tender_id: str, answers: Iterable[QuestionAnswer]
    ) -> None:
        tender = self.get(tender_id)
        if not tender:
            return
        tender.highlight_answers = list(answers)

    def update_full_answers(self, tender_id: str, answers: Iterable[QuestionAnswer]) -> None:
        tender = self.get(tender_id)
        if not tender:
            return
        tender.full_answers = list(answers)

    def update_evaluation(self, tender_id: str, evaluation: Evaluation) -> None:
        tender = self.get(tender_id)
        if not tender:
            return
        tender.evaluation = evaluation

    def list(self) -> List[Tender]:
        return list(self._store.values())


tender_repo = TenderRepository()
