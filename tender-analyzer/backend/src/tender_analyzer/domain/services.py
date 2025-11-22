from typing import Iterable

from tender_analyzer.domain.models import QuestionAnswer, Tender
from tender_analyzer.domain.repositories import TenderRepository


class TenderService:
    def __init__(self, repository: TenderRepository) -> None:
        self.repository = repository

    def find(self, tender_id: str) -> Tender | None:
        return self.repository.get(tender_id)

    def add_highlight_answers(self, tender_id: str, answers: Iterable[QuestionAnswer]) -> None:
        self.repository.update_highlight_answers(tender_id, answers)

    def add_full_answers(self, tender_id: str, answers: Iterable[QuestionAnswer]) -> None:
        self.repository.update_full_answers(tender_id, answers)
