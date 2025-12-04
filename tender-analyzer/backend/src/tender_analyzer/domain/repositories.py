import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from tender_analyzer.common.state.enums import TenderState
from tender_analyzer.domain.models import Evaluation, QuestionAnswer, Tender


_STORAGE_DIR = Path(__file__).parent.parent / "storage" / "tenders"
_STORAGE_DIR.mkdir(exist_ok=True, parents=True)


class TenderRepository:
    def __init__(self) -> None:
        self._store: Dict[str, Tender] = {}
        self._load_from_disk()

    def _tender_path(self, tender_id: str) -> Path:
        return _STORAGE_DIR / f"{tender_id}.json"

    def _save_to_disk(self, tender: Tender) -> None:
        try:
            path = self._tender_path(tender.id)
            payload = tender.dict()
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        except Exception:
            # Persistence issues must not break core request handling.
            return

    def _delete_from_disk(self, tender_id: str) -> None:
        try:
            path = self._tender_path(tender_id)
            if path.exists():
                path.unlink()
        except Exception:
            return

    def _load_from_disk(self) -> None:
        if not _STORAGE_DIR.exists():
            return
        for path in _STORAGE_DIR.glob("*.json"):
            try:
                with path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                tender = Tender(**payload)
                self._store[tender.id] = tender
            except Exception:
                # Skip corrupted or incompatible files.
                continue

    def create(self, tender: Tender) -> Tender:
        self._store[tender.id] = tender
        self._save_to_disk(tender)
        return tender

    def get(self, tender_id: str) -> Optional[Tender]:
        return self._store.get(tender_id)

    def delete(self, tender_id: str) -> None:
        """Remove a tender from the in-memory store and disk."""
        self._store.pop(tender_id, None)
        self._delete_from_disk(tender_id)

    def set_state(self, tender_id: str, state: TenderState) -> None:
        tender = self.get(tender_id)
        if not tender:
            return
        tender.state = state
        self._save_to_disk(tender)

    def update_highlight_answers(self, tender_id: str, jsonl_content: str) -> None:
        """
        Store the summary JSONL content for a tender.
        """
        tender = self.get(tender_id)
        if not tender:
            raise ValueError(f"Tender with ID {tender_id} not found.")

        tender.highlight_answers = jsonl_content
        self._save_to_disk(tender)

    def update_project_card_fields(self, tender_id: str, fields: Dict[str, str]) -> None:
        tender = self.get(tender_id)
        if not tender:
            return
        tender.project_card_fields = dict(fields)
        self._save_to_disk(tender)

    def update_full_answers(self, tender_id: str, answers: Iterable[QuestionAnswer]) -> None:
        tender = self.get(tender_id)
        if not tender:
            return
        tender.full_answers = list(answers)
        self._save_to_disk(tender)

    def update_evaluation(self, tender_id: str, evaluation: Evaluation) -> None:
        tender = self.get(tender_id)
        if not tender:
            return
        tender.evaluation = evaluation
        self._save_to_disk(tender)

    def list(self) -> List[Tender]:
        return list(self._store.values())


tender_repo = TenderRepository()

