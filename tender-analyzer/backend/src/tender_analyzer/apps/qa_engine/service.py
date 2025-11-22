import re
from typing import Iterable, List, Dict

from tender_analyzer.domain.models import QuestionAnswer, Tender


class AnalysisService:
    HIGHLIGHT_QUESTIONS = [
        "What is the submission deadline?",
        "What is the required submission format (physical, electronic, or both)?",
        "Are there any mandatory administrative forms?",
        "Is there a mandatory bid bond?",
        "Are there any nationality or registration restrictions?",
        "Are there any mandatory partnerships (e.g., local firms) required?",
        "What are the tendering eligiblity criterias in terms of financial?",
        "What are the tendering eligiblity criterias in terms of technical?",
        "What are the tendering eligiblity criterias in terms of language skills?",
        "What is the type and nature of the project (e.g., new construction, rehabilitation, study, supervision)?",
        "Where is the project located (country, region, city) and are there any site-specific constraints?",
        "Is the project location logistically feasible?",
        "Are the project objectives and deliverables clearly defined?",
        "How will the proposals be evaluated (e.g., lowest price, quality-based, mixed)?",
        "Is there a weighting system for price vs. quality?",
        "Is there an indication of the budget range, and does it match our expectations?",
        "Is the budget realistic compared to the scope?",
        "Are the payment terms reasonable (e.g., advance payments, milestones)?",
        "Does the contract require the bidder to provide performance guarantees?",
        "Would we be required to pre-finance a significant portion of the work before payments start?",
        "Is the client's financial stability confirmed, or is there a risk of non-payment?",
        "What are the liability caps, if any?",
        "Are we required to take on unlimited liability?",
        "Are there unreasonable termination clauses?",
        "Are there indications that the project is underfunded or at risk of cancellation?",
        "Is this a recurring contract, or does the tender has the chance to secure multiple future contracts if perform well?",
    ]

    FULL_QUESTIONS = [
        "What are the major deliverables?",
        "What are the dependencies and risks?",
        "What is the estimated budget for this scope?",
        "Who are the key internal stakeholders or partners?",
        "What is the planned evaluation and award timeline?",
        "Are there any mandatory compliance or insurance requirements?",
        "How would we summarize the opportunity in one sentence?",
    ]

    QUESTION_KEYWORDS: Dict[str, List[str]] = {
        "What is the submission deadline?": ["deadline", "due date", "submit by"],
        "What is the required submission format (physical, electronic, or both)?": [
            "submission format",
            "electronic",
            "physical",
            "portal",
            "delivery",
        ],
        "Are there any mandatory administrative forms?": ["administrative form", "form", "documentation"],
        "Is there a mandatory bid bond?": ["bid bond", "bond", "guarantee"],
        "Are there any nationality or registration restrictions?": [
            "nationality",
            "registration",
            "citizen",
            "domicile",
        ],
        "Are there any mandatory partnerships (e.g., local firms) required?": [
            "partnership",
            "local partner",
            "joint venture",
        ],
        "What are the tendering eligiblity criterias in terms of financial?": [
            "financial",
            "turnover",
            "revenue",
            "capital",
        ],
        "What are the tendering eligiblity criterias in terms of technical?": [
            "technical",
            "experience",
            "capacity",
        ],
        "What are the tendering eligiblity criterias in terms of language skills?": [
            "language",
            "linguistic",
        ],
        "What is the type and nature of the project (e.g., new construction, rehabilitation, study, supervision)?": [
            "new construction",
            "rehabilitation",
            "study",
            "supervision",
            "project type",
        ],
        "Where is the project located (country, region, city) and are there any site-specific constraints?": [
            "located",
            "location",
            "country",
            "region",
            "site",
        ],
        "Is the project location logistically feasible?": ["logistics", "logistically", "accessible", "feasible"],
        "Are the project objectives and deliverables clearly defined?": [
            "objectives",
            "deliverables",
            "scope",
        ],
        "How will the proposals be evaluated (e.g., lowest price, quality-based, mixed)?": [
            "evaluated",
            "evaluation",
            "quality",
            "price",
            "weight",
        ],
        "Is there a weighting system for price vs. quality?": [
            "weighting system",
            "weighting",
            "price vs",
            "quality vs",
        ],
        "Is there an indication of the budget range, and does it match our expectations?": [
            "budget range",
            "budget",
            "estimate",
            "expectation",
        ],
        "Is the budget realistic compared to the scope?": [
            "realistic",
            "scope",
            "reasonable",
        ],
        "Are the payment terms reasonable (e.g., advance payments, milestones)?": [
            "payment terms",
            "advance",
            "milestone",
            "payment",
        ],
        "Does the contract require the bidder to provide performance guarantees?": [
            "performance guarantee",
            "guarantee",
            "performance bond",
        ],
        "Would we be required to pre-finance a significant portion of the work before payments start?": [
            "pre-finance",
            "pre finance",
            "advance funding",
        ],
        "Is the client's financial stability confirmed, or is there a risk of non-payment?": [
            "financial stability",
            "non-payment",
            "client risk",
        ],
        "What are the liability caps, if any?": ["liability cap", "liability"],
        "Are we required to take on unlimited liability?": ["unlimited liability"],
        "Are there unreasonable termination clauses?": ["termination", "terminate", "termination clause"],
        "Are there indications that the project is underfunded or at risk of cancellation?": [
            "underfunded",
            "cancellation",
            "risk",
        ],
        "Is this a recurring contract, or does the tender has the chance to secure multiple future contracts if perform well?": [
            "recurring",
            "future contract",
            "follow-on",
        ],
    }

    def run_highlight_qa(self, tender: Tender) -> List[QuestionAnswer]:
        return [
            QuestionAnswer(
                question=question,
                answer=self._build_answer(question, tender.analysis_corpus),
            )
            for question in self.HIGHLIGHT_QUESTIONS
        ]

    def run_full_qa(
        self, tender: Tender, cached_answers: Iterable[QuestionAnswer] = ()
    ) -> List[QuestionAnswer]:
        seen = {answer.question for answer in cached_answers}
        answers: List[QuestionAnswer] = list(cached_answers)
        for question in self.FULL_QUESTIONS:
            if question in seen:
                continue
            answers.append(
                QuestionAnswer(question=question, answer=self._build_answer(question, tender.analysis_corpus))
            )
        return answers

    def _build_answer(self, question: str, corpus: str) -> str:
        corpus_text = (corpus or "").strip()
        sentence = self._find_best_sentence(question, corpus_text)
        if sentence:
            return self._normalize_answer(sentence)
        return f"Information not found yet for: {question}"

    def _find_best_sentence(self, question: str, corpus: str) -> str:
        sentences = self._split_sentences(corpus)
        keywords = self.QUESTION_KEYWORDS.get(question, [])
        for sentence in sentences:
            lower = sentence.lower()
            if any(keyword in lower for keyword in keywords):
                return sentence
        return sentences[0] if sentences else ""

    @staticmethod
    def _split_sentences(corpus: str) -> List[str]:
        if not corpus:
            return []
        parts = re.split(r"(?<=[.!?])\s+", corpus)
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        trimmed = answer.strip()
        if len(trimmed) > 400:
            return trimmed[:400].rstrip() + "..."
        return trimmed
