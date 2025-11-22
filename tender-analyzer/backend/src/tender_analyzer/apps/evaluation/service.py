from tender_analyzer.domain.models import Evaluation, Tender


class EvaluationService:
    def run_evaluation(self, tender: Tender) -> Evaluation:
        return Evaluation(
            summary="The documents were ingested and analyzed with placeholder QA answers.",
            risk_level="Medium",
            recommendation="Review the full report and confirm the opportunities manually.",
        )
