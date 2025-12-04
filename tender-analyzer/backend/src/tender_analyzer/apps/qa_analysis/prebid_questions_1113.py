from typing import Dict, List

QUESTIONS: Dict[str, Dict[str, List[str]]] = {
    "Administrative Questions": {
        "Eligibility & Compliance": [
            # "What are the tendering eligiblity criterias?",
            "What are the documents required to be submitted before submission deadline? List all of them and indicate how to prepare these docs.",
            # "What are the required key experts or engineer or personnels to meet the criteria? Is there any language skills requirements of key personnel? (e.g., minimum proficiency levels, required certifications, or project-specific communication capabilities)? Any nationality restrictions? Does Gruner has key experts meeting these criterias?",
            # "Are there any mandatory partnerships (e.g., local firms) required?",
            # "Are there specific financial stability criteria (e.g., turnover, bank guarantees)?",
            # "What types of technical certificates, accreditations, or licenses are required for the tenderer's company entity to meet the technical criteria? Does Gruner have the required technical certificates, accreditations, or licenses?",
            # "Does the client require previous experience with them or in a specific country/region? Does Gruner have such experience?",
        ],
        "Bid Submission Process": [
            # "Are there page or word limits for key sections of the bid?",
            # "How many copies of the bid need to be submitted?",
            # "Are there any pre-bid meetings, and is attendance mandatory?",
        ],
        "Procurement & Decision Process": [
            # "Is there a clear timeline for the evaluation and award process? And who are the key decision-makers in the evaluation process?",
            # "Is there an appeal process if we are not selected? And will we receive feedback?",
            # "Are there specific anti-corruption or ethical compliance requirements?",
        ],
    },

    "Technical Questions": {
        "Scope & Objectives": [
            # "What are the project objectives and deliverables? Write a brief summary (indicate: the location of the project and its specialness and potential difficulties, Does the project require innovative solutions? Does the scope require interdisciplinary expertise? Should the tenderer provide software, tools, or proprietary methodologies?).",
        ],
        "Resources & Teaming": [
            # "Does the project require full-time commitment from key experts?",
            # "Will our proposed team require security clearances or special certifications?",
            # "Is the tenderer required to provide local staff? Or establish a local presence (office, legal entity)?",
        ],
        # "Project Risks & Feasibility": [
        #     "Are there significant political or security risks in the project location?",
        #     "Are there indications that the project is underfunded or at risk of cancellation?",
        #     "Does the project require significant investment before payments start?",
        #     "Is the local regulatory environment stable and conducive to project execution?",
        #     "Does the project require extensive on-site presence, and is this feasible?",
        #     "Are there any known environmental or social risks associated with the project?",
        #     "Is there clarity on the client's decision-making authority during the project?",
        #     "Are there dependencies on third parties (e.g., government approvals, permits)?",
        #     "Are there requirements for specific materials, equipment, or software? Will that might be difficult to procure?",
        #     "Is the required reporting/documentation workload excessive or unclear?",
        # ],
    },

    "Contractual Questions": {
        "Financial & Payment Terms": [
            # "Summary the payment terms (indicate: payment currency, is there any advance payments required, milestones and acceptance criteria, Does the contract allow for price adjustments if costs increase, any penalties for late delivery, is there any required local banking partner)?",
            # "Is there a clear mechanism for handling disputes related to invoicing?",
            # "Are payments tied to subjective acceptance criteria by the client?",
            # "Does the contract require the bidder to provide performance guarantees?",
            # "Summary the tax obligations and tax terms (indicate: who is responsible for local taxes?)",
            # "Will we need a local banking partner to process payments?",
        ],
        # "Liabilities & Risks": [
        #     "What are the liability caps, if any?",
        #     "Are we required to take on unlimited liability?",
        #     "Are we liable for delays caused by external factors (e.g., permitting, approvals)?",
        #     "Does the contract require expensive insurance policies?",
        #     "Are there obligations beyond project completion (e.g., warranty periods)?",
        #     "Are there force majeure clauses, and are they reasonable?",
        #     "Does the contract impose severe restrictions on subcontracting?",
        #     "Are there unreasonable termination clauses?",
        #     "Are we required to indemnify the client against all risks?",
        # ],
        # "Legal & Compliance": [
        #     "Does the governing law of the contract pose risks (e.g., unfamiliar jurisdiction)?",
        #     "Are there strict intellectual property clauses that might limit future use of our work?",
        #     "Are there any confidentiality clauses that could impact other projects?",
        #     "Does the contract impose restrictions on working with competitors?",
        #     "Are there provisions for contract renegotiation if major circumstances change?",
        #     "Are dispute resolution mechanisms fair (e.g., arbitration, mediation)?",
        #     "Does the contract allow for periodic progress reviews and amendments?",
        #     "Are there mandatory ESG (Environmental, Social, Governance) compliance requirements?",
        #     "Are we required to comply with local labor laws that might be difficult to manage?",
        #     "Are there excessive bureaucratic compliance requirements that increase cost?",
        # ],
    },

    # "Strategic Considerations": {
    #     "General": [
    #         "Is this a recurring contract, and do we have the chance to secure multiple future contracts if we perform well?",
    #     ],
    # },

    # "Financial & Commercial Risk": {
    #     "General": [
    #         "Is the client's financial stability confirmed, or is there a risk of non-payment?",
    #         "Are there hidden costs not explicitly covered in the tender documents (e.g., travel, software licensing, compliance costs)?",
    #         "Does the tender allow for cost recovery in case of unexpected delays or scope changes?",
    #         "Would we be required to pre-finance a significant portion of the work before payments start?",
    #         "How does this project compare in profitability to other opportunities we could pursue instead?",
    #     ],
    # },


    # "Client & Relationship Factors": {
    #     "General": [
    #         "Are there signs that this project might be politically or bureaucratically complicated?",
    #         "Does the tender include clear escalation mechanisms in case of disputes?",
    #     ],
    # },

    # "Post-Bid Considerations": {
    #     "General": [
    #         "If we lose, will we receive feedback on how to improve future bids?"
    #     ],
    # },
}
