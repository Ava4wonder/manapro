QUESTION_LABELS_REFINED = {
    "Administrative Questions": {
        "Eligibility & Compliance": {
            "What are the tendering eligiblity criterias in terms of administrative?":
                [["IT-EL"]],
            "What are the tendering eligiblity criterias in terms of financial?":
                [["IT-EL"]],
            "What are the tendering eligiblity criterias in terms of technical?":
                [["IT-EL", "IT-EVAL"]],
            "What are the required key experts or engineer or personnels to meet the criteria?":
                [["IT-EL", "RET"]],
            "What are the tendering eligiblity criterias in terms of language skills?":
                [["IT-EL", "IT-SCH"]],
            "Are there any nationality or registration restrictions?":
                [["IT-EL"]],
            "Are there any mandatory partnerships (e.g., local firms) required?":
                [["IT-EL", "RET"]],
            "Does the client require previous experience with them or in a specific country/region?":
                [["IT-EL"]],
            "Are there specific financial stability criteria (e.g., turnover, bank guarantees)?":
                [["IT-EL"]],
        },
        "Bid Submission Process": {
            "What is the submission deadline?":
                [["IT-SCH"]],
            "What is the required submission format (physical, electronic, or both)?":
                [["IT-SUB", "IT-SCH"]],
            "Are there mandatory administrative forms?":
                [["IT-SUB", "RET"]],
            "Is there a mandatory bid bond?":
                [["CC-PAY", "ANNEX"]],
            "Are there page or word limits for key sections of the bid?":
                [["IT-SUB"]],
            "How many copies of the bid need to be submitted?":
                [["IT-SUB", "IT-SCH"]],
            "Are there any pre-bid meetings, and is attendance mandatory?":
                [["IT-CLAR", "IT-SUB"]],
        },
        "Evaluation Criteria & Competition": {
            "How will the proposals be evaluated (e.g., lowest price, quality-based, mixed)?":
                [["IT-EVAL", "EVAL"]],
            "Is there a weighting system for price vs. quality?":
                [["IT-EVAL", "EVAL"]],
            "Is the competition limited to pre-qualified bidders?":
                [["IT-EL", "ANNEX"]],
            "Has the tender been reissued? If so, why was it not awarded previously?":
                [["TN", "ANNEX"]],
            "Is there an indication of the budget range, and does it match our expectations?":
                [["TN", "CC-PAY"]],
        },
        "Procurement & Decision Process": {
            "Is there a clear timeline for the evaluation and award process?":
                [["IT-EVAL", "EVAL"]],
            "Who are the key decision-makers in the evaluation process?":
                [["EVAL"]],
            "Is there an appeal process if we are not selected?":
                [["EVAL"]],
            "Will we receive feedback if unsuccessful?":
                [["EVAL"]],
            "Are there specific anti-corruption or ethical compliance requirements?":
                [["IT-EL", "CC-HSE-REG"]],
            "Will the contract be awarded based on technical merit alone or lowest price?":
                [["IT-EVAL", "EVAL"]],
            "Does the tendering process allow for clarifications and addenda before submission?":
                [["IT-CLAR", "ANNEX"]],
        },
    },

    "Technical Questions": {
        "Scope & Objectives": {
            "Are the project objectives and deliverables clearly defined?":
                [["TECH-SCOPE"]],
            "Does the project require innovative solutions?":
                [["TECH-SCOPE", "TECH-INNOV"]],
            "Does the project involve new methodologies, technologies, or regulations?":
                [["TECH-METH", "TECH-INNOV"]],
            "Does the scope require interdisciplinary expertise?":
                [["TECH-SCOPE", "TECH-TEAM", "IT-EL"]],
            "Is the project location logistically feasible?":
                [["TECH-SITE", "SITE"]],
            "Does the scope suggest a high risk of scope creep?":
                [["TECH-SCOPE", "CC-VAR"]],
            "Are there strict quality control or certification requirements?":
                [["TECH-QAQC", "CC-QA"]],
            "Are we expected to provide software, tools, or proprietary methodologies?":
                [["TECH-RES", "TECH-METH", "CC-IPCONF"]],
        },
        "Resources & Teaming": {
            "Do we need to subcontract or form a consortium, and if so, who are potential partners?":
                [["IT-EL", "RET", "CC-ROLE"]],
            "Are there nationality restrictions on experts?":
                [["IT-EL"]],
            "Does the project require full-time commitment from key experts?":
                [["TECH-TEAM", "CC-ROLE"]],
            "Are we required to provide local staff?":
                [["TECH-TEAM", "CC-ROLE"]],
            "Will our proposed team require security clearances or special certifications?":
                [["TECH-TEAM", "CC-HSE-REG"]],
            "Are we required to establish a local presence (office, legal entity)?":
                [["TECH-SITE", "CC-HSE-REG"]],
        },
        "Project Risks & Feasibility": {
            "Are there significant political or security risks in the project location?":
                [["TECH-SITE", "SITE"]],
            "Are there indications that the project is underfunded or at risk of cancellation?":
                [["TN", "CC-PAY"]],
            "Does the project require significant investment before payments start?":
                [["CC-PAY"]],
            "Is the local regulatory environment stable and conducive to project execution?":
                [["TECH-ENV", "CC-HSE-REG"]],
            "Does the project require extensive on-site presence, and is this feasible?":
                [["TECH-SITE", "SITE"]],
            "Are there any known environmental or social risks associated with the project?":
                [["TECH-ENV", "TECH-SITE", "SITE", "CC-HSE-REG"]],
            "Is there clarity on the client's decision-making authority during the project?":
                [["CC-ROLE"]],
            "Are there dependencies on third parties (e.g., government approvals, permits)?":
                [["TECH-INT", "TECH-ENV", "CC-RISK"]],
            "Are there requirements for specific materials, equipment, or software that might be difficult to procure?":
                [["TECH-RES", "TECH-SPEC"]],
            "Is the required reporting/documentation workload excessive or unclear?":
                [["TECH-SUB", "CC-IPCONF"]],
        },
    },

    "Contractual Questions": {
        "Financial & Payment Terms": {
            "Is the budget realistic compared to the scope?":
                [["TECH-SCOPE", "CC-PAY"]],
            "Are the payment terms reasonable (e.g., advance payments, milestones)?":
                [["CC-PAY"]],
            "Is the payment currency stable, or does it pose an exchange rate risk?":
                [["CC-PAY"]],
            "Does the contract allow for price adjustments if costs increase?":
                [["CC-PAY"]],
            "Are there penalties for late delivery, and are they proportionate?":
                [["CC-TIME"]],
            "Is there a clear mechanism for handling disputes related to invoicing?":
                [["CC-PAY", "CC-DISP"]],
            "Are payments tied to subjective acceptance criteria by the client?":
                [["CC-PAY", "CC-QA"]],
            "Does the contract require the bidder to provide performance guarantees?":
                [["CC-RISK", "ANNEX"]],
            "Are tax obligations clear, and who is responsible for local taxes?":
                [["CC-HSE-REG"]],
            "Will we need a local banking partner to process payments?":
                [["CC-PAY"]],
        },
        "Liabilities & Risks": {
            "What are the liability caps, if any?":
                [["CC-RISK"]],
            "Are we required to take on unlimited liability?":
                [["CC-RISK"]],
            "Are we liable for delays caused by external factors (e.g., permitting, approvals)?":
                [["CC-RISK", "CC-TIME"]],
            "Does the contract require expensive insurance policies?":
                [["CC-INS"]],
            "Are there obligations beyond project completion (e.g., warranty periods)?":
                [["CC-QA"]],
            "Are there force majeure clauses, and are they reasonable?":
                [["CC-FM"]],
            "Does the contract impose severe restrictions on subcontracting?":
                [["CC-ROLE"]],
            "Are there unreasonable termination clauses?":
                [["CC-TERM"]],
            "Are we required to indemnify the client against all risks?":
                [["CC-RISK"]],
        },
        "Legal & Compliance": {
            "Does the governing law of the contract pose risks (e.g., unfamiliar jurisdiction)?":
                [["CC-DISP"]],
            "Are there strict intellectual property clauses that might limit future use of our work?":
                [["CC-IPCONF"]],
            "Are there any confidentiality clauses that could impact other projects?":
                [["CC-IPCONF"]],
            "Does the contract impose restrictions on working with competitors?":
                [["CC-IPCONF"]],
            "Are there provisions for contract renegotiation if major circumstances change?":
                [["CC-BOILER"]],
            "Are dispute resolution mechanisms fair (e.g., arbitration, mediation)?":
                [["CC-DISP"]],
            "Does the contract allow for periodic progress reviews and amendments?":
                [["CC-BOILER"]],
            "Are there mandatory ESG (Environmental, Social, Governance) compliance requirements?":
                [["CC-HSE-REG"]],
            "Are we required to comply with local labor laws that might be difficult to manage?":
                [["CC-HSE-REG"]],
            "Are there excessive bureaucratic compliance requirements that increase cost?":
                [["CC-HSE-REG"]],
        },
    },

    "Strategic Considerations": {
        "General": {
            "Is this a recurring contract, and do we have the chance to secure multiple future contracts if we perform well?":
                [["TN", "CC-TERM"]],
        },
    },

    "Financial & Commercial Risk": {
        "General": {
            "Is the client's financial stability confirmed, or is there a risk of non-payment?":
                [["TN", "CC-PAY"]],
            "Are there hidden costs not explicitly covered in the tender documents (e.g., travel, software licensing, compliance costs)?":
                [["TECH-SCOPE", "CC-PAY"]],
            "Does the tender allow for cost recovery in case of unexpected delays or scope changes?":
                [["CC-PAY"]],
            "Would we be required to pre-finance a significant portion of the work before payments start?":
                [["CC-PAY"]],
            "How does this project compare in profitability to other opportunities we could pursue instead?":
                [["TECH-SCOPE", "CC-PAY"]],
        },
    },

    "Client & Relationship Factors": {
        "General": {
            "Are there signs that this project might be politically or bureaucratically complicated?":
                [["TN", "TECH-ENV", "CC-HSE-REG"]],
            "Does the tender include clear escalation mechanisms in case of disputes?":
                [["CC-DISP", "EVAL"]],
        },
    },

    "Post-Bid Considerations": {
        "General": {
            "If we lose, will we receive feedback on how to improve future bids?":
                [["EVAL"]],
        },
    },
}
