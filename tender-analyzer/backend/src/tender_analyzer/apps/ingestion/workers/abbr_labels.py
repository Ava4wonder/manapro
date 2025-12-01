LABEL_DICTIONARY = {

    # ------------------------------------------------------------
    # 1. TENDER NOTICE / INTRODUCTORY SECTION
    # ------------------------------------------------------------
    "TN": {
        "name": "Tender Notice / Invitation to Tender",
        "description": (
            "Front-page information and administrative introduction to the tender: "
            "project title, procuring entity, funding source, publication date, how to obtain documents, "
            "basic scope summary, and high-level submission instructions."
        )
    },

    # ------------------------------------------------------------
    # 2. INSTRUCTIONS TO TENDERERS (ITT) — PROCESS & RULES
    # ------------------------------------------------------------
    "IT-EL": {
        "name": "Eligibility & Qualification Requirements",
        "description": (
            "All administrative, financial, technical, legal, and experience criteria that bidders must meet. "
            "Includes firm qualifications, key staff requirements, JV rules, nationality rules, licensing, "
            "financial capacity, certifications, and mandatory compliance."
        )
    },

    "IT-SUB": {
        "name": "Submission Rules & Format",
        "description": (
            "Instructions on how to submit the bid: packaging/envelopes, electronic vs physical submission, "
            "required forms, page limits, copy requirements, signature/sealing rules."
        )
    },

    "IT-SCH": {
        "name": "Tender Timeline, Deadlines & Validity",
        "description": (
            "Submission deadline, bid validity period, schedule for questions, opening date/time, "
            "and any timing rules applicable to the tender procedure."
        )
    },

    "IT-CLAR": {
        "name": "Clarification Rules & Communication Protocols",
        "description": (
            "How bidders may ask questions, how the procuring entity responds, addenda issuance, "
            "pre-bid meetings, formal communication channels, fairness rules for disseminating clarifications."
        )
    },

    "IT-EVAL": {
        "name": "Evaluation Procedure & Methodology",
        "description": (
            "The evaluation model: single-envelope/two-envelope process, threshold criteria, "
            "technical vs financial scoring, weighting system, compliance checks."
        )
    },

    # ------------------------------------------------------------
    # 3. RETURNABLE DOCUMENTS
    # ------------------------------------------------------------
    "RET": {
        "name": "Returnable Forms & Schedules",
        "description": (
            "Forms bidders must submit: Form of Tender, key staff CVs, price schedules, BOQ, qualifications forms, "
            "resources/equipment lists, compliance declarations, JV agreements, checklists."
        )
    },

    # ------------------------------------------------------------
    # 4. TECHNICAL REQUIREMENTS — FINE-GRAINED (TECH-xx)
    # ------------------------------------------------------------

    "TECH-SCOPE": {
        "name": "Technical Scope, Objectives & Deliverables",
        "description": (
            "Overall project objectives, work description, expected outputs, deliverables, and functional outcomes."
        )
    },

    "TECH-SPEC": {
        "name": "Technical Specifications & Standards",
        "description": (
            "Detailed requirements for materials, design parameters, standards, performance requirements, "
            "engineering tolerances, testing standards, and workmanship quality."
        )
    },

    "TECH-METH": {
        "name": "Methodology & Execution Procedures",
        "description": (
            "Required or recommended methodologies for performing the work, technical approaches, "
            "step-by-step execution processes, technical procedures, engineering workflows."
        )
    },

    "TECH-QAQC": {
        "name": "Technical Quality Assurance & Quality Control",
        "description": (
            "Engineering inspections, testing procedures, acceptance criteria, QA/QC plans, "
            "calibration requirements, approval processes for technical submissions."
        )
    },

    "TECH-RES": {
        "name": "Technical Resources, Equipment & Tools",
        "description": (
            "Minimum required equipment, tools, machinery, testing instruments, software platforms, "
            "ICT systems, BIM/GIS requirements, and any specialised technical tools."
        )
    },

    "TECH-TEAM": {
        "name": "Technical Expertise & Personnel Qualifications",
        "description": (
            "Technical competence requirements for engineers, specialists, team members; "
            "mandatory certifications, domain expertise, professional licenses, skill sets."
        )
    },

    "TECH-PROG": {
        "name": "Technical Programme, Sequencing & Milestones",
        "description": (
            "Technical dimension of scheduling: phasing, sequencing of activities, method statements, "
            "technical milestones, work breakdown structure, technical planning constraints."
        )
    },

    "TECH-INT": {
        "name": "Technical Interfaces & Coordination Requirements",
        "description": (
            "Interfaces with other contractors, utilities, authorities, external stakeholders, "
            "cross-discipline interactions, technical coordination obligations."
        )
    },

    "TECH-SITE": {
        "name": "Site Information, Conditions & Constraints",
        "description": (
            "Geotechnical, hydrological, topographical, climatic, logistical and access requirements; "
            "existing infrastructure data, site constraints, utility conflicts."
        )
    },

    "TECH-SUB": {
        "name": "Technical Documentation & Submittals",
        "description": (
            "Engineering drawings, calculations, shop drawings, method statements, inspection and test plans, "
            "monitoring data, technical reports, as-built documentation, and submittal schedules."
        )
    },

    "TECH-ENV": {
        "name": "Environmental & Social Technical Requirements",
        "description": (
            "EIA/ESMP technical obligations, environmental monitoring plans, social safeguards, "
            "pollution and impact mitigation, biodiversity considerations, compliance with environmental codes."
        )
    },

    "TECH-INNOV": {
        "name": "Innovation, Digital Tools & Advanced Technologies",
        "description": (
            "Requirements for innovative solutions, BIM, digital twins, LiDAR, remote monitoring, "
            "AI/ML tools, automation, smart systems, or other advanced technology."
        )
    },

    # ------------------------------------------------------------
    # 5. DRAWINGS / SITE INFORMATION
    # ------------------------------------------------------------
    "SITE": {
        "name": "Drawings, Plans & Existing Site Data",
        "description": (
            "Architectural, structural, civil, geotechnical drawings; site plans, maps, surveys; "
            "existing utilities, right-of-way maps, constraints relevant to technical planning."
        )
    },

    # ------------------------------------------------------------
    # 6. CONTRACT CONDITIONS — FINE-GRAINED (CC-xx)
    # ------------------------------------------------------------

    "CC-DEF": {
        "name": "Definitions, Interpretation & Contract Precedence",
        "description": (
            "Definitions of key terms, rules for interpreting documents, hierarchy of documents, "
            "entire-agreement clauses, conflict resolution between documents."
        )
    },

    "CC-NOTICE": {
        "name": "Notices & Formal Communications",
        "description": (
            "Procedures for issuing formal notices, instructions, claims, amendments; accepted delivery methods; "
            "address details; time requirements for notice validity."
        )
    },

    "CC-ROLE": {
        "name": "Roles & Responsibilities of Parties",
        "description": (
            "Employer's obligations, contractor responsibilities, engineer/supervisor authority, "
            "subcontracting/assignment rules, and administrative duties."
        )
    },

    "CC-TIME": {
        "name": "Time for Completion, Extensions & Delays",
        "description": (
            "Commencement, time for completion, milestones, EOT claims, delay penalties, "
            "suspension/resumption of works."
        )
    },

    "CC-PAY": {
        "name": "Payment Terms, Contract Price & Adjustments",
        "description": (
            "Payment schedule, milestones, retention, final payment; currency rules; price adjustment/escalation; "
            "cost-based compensation; invoicing rules."
        )
    },

    "CC-VAR": {
        "name": "Variation, Change Orders & Extra Work Management",
        "description": (
            "Procedures for issuing and approving variations; valuation methods; "
            "cost/time impacts; change order governance."
        )
    },

    "CC-RISK": {
        "name": "Risk Allocation, Indemnities & Liability",
        "description": (
            "Allocation of risks such as site conditions, third-party damage, indemnity obligations, "
            "contractor/employer liabilities, performance guarantees as risk tools."
        )
    },

    "CC-INS": {
        "name": "Insurance Requirements",
        "description": (
            "Mandatory insurance policies: works insurance, third-party liability, employer’s liability, "
            "professional indemnity, worker compensation, coverage limits."
        )
    },

    "CC-FM": {
        "name": "Force Majeure & Unforeseen Conditions",
        "description": (
            "Definitions of force majeure events, procedures, relief events, consequences for time and payment; "
            "differing site conditions claims, extraordinary events."
        )
    },

    "CC-QA": {
        "name": "Contractual Quality, Testing & Defects Liability",
        "description": (
            "Contractual QA obligations, inspection rights, rejection of works, defects liability period, "
            "warranty provisions, completion/hand-over certification."
        )
    },

    "CC-TERM": {
        "name": "Termination, Suspension & Post-Termination Rights",
        "description": (
            "Employer’s right to terminate/suspend, contractor's right to suspend/terminate, "
            "procedures, compensation on termination, settlement of outstanding obligations."
        )
    },

    "CC-DISP": {
        "name": "Dispute Resolution & Governing Law",
        "description": (
            "Claims procedures, dispute boards, arbitration, litigation, mediation, governing law, "
            "venue, escalation paths, time limits."
        )
    },

    "CC-IPCONF": {
        "name": "Intellectual Property, Confidentiality & Documentation Ownership",
        "description": (
            "Ownership of design data, drawings, documents, confidentiality obligations, "
            "use of proprietary information, restrictions on disclosure."
        )
    },

    "CC-HSE-REG": {
        "name": "HSE, Labour, Environmental & Regulatory Compliance",
        "description": (
            "Health, safety and environment obligations; compliance with labour laws, permits, "
            "zoning rules, statutory requirements, ESG requirements, environmental/social safeguards."
        )
    },

    "CC-BOILER": {
        "name": "General Boilerplate Clauses",
        "description": (
            "General legal clauses such as severability, amendment, waiver, assignment rules, "
            "entire agreement statements not already covered elsewhere."
        )
    },

    # ------------------------------------------------------------
    # 7. TENDER EVALUATION & AWARD
    # ------------------------------------------------------------
    "EVAL": {
        "name": "Evaluation & Award Procedures",
        "description": (
            "Implementation of evaluation rules, scoring, shortlisting, post-qualification, "
            "award decision logic, feedback and debriefing rules, appeals."
        )
    },

    # ------------------------------------------------------------
    # 8. ANNEXES
    # ------------------------------------------------------------
    "ANNEX": {
        "name": "Annexes, Bonds, Forms & Supporting Documents",
        "description": (
            "Bid bond templates, performance bond forms, legal declarations, addenda, "
            "clarification logs, prequalification forms, standardized submission templates."
        )
    },
}
