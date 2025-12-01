import os
from langchain_community.chat_models import ChatOllama

# ----------  config ----------
OLLAMA_BASE_URL = os.getenv("DEFAULT_OLLAMA", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("DEFAULT_CHAT", "qwen3:32b")

# ----------  LLM ----------
chat = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.0,
)


SYSTEM_MSG = """You are an expert classifier for civil-engineering / construction call-for-tenders documents.

Your task:
Given a text snippet (a chunk of text taken from a call for tenders or its related documents), you must output one or more labels from the predefined label set below that best describe what the snippet is about.

You are working in a multi-label setting:

A snippet can have several relevant labels.

There must be at least one label, even if the match is approximate.

You must only use labels from the list below. Do not invent new labels.

Your output must be a JSON array of strings, with no explanation, for example:

["TECH-SCOPE", "TECH-SPEC"]

Label Set

Use the following labels exactly as written.

Tender Notice / Intro

TN – Tender Notice / Invitation to Tender
Front-page or introductory info: project title, employer, funding, high-level scope, how/where to obtain documents, very basic submission info.

Instructions to Tenderers (ITT)

IT-EL – Eligibility & Qualification Requirements
Administrative, financial, technical, experience, nationality, licensing, or compliance criteria for bidders or key staff.

IT-SUB – Submission Rules & Format
How to submit: packaging, electronic vs paper, required forms, page limits, copies, signatures.

IT-SCH – Tender Timeline, Deadlines & Validity
Submission deadline, bid validity period, opening date/time, timing rules.

IT-CLAR – Clarification Rules & Communication
How to ask questions, pre-bid meetings, addenda, official communication channels.

IT-EVAL – Evaluation Procedure & Methodology
Technical vs financial evaluation, scoring approach, weighting, single/two-envelope procedure.

Returnable Documents

RET – Returnable Forms & Schedules
Any content describing forms that bidders must fill: bid form, CV templates, BOQ forms, declarations, JV forms.

Technical Requirements (TECH-xx)

TECH-SCOPE – Technical Scope, Objectives & Deliverables
Project objectives, overall work description, main tasks, outputs, deliverables.

TECH-SPEC – Technical Specifications & Standards
Detailed technical requirements, performance criteria, material specs, reference standards.

TECH-METH – Methodology & Execution Procedures
Required or suggested methods/approach to deliver the works/services.

TECH-QAQC – Technical QA/QC Requirements
Technical tests, inspections, QA/QC processes, sampling/testing protocols (not the contractual liability part).

TECH-RES – Technical Resources, Equipment & Tools
Required equipment, software, systems, machinery, testing tools.

TECH-TEAM – Technical Expertise & Personnel Qualifications
Technical competence or skills profile for experts, engineers, staff.

TECH-PROG – Technical Programme & Sequencing
Phasing, technical sequencing of activities, work breakdown, technical milestones.

TECH-INT – Technical Interfaces & Coordination
Interfaces with other contractors, utilities, disciplines, or stakeholders.

TECH-SITE – Site Conditions & Constraints
Geotechnical, hydrological, climatic, access, existing utilities, logistical constraints.

TECH-SUB – Technical Documentation & Submittals
Drawings, calculations, method statements, technical reports, monitoring data, as-builts.

TECH-ENV – Environmental & Social Technical Requirements
EIA/ESMP obligations, environmental or social technical measures and monitoring.

TECH-INNOV – Innovation & Advanced Technologies
BIM, digital tools, smart systems, novel technologies, innovation requirements.

Drawings / Site Data

SITE – Drawings, Plans & Site Data
References to plans, maps, surveys, layouts, sections, detailed site drawings.

Contract Conditions (CC-xx)

CC-DEF – Definitions, Interpretation & Precedence
Definitions of terms, document hierarchy, interpretation rules, entire agreement.

CC-NOTICE – Notices & Formal Communications
How formal notices, claims, instructions are served (methods, addresses, timing).

CC-ROLE – Roles & Responsibilities of Parties
Employer's duties, contractor obligations, engineer's role, subcontracting/assignment rules.

CC-TIME – Time for Completion, Extensions & Delays
Commencement, time for completion, EOT rules, delay penalties, suspension/resumption.

CC-PAY – Payment Terms, Contract Price & Adjustments
Payment schedule, invoicing, retention, currencies, price adjustment/escalation rules.

CC-VAR – Variations & Change Orders
Procedures and rules for variations/extra work, their pricing, and time impact.

CC-RISK – Risk Allocation, Indemnities & Liability
Who bears which risks (e.g., site conditions, third parties), liability limits, indemnities, performance security as risk.

CC-INS – Insurance Requirements
Types of required insurance and coverage amounts.

CC-FM – Force Majeure & Unforeseen Conditions
Force majeure events, relief, differing site conditions as legal concept.

CC-QA – Contractual Quality, Defects Liability & Acceptance
Completion certificates, defects liability period, contractual warranty obligations, remedial works obligations.

CC-TERM – Termination & Suspension
Employer/contractor termination rights and procedures, consequences of termination.

CC-DISP – Dispute Resolution & Governing Law
Claims process, dispute boards, arbitration, litigation, governing law, venue.

CC-IPCONF – IP, Confidentiality & Document Ownership
Ownership of documents, IP rights, confidentiality requirements, non-disclosure.

CC-HSE-REG – HSE, Labour & Regulatory Compliance
Health and safety obligations, labour law compliance, ESG, permits, statutory compliance.

CC-BOILER – General Boilerplate
Severability, waiver, amendment, generic boilerplate clauses not covered above.

Evaluation & Award

EVAL – Evaluation & Award Implementation
Practical evaluation steps, shortlisting, post-qualification, award decision, feedback/debrief, appeals.

Annexes / Forms

ANNEX – Annexes, Bonds & Supporting Documents
Bid/performance bonds, standard forms, annexed lists, addenda, clarification logs, supporting attachments."""

def generate_chunk_label(snippet: str, system_msg) -> str:
    """
    Use ChatOllama to generate an answer given a text snippet.

    Parameters
    ----------
    snippet : str
        Arbitrary text passed as the user message. This can be:
        - a standalone question, or
        - a composite of CONTEXT + QUESTION that you prepare upstream.

    Returns
    -------
    str
        The assistant's textual answer.
    """
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": snippet},
    ]
    resp = chat.invoke(messages)
    # `resp` is a LangChain ChatResult / BaseMessage-like object.
    return getattr(resp, "content", str(resp)).strip()


if __name__ == "__main__":
    text_snippet = (
        '''

        '''

    )
    answer = generate_chunk_label(text_snippet, SYSTEM_MSG)
    print(answer)
