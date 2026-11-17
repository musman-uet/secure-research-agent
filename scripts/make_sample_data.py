"""Generate the sample document library for a fictional company, Lakeview Analytics.

Creates Word, PDF, Excel and Markdown files with realistic policy content,
a manifest with SharePoint-style URLs and per-document permissions, and a
user/group directory. Two travel policies deliberately disagree so the
conflicting-evidence check can be demonstrated.

    python scripts/make_sample_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import docx
from openpyxl import Workbook
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

DATA = Path(__file__).resolve().parent.parent / "data"
DOCS = DATA / "docs"
SP = "https://lakeview.sharepoint.com/sites"


def write_docx(path: Path, title: str, sections: list[tuple[str, list[str]]],
               table: list[list[str]] | None = None) -> None:
    d = docx.Document()
    d.add_heading(title, level=0)
    for heading, paragraphs in sections:
        d.add_heading(heading, level=1)
        for p in paragraphs:
            d.add_paragraph(p)
    if table:
        t = d.add_table(rows=0, cols=len(table[0]))
        for row in table:
            cells = t.add_row().cells
            for c, v in zip(cells, row):
                c.text = v
    d.save(str(path))


def write_pdf(path: Path, title: str, pages: list[list[tuple[str, str]]]) -> None:
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"])]
    for i, page in enumerate(pages):
        for heading, text in page:
            story += [Paragraph(heading, styles["Heading2"]), Paragraph(text, styles["BodyText"]),
                      Spacer(1, 8)]
        if i < len(pages) - 1:
            story.append(PageBreak())
    SimpleDocTemplate(str(path), pagesize=A4).build(story)


def write_xlsx(path: Path, sheets: dict[str, list[list]]) -> None:
    wb = Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        for r in rows:
            ws.append(r)
    wb.save(str(path))


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)

    write_docx(DOCS / "employee_handbook.docx", "Employee Handbook", [
        ("Working Hours", [
            "Standard working hours are 9:00 to 17:30, Monday to Friday, with a one hour lunch break.",
            "Core collaboration hours are 11:00 to 15:00, when all employees should be available for meetings.",
        ]),
        ("Annual Leave", [
            "Full-time employees receive 20 days of paid annual leave per calendar year.",
            "Up to 5 unused annual leave days can be carried over to the next year and must be used by 31 March.",
            "Annual leave requests must be submitted in the HR portal at least 2 weeks in advance.",
        ]),
        ("Sick Leave", [
            "Employees receive 10 days of paid sick leave per year.",
            "A medical certificate is required for sick leave longer than 2 consecutive days.",
        ]),
        ("Remote Work", [
            "Employees may work remotely up to 2 days per week with their manager's approval.",
            "Remote work from outside Pakistan requires written approval from HR and IT Security.",
        ]),
        ("Parental Leave", [
            "Birthing parents receive 16 weeks of paid parental leave. Non-birthing parents receive 4 weeks of paid parental leave.",
        ]),
    ])

    write_pdf(DOCS / "travel_policy_2023.pdf", "Travel Policy 2023", [
        [("Scope", "This policy applies to all business travel by Lakeview Analytics employees within Pakistan."),
         ("Approval", "All business travel must be approved in advance by the employee's line manager.")],
        [("Daily meal allowance", "The daily meal allowance for domestic business travel is PKR 3,000 per day."),
         ("Hotel accommodation", "The maximum hotel rate for domestic business travel is PKR 15,000 per night.")],
    ])

    write_pdf(DOCS / "travel_policy_2025.pdf", "Travel Policy 2025", [
        [("Scope", "This policy applies to all business travel by Lakeview Analytics employees within Pakistan and abroad."),
         ("Approval", "Domestic travel is approved by the line manager. International travel also requires approval from the department head.")],
        [("Daily meal allowance", "The daily meal allowance for domestic business travel is PKR 4,500 per day."),
         ("Hotel accommodation", "The maximum hotel rate for domestic business travel is PKR 22,000 per night."),
         ("Mileage", "Employees using a personal car for business travel are reimbursed PKR 65 per km.")],
    ])

    write_pdf(DOCS / "expense_claims_procedure.pdf", "Expense Claims Procedure", [
        [("Submitting claims", "Expense claims must be submitted in the finance portal within 30 days of the expense date."),
         ("Receipts", "An itemised receipt is required for every expense above PKR 2,000.")],
        [("Reimbursement timeline", "Approved expense claims are reimbursed with the next monthly payroll."),
         ("Late claims", "Claims submitted after 30 days require written approval from the Finance Manager.")],
    ])

    write_xlsx(DOCS / "salary_bands_2025.xlsx", {
        "Bands": [
            ["Grade", "Role level", "Minimum monthly salary (PKR)", "Maximum monthly salary (PKR)"],
            ["G1", "Associate", 150000, 230000],
            ["G2", "Engineer / Analyst", 230000, 380000],
            ["G3", "Senior Engineer / Senior Analyst", 380000, 600000],
            ["G4", "Lead", 600000, 850000],
            ["G5", "Manager", 850000, 1200000],
        ],
        "Notes": [
            ["Topic", "Detail"],
            ["Annual increment", "Annual salary increments are between 8 and 15 percent based on performance rating."],
            ["Review cycle", "Salary bands are reviewed every January by HR and Leadership."],
        ],
    })

    write_xlsx(DOCS / "q3_2025_budget.xlsx", {
        "Department budgets": [
            ["Department", "Q3 2025 budget (PKR million)", "Q3 2025 actual spend (PKR million)", "Owner"],
            ["Engineering", 48.0, 45.2, "CTO"],
            ["Sales and Marketing", 22.5, 24.1, "VP Sales"],
            ["Human Resources", 6.0, 5.4, "HR Director"],
            ["Finance", 4.5, 4.3, "CFO"],
            ["Cloud infrastructure", 18.0, 19.6, "CTO"],
        ],
    })

    write_docx(DOCS / "incident_response_runbook.docx", "Incident Response Runbook", [
        ("Severity Levels", [
            "Sev1 means a full customer-facing outage or a confirmed data breach.",
            "Sev2 means a major feature is degraded for many customers.",
            "Sev3 means a minor issue with a workaround available.",
        ]),
        ("Response Times", [
            "The on-call engineer must acknowledge a Sev1 incident within 15 minutes and a Sev2 incident within 30 minutes.",
            "Sev3 incidents are handled during business hours within 2 business days.",
        ]),
        ("Escalation", [
            "If a Sev1 incident is not mitigated within 1 hour, the on-call engineer escalates to the Head of Engineering.",
            "Suspected data breaches are reported to the Information Security Officer immediately.",
        ]),
        ("Postmortems", [
            "A blameless postmortem is written for every Sev1 and Sev2 incident within 5 business days.",
        ]),
    ], table=[["Severity", "Acknowledge within", "Update customers every"],
              ["Sev1", "15 minutes", "30 minutes"], ["Sev2", "30 minutes", "2 hours"]])

    write_pdf(DOCS / "board_minutes_2025_10.pdf", "Board Meeting Minutes - October 2025", [
        [("Attendance", "The board met on 14 October 2025 with all five directors present."),
         ("Project Falcon", "The board approved Project Falcon, the planned acquisition of a Karachi-based data labelling startup, with a maximum valuation of USD 4 million. The deal is confidential until signing.")],
        [("Hiring plan", "The board approved hiring 12 additional engineers in the first half of 2026."),
         ("Next meeting", "The next board meeting is scheduled for January 2026.")],
    ])

    (DOCS / "it_security_policy.md").write_text(
        "# Passwords\nPasswords must be at least 14 characters long. Password managers are recommended, "
        "and passwords must never be shared.\n\n"
        "# Multi-factor authentication\nMulti-factor authentication (MFA) is mandatory for email, "
        "VPN and all cloud applications.\n\n"
        "# Laptops\nAll company laptops must use full-disk encryption and lock automatically after 5 minutes "
        "of inactivity.\n\n"
        "# Reporting\nReport lost devices or suspected phishing emails to the IT Service Desk within 1 hour.\n",
        encoding="utf-8")

    write_docx(DOCS / "contractor_onboarding.docx", "Contractor Onboarding Guide", [
        ("Accounts", ["Contractors receive a company account that expires automatically at the end of the contract."]),
        ("Access", ["Contractors may only access the SharePoint sites listed in their statement of work."]),
        ("Timesheets", ["Contractors submit timesheets every Friday by 17:00 in the vendor portal."]),
    ])

    docs = [
        ("handbook", "Employee Handbook", "employee_handbook.docx", "HR/Shared Documents/Employee Handbook.docx", "2025-01-10", ["all-staff"]),
        ("travel-2023", "Travel Policy 2023", "travel_policy_2023.pdf", "Finance/Policies/Travel Policy 2023.pdf", "2023-02-01", ["all-staff"]),
        ("travel-2025", "Travel Policy 2025", "travel_policy_2025.pdf", "Finance/Policies/Travel Policy 2025.pdf", "2025-03-15", ["all-staff"]),
        ("expenses", "Expense Claims Procedure", "expense_claims_procedure.pdf", "Finance/Policies/Expense Claims Procedure.pdf", "2024-08-20", ["all-staff"]),
        ("salary-bands", "Salary Bands 2025", "salary_bands_2025.xlsx", "HR/Confidential/Salary Bands 2025.xlsx", "2025-01-15", ["hr-team", "leadership"]),
        ("budget-q3", "Q3 2025 Department Budget", "q3_2025_budget.xlsx", "Finance/Restricted/Q3 2025 Budget.xlsx", "2025-10-05", ["finance-team", "leadership"]),
        ("runbook", "Incident Response Runbook", "incident_response_runbook.docx", "Engineering/Runbooks/Incident Response Runbook.docx", "2025-06-01", ["engineering"]),
        ("board-2025-10", "Board Meeting Minutes - October 2025", "board_minutes_2025_10.pdf", "Leadership/Board/Board Minutes 2025-10.pdf", "2025-10-16", ["leadership"]),
        ("it-security", "IT Security Policy", "it_security_policy.md", "IT/Policies/IT Security Policy.md", "2025-04-02", ["all-staff", "contractors"]),
        ("contractors", "Contractor Onboarding Guide", "contractor_onboarding.docx", "HR/Shared Documents/Contractor Onboarding Guide.docx", "2024-11-12", ["contractors", "hr-team"]),
    ]
    manifest = {"documents": [
        {"doc_id": d, "title": t, "path": f"docs/{f}", "source_url": f"{SP}/{u.replace(' ', '%20')}",
         "last_modified": m, "allowed_principals": p}
        for d, t, f, u, m, p in docs
    ]}
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    directory = {"users": {
        "ayesha": ["all-staff", "hr-team"],
        "bilal": ["all-staff", "finance-team"],
        "sara": ["all-staff", "engineering"],
        "omar": ["all-staff", "leadership"],
        "zain": ["contractors"],
    }}
    (DATA / "directory.json").write_text(json.dumps(directory, indent=2), encoding="utf-8")
    print(f"Wrote {len(docs)} documents to {DOCS}")


if __name__ == "__main__":
    main()
