"""
Render the stakeholder markdown report to HTML and PDF.

Usage:
    python report/render_report.py
"""

from __future__ import annotations

import html
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_PATH = REPO_ROOT / "report" / "kenya_sentiment_report.md"
OUTPUT_DIR = REPO_ROOT / "outputs" / "reports"
HTML_PATH = OUTPUT_DIR / "Kenya_Sentiment_Analysis_Prototype_Report.html"
PDF_PATH = OUTPUT_DIR / "Kenya_Sentiment_Analysis_Prototype_Report.pdf"
FALLBACK_PDF_PATH = OUTPUT_DIR / "Kenya_Sentiment_Analysis_Prototype_Report_refreshed.pdf"

CHROME_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
]


def find_chrome() -> Path:
    for candidate in CHROME_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Google Chrome was not found in the default install paths.")


def format_inline(text: str) -> str:
    escaped = html.escape(text, quote=False)

    escaped = re.sub(
        r"`([^`]+)`",
        lambda match: f"<code>{html.escape(match.group(1))}</code>",
        escaped,
    )
    escaped = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda match: (
            f'<a href="{html.escape(match.group(2), quote=True)}">'
            f"{match.group(1)}</a>"
        ),
        escaped,
    )
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", escaped)
    return escaped


def split_table_row(row: str) -> list[str]:
    parts = [part.strip() for part in row.strip().strip("|").split("|")]
    return parts


def is_table_divider(row: str) -> bool:
    cells = split_table_row(row)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def render_table(table_lines: list[str]) -> str:
    rows = [split_table_row(line) for line in table_lines]
    if len(rows) < 2 or not is_table_divider(table_lines[1]):
        rendered_rows = "".join(
            "<tr>" + "".join(f"<td>{format_inline(cell)}</td>" for cell in row) + "</tr>"
            for row in rows
        )
        return f"<table><tbody>{rendered_rows}</tbody></table>"

    headers = rows[0]
    body_rows = rows[2:]
    thead = "<thead><tr>" + "".join(f"<th>{format_inline(cell)}</th>" for cell in headers) + "</tr></thead>"
    tbody = "<tbody>" + "".join(
        "<tr>" + "".join(f"<td>{format_inline(cell)}</td>" for cell in row) + "</tr>"
        for row in body_rows
    ) + "</tbody>"
    return f"<table>{thead}{tbody}</table>"


def markdown_to_html(markdown_text: str) -> str:
    output: list[str] = []
    paragraph_lines: list[str] = []
    list_type: str | None = None
    list_items: list[str] = []
    table_lines: list[str] = []
    quote_lines: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if paragraph_lines:
            text = " ".join(line.strip() for line in paragraph_lines)
            output.append(f"<p>{format_inline(text)}</p>")
            paragraph_lines = []

    def flush_list() -> None:
        nonlocal list_type, list_items
        if list_type and list_items:
            items = "".join(f"<li>{format_inline(item)}</li>" for item in list_items)
            output.append(f"<{list_type}>{items}</{list_type}>")
        list_type = None
        list_items = []

    def flush_table() -> None:
        nonlocal table_lines
        if table_lines:
            output.append(render_table(table_lines))
            table_lines = []

    def flush_quotes() -> None:
        nonlocal quote_lines
        if quote_lines:
            quote_html = "".join(f"<p>{format_inline(line)}</p>" for line in quote_lines)
            output.append(f"<blockquote>{quote_html}</blockquote>")
            quote_lines = []

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            flush_list()
            flush_table()
            flush_quotes()
            continue

        if stripped.startswith("<") and stripped.endswith(">"):
            flush_paragraph()
            flush_list()
            flush_table()
            flush_quotes()
            output.append(line)
            continue

        if re.match(r"^\|.*\|$", stripped):
            flush_paragraph()
            flush_list()
            flush_quotes()
            table_lines.append(stripped)
            continue
        flush_table()

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            flush_paragraph()
            flush_list()
            flush_quotes()
            level = len(heading_match.group(1))
            output.append(f"<h{level}>{format_inline(heading_match.group(2))}</h{level}>")
            continue

        if stripped in {"---", "***"}:
            flush_paragraph()
            flush_list()
            flush_quotes()
            output.append("<hr/>")
            continue

        unordered_match = re.match(r"^-\s+(.*)$", stripped)
        if unordered_match:
            flush_paragraph()
            flush_quotes()
            if list_type not in {None, "ul"}:
                flush_list()
            list_type = "ul"
            list_items.append(unordered_match.group(1))
            continue

        ordered_match = re.match(r"^\d+\.\s+(.*)$", stripped)
        if ordered_match:
            flush_paragraph()
            flush_quotes()
            if list_type not in {None, "ol"}:
                flush_list()
            list_type = "ol"
            list_items.append(ordered_match.group(1))
            continue

        quote_match = re.match(r"^>\s?(.*)$", stripped)
        if quote_match:
            flush_paragraph()
            flush_list()
            quote_lines.append(quote_match.group(1))
            continue

        flush_list()
        flush_quotes()
        paragraph_lines.append(stripped)

    flush_paragraph()
    flush_list()
    flush_table()
    flush_quotes()
    return "\n".join(output)


def build_html_document(body_html: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Kenya Sentiment Analysis Prototype Report</title>
  <style>
    @page {{
      size: A4;
      margin: 18mm 16mm;
    }}
    body {{
      font-family: Georgia, 'Times New Roman', serif;
      color: #1a1a1a;
      line-height: 1.55;
      font-size: 12pt;
      margin: 0 auto;
      max-width: 900px;
    }}
    h1, h2, h3, h4 {{
      color: #1f4e79;
      margin: 1.1em 0 0.45em 0;
      break-after: avoid-page;
    }}
    h1 {{ font-size: 24pt; }}
    h2 {{ font-size: 18pt; }}
    h3 {{ font-size: 14pt; }}
    p, li, blockquote, td, th {{
      font-size: 11.5pt;
    }}
    p {{
      margin: 0.45em 0 0.8em 0;
    }}
    ul, ol {{
      margin: 0.4em 0 0.9em 1.4em;
    }}
    li {{
      margin: 0.2em 0;
    }}
    blockquote {{
      margin: 1em 0;
      padding: 0.2em 1em;
      border-left: 4px solid #1f4e79;
      background: #f5f8fb;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 0.8em 0 1.1em 0;
      break-inside: avoid;
    }}
    th, td {{
      border: 1px solid #d8dee6;
      padding: 8px 10px;
      vertical-align: top;
      text-align: left;
    }}
    th {{
      background: #edf3f8;
      color: #183a59;
    }}
    code {{
      background: #f3f4f6;
      border-radius: 4px;
      padding: 0.1em 0.3em;
      font-family: Consolas, 'Courier New', monospace;
      font-size: 0.92em;
    }}
    hr {{
      border: none;
      border-top: 1px solid #cbd5e0;
      margin: 1.2em 0;
    }}
    a {{
      color: #1f4e79;
      text-decoration: none;
    }}
  </style>
</head>
<body>
{body_html}
</body>
</html>
"""


def render_pdf(html_path: Path, pdf_path: Path) -> None:
    chrome_path = find_chrome()
    chrome_profile_dir = OUTPUT_DIR / ".chrome-render-profile"
    powershell_command = (
        f"$profileDir = '{chrome_profile_dir}'; "
        "New-Item -ItemType Directory -Force -Path $profileDir | Out-Null; "
        f"& '{chrome_path}' "
        "--headless "
        "--disable-gpu "
        "--no-sandbox "
        "--allow-file-access-from-files "
        f"--user-data-dir='{chrome_profile_dir}' "
        f"--print-to-pdf='{pdf_path}' "
        f"'{html_path.resolve().as_uri()}'"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", powershell_command],
        check=True,
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    markdown_text = MARKDOWN_PATH.read_text(encoding="utf-8")
    body_html = markdown_to_html(markdown_text)
    html_document = build_html_document(body_html)
    HTML_PATH.write_text(html_document, encoding="utf-8")
    print(f"Wrote HTML report to {HTML_PATH}")

    final_pdf_path = PDF_PATH
    try:
        render_pdf(HTML_PATH, PDF_PATH)
    except subprocess.CalledProcessError:
        final_pdf_path = FALLBACK_PDF_PATH
        render_pdf(HTML_PATH, FALLBACK_PDF_PATH)
    print(f"Wrote PDF report to {final_pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
