"""Render a Markdown study report as a self-contained, local HTML document."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

from markdown_it import MarkdownIt


CSS = """
:root { color-scheme: light; --ink:#182536; --muted:#526174; --line:#d8e1e8; }
* { box-sizing:border-box; }
body { margin:0; background:#f3f6f8; color:var(--ink);
  font:17px/1.65 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
main { max-width:1040px; margin:32px auto; background:white; padding:52px 64px;
  border:1px solid var(--line); border-radius:10px; }
h1 { font-size:2.15rem; line-height:1.2; letter-spacing:-.035em; margin:0 0 20px; }
h2 { font-size:1.5rem; line-height:1.3; margin:42px 0 16px; padding-top:12px;
  border-top:1px solid var(--line); }
h3 { font-size:1.15rem; margin:28px 0 10px; }
p,ul,ol { margin:14px 0; }
li { margin:8px 0; }
a { color:#1555a1; text-underline-offset:3px; overflow-wrap:anywhere; }
code { font-size:.88em; background:#edf2f6; padding:2px 4px; border-radius:3px;
  overflow-wrap:anywhere; }
blockquote { margin:24px 0; padding:8px 22px; background:#eef5fa;
  border-left:4px solid #3378ae; }
.table-scroll { overflow-x:auto; margin:20px 0; }
table { border-collapse:collapse; width:100%; font-size:.9rem; line-height:1.45; }
th,td { padding:11px 12px; border-bottom:1px solid var(--line); vertical-align:top; }
th { background:#eaf0f5; text-align:left; font-weight:650; }
tr:nth-child(even) td { background:#f8fafc; }
footer { color:var(--muted); border-top:1px solid var(--line); margin-top:36px;
  padding-top:18px; font-size:.85rem; }
@media(max-width:720px) {
  body { font-size:16px; } main { margin:0; padding:28px 20px; border:0;
  border-radius:0; } h1 { font-size:1.8rem; } th,td { min-width:100px; }
}
@media print {
  body { background:white; font-size:11pt; } main { margin:0; border:0;
  padding:0; max-width:none; } h2,h3 { break-after:avoid; } tr { break-inside:avoid; }
  .table-scroll { overflow:visible; } a { color:inherit; }
}
"""


def render(source: Path, destination: Path) -> None:
    text = source.read_text(encoding="utf-8")
    title = next(line[2:] for line in text.splitlines() if line.startswith("# "))
    renderer = MarkdownIt("commonmark", {"html": False}).enable("table")
    body = renderer.render(text)
    body = body.replace("<table>", '<div class="table-scroll"><table>')
    body = body.replace("</table>", "</table></div>")
    document = (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>"
        f"{body}<footer>Local research report · "
        f'<a href="{html.escape(source.name, quote=True)}">Markdown source</a>'
        " · Supporting evidence is linked inline.</footer></main></body></html>\n"
    )
    destination.write_text(document, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.source.with_suffix(".html")
    render(args.source, output)
    print(output)
