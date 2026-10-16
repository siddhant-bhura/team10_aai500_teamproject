"""Build reports/Final-Project-Report-Team-10.pdf from the markdown source.

Needs two extra packages that the analysis scripts do not need:
    pip install -r requirements-pdf.txt && playwright install chromium
AI-tool disclosure: written with the help of Claude Code (Anthropic);
see README.md.
Run from anywhere:  python tools/build_report.py
If those packages are missing it prints a message and exits with code 0
(the PDF step is optional).
"""
import pathlib
import sys

try:
    import mistune
    from playwright.sync_api import sync_playwright
except ImportError as exc:
    print(f"Report build skipped (optional): {exc.name} is not installed. "
          "To build the PDF: pip install -r requirements-pdf.txt "
          "&& playwright install chromium")
    sys.exit(0)

root = pathlib.Path(__file__).resolve().parent.parent / "reports"
md = (root / "Final-Project-Report-Team-10.md").read_text(encoding="utf-8")
html = mistune.create_markdown(escape=False, plugins=["table"])(md)
css = """
@page{size:Letter;margin:0.9in 0.8in}
body{font-family:Georgia,'Times New Roman',serif;font-size:10.5pt;line-height:1.45;color:#111}
h1{font-size:26pt;margin:0 0 6pt}h2{font-size:16pt;border-bottom:1px solid #999;padding-bottom:3pt;margin-top:0}
h3{font-size:12.5pt;margin-top:16pt;break-after:avoid}
.pagebreak{break-after:page}
.cover{text-align:center;padding-top:2.2in}.cover h2{border:none;font-size:18pt}.cover p{margin:10pt 0}
table{border-collapse:collapse;width:100%;margin:8pt 0;font-size:9.5pt;break-inside:avoid}
th,td{border:1px solid #bbb;padding:3pt 6pt;text-align:left;vertical-align:top}th{background:#eee}
img{max-width:100%;display:block;margin:8pt auto 2pt}
p:has(>img){break-inside:avoid;text-align:center}
pre{background:#f5f5f5;border:1px solid #ddd;padding:6pt;font-size:6.3pt;line-height:1.3;white-space:pre;break-inside:avoid}
code{font-family:Menlo,monospace;font-size:8.8pt}pre code{font-size:6.3pt}
em{color:#333}
.nb h4{font-size:11pt;margin:12pt 0 4pt}.nb pre{font-size:6.3pt;margin:2pt 0;white-space:pre-wrap}
.nb .output_area pre{background:#fff}.nb{max-width:100%;overflow:hidden}.nb table{font-size:6.5pt;width:100%}
.nb pre{word-break:break-all}
"""  # noqa: E501 (CSS text)


def notebook_appendix():
    """Return the executed notebook as an HTML fragment (code and output).

    Image outputs are replaced by a short note: most repeat figures that are
    already in Sections 3 to 5 and they would add about ten pages. Returns an
    empty string if nbconvert is not installed.
    """
    try:
        import nbformat
        from nbconvert import HTMLExporter
    except ImportError:
        print("Notebook appendix skipped: nbconvert is not installed.")
        return ""
    nb_path = root.parent / "notebooks" / "final_analysis.ipynb"
    nb = nbformat.read(nb_path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            data = out.get("data")
            if data and data.get("text/html", "").count("<th") > 14:
                data.pop("text/html")  # too wide for a page; use plain text
            if data and "image/png" in data:
                data.pop("image/png")
                data["text/plain"] = (
                    "[figure left out of this PDF copy; "
                    "see the notebook file and reports/]")
                data.pop("text/html", None)
    exporter = HTMLExporter(template_name="basic",
                            exclude_input_prompt=True,
                            exclude_output_prompt=True)
    body, _ = exporter.from_notebook_node(nb)
    for old, new in (("<h1", "<h4"), ("</h1>", "</h4>"),
                     ("<h2", "<h4"), ("</h2>", "</h4>")):
        body = body.replace(old, new)
    return '<div class="nb">' + body + "</div>"


html += notebook_appendix()
tmp = root / "_build.html"
tmp.write_text(f'<html><head><meta charset="utf-8"><style>{css}</style>'
               f'</head><body>{html}</body></html>', encoding="utf-8")
footer = ('<div style="font-size:8px;width:100%;text-align:center">'
          'AAI-500 Team 10 | Page <span class="pageNumber"></span> of '
          '<span class="totalPages"></span></div>')
try:
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto(tmp.as_uri())
        pg.wait_for_load_state("load")
        pg.pdf(path=str(root / "Final-Project-Report-Team-10.pdf"),
               format="Letter", print_background=True,
               display_header_footer=True,
               header_template="<span></span>", footer_template=footer,
               margin={"top": "0.8in", "bottom": "0.8in",
                       "left": "0.8in", "right": "0.8in"})
        b.close()
finally:
    tmp.unlink(missing_ok=True)
print("Wrote reports/Final-Project-Report-Team-10.pdf")
