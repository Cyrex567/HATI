"""Render report_content.txt into a branded HATI PDF with ReportLab.

You should not need to edit this file to change the document. All prose lives
in report_content.txt. This script owns the layout: fonts, palette, cover page,
running header and footer, table of contents, tables, figures and callout boxes.

    python build_report.py                 -> HATI_saturation_analysis.pdf
    python build_report.py other.txt out.pdf
"""
from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                                NextPageTemplate, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

HERE = Path(__file__).resolve().parent

# ------------------------------------------------------------------ palette
NAVY = colors.HexColor("#0D1B2A")
BLUE = colors.HexColor("#1F3A5F")
RED = colors.HexColor("#C1121F")
SILVER = colors.HexColor("#8B97A5")
TEAL = colors.HexColor("#1B7A6E")
AMBER = colors.HexColor("#9A6B00")
INK = colors.HexColor("#1C232D")
INK_DIM = colors.HexColor("#4A5561")
RULE = colors.HexColor("#D8DEE7")
PLAINBG = colors.HexColor("#EAF2FB")
KEEPBG = colors.HexColor("#EAF5EF")
LIMITBG = colors.HexColor("#FBEEEC")
NOTEBG = colors.HexColor("#FBF4E4")
CODEBG = colors.HexColor("#0A111C")
CODEINK = colors.HexColor("#E8EEF7")

BOX_KINDS = {
    "plain": (PLAINBG, BLUE),
    "keep": (KEEPBG, TEAL),
    "limit": (LIMITBG, RED),
    "note": (NOTEBG, AMBER),
}

# ------------------------------------------------------------------ fonts
def _register(name: str, files: dict[str, str], fallback: dict[str, str]) -> dict[str, str]:
    """Register a TrueType family from C:/Windows/Fonts, or fall back to built-ins."""
    fontdir = Path("C:/Windows/Fonts")
    try:
        for variant, fname in files.items():
            pdfmetrics.registerFont(TTFont(f"{name}-{variant}", str(fontdir / fname)))
        pdfmetrics.registerFontFamily(
            name,
            normal=f"{name}-regular", bold=f"{name}-bold",
            italic=f"{name}-italic", boldItalic=f"{name}-bolditalic")
        return {k: f"{name}-{k}" for k in files}
    except Exception:  # noqa: BLE001
        return fallback


BODY = _register("Palatino",
                 {"regular": "pala.ttf", "bold": "palab.ttf",
                  "italic": "palai.ttf", "bolditalic": "palabi.ttf"},
                 {"regular": "Times-Roman", "bold": "Times-Bold",
                  "italic": "Times-Italic", "bolditalic": "Times-BoldItalic"})
SANS = _register("Segoe",
                 {"regular": "segoeui.ttf", "bold": "seguisb.ttf",
                  "italic": "segoeuii.ttf", "bolditalic": "seguisbi.ttf"},
                 {"regular": "Helvetica", "bold": "Helvetica-Bold",
                  "italic": "Helvetica-Oblique", "bolditalic": "Helvetica-BoldOblique"})
MONO = _register("Consolas",
                 {"regular": "consola.ttf", "bold": "consolab.ttf",
                  "italic": "consolai.ttf", "bolditalic": "consolaz.ttf"},
                 {"regular": "Courier", "bold": "Courier-Bold",
                  "italic": "Courier-Oblique", "bolditalic": "Courier-BoldOblique"})

# ------------------------------------------------------------------ page geometry
PAGE_W, PAGE_H = A4
MARGIN_L = 2.3 * cm
MARGIN_R = 2.3 * cm
MARGIN_T = 2.6 * cm
MARGIN_B = 2.4 * cm
TEXT_W = PAGE_W - MARGIN_L - MARGIN_R

# ------------------------------------------------------------------ styles
def S(name, **kw):
    base = dict(fontName=BODY["regular"], fontSize=10.2, leading=14.4,
                textColor=INK, spaceAfter=6, alignment=TA_LEFT)
    base.update(kw)
    return ParagraphStyle(name, **base)


ST = {
    "body": S("body"),
    "h1": S("h1", fontName=SANS["bold"], fontSize=19, leading=23, textColor=NAVY,
            spaceBefore=14, spaceAfter=4),
    "h2": S("h2", fontName=SANS["bold"], fontSize=13, leading=16.5, textColor=BLUE,
            spaceBefore=14, spaceAfter=4),
    "h3": S("h3", fontName=SANS["bold"], fontSize=10.8, leading=14, textColor=NAVY,
            spaceBefore=10, spaceAfter=3),
    "bullet": S("bullet", leftIndent=14, bulletIndent=3, spaceAfter=3),
    "number": S("number", leftIndent=18, bulletIndent=3, spaceAfter=3),
    "quote": S("quote", fontName=BODY["italic"], fontSize=11.4, leading=15.5,
               textColor=BLUE, leftIndent=18, rightIndent=18, spaceBefore=6, spaceAfter=8),
    "cmd": S("cmd", fontName=MONO["regular"], fontSize=9, leading=12.5, textColor=CODEINK,
             backColor=CODEBG, borderPadding=(5, 8, 5, 8), leftIndent=8, spaceBefore=3,
             spaceAfter=7),
    "caption": S("caption", fontName=SANS["regular"], fontSize=8.6, leading=11.5,
                 textColor=INK_DIM, spaceBefore=4, spaceAfter=12),
    "tcaption": S("tcaption", fontName=SANS["regular"], fontSize=8.6, leading=11.5,
                  textColor=INK_DIM, spaceBefore=3, spaceAfter=12),
    "th": S("th", fontName=SANS["bold"], fontSize=8.4, leading=10.5, textColor=NAVY, spaceAfter=0),
    "td": S("td", fontSize=9.1, leading=11.6, textColor=INK, spaceAfter=0),
    "boxtitle": S("boxtitle", fontName=SANS["bold"], fontSize=8.6, leading=11, spaceAfter=2),
    "boxbody": S("boxbody", fontSize=9.6, leading=13.2, spaceAfter=4),
    "cover_title": S("cover_title", fontName=SANS["bold"], fontSize=27, leading=32,
                     textColor=NAVY, alignment=TA_CENTER, spaceAfter=10),
    "cover_sub": S("cover_sub", fontName=SANS["regular"], fontSize=12.5, leading=17,
                   textColor=BLUE, alignment=TA_CENTER, spaceAfter=0),
    "cover_meta": S("cover_meta", fontSize=11.5, leading=16, textColor=INK, alignment=TA_CENTER,
                    spaceAfter=0),
    "cover_meta_i": S("cover_meta_i", fontName=BODY["italic"], fontSize=10.5, leading=15,
                      textColor=INK_DIM, alignment=TA_CENTER, spaceAfter=0),
    "cover_foot": S("cover_foot", fontName=SANS["regular"], fontSize=8, leading=11,
                    textColor=SILVER, alignment=TA_CENTER),
    "toc_title": S("toc_title", fontName=SANS["bold"], fontSize=15, leading=19, textColor=NAVY,
                   spaceAfter=10),
    "toc1": S("toc1", fontName=SANS["regular"], fontSize=10, leading=15, textColor=NAVY,
              leftIndent=0),
    "toc2": S("toc2", fontSize=9.4, leading=13.2, textColor=INK_DIM, leftIndent=16),
}

# ------------------------------------------------------------------ inline markup
_INLINE = [
    (re.compile(r"\*\*(.+?)\*\*"), r"<b>\1</b>"),
    (re.compile(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)"), r"<i>\1</i>"),
    (re.compile(r"`(.+?)`"), rf'<font face="{MONO["regular"]}" size="8.8">\1</font>'),
]


def inline(text: str) -> str:
    """Escape for Paragraph, then apply the tiny markup."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for rx, rep in _INLINE:
        text = rx.sub(rep, text)
    return text


# ------------------------------------------------------------------ flowable builders
def figure(path: str, caption: str, width_frac: float) -> list:
    p = (HERE / path)
    if not p.exists():
        return [Paragraph(f"[missing figure: {path}]", ST["caption"])]
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(p)).getSize()
    w = TEXT_W * max(0.2, min(1.0, width_frac))
    h = w * ih / iw
    max_h = PAGE_H - MARGIN_T - MARGIN_B - 3 * cm
    if h > max_h:
        h = max_h
        w = h * iw / ih
    img = Image(str(p), width=w, height=h)
    img.hAlign = "CENTER"
    return [KeepTogether([Spacer(1, 4), img, Paragraph(inline(caption), ST["caption"])])]


def table(rows: list[list[str]], caption: str | None) -> list:
    if not rows:
        return []
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    data = [[Paragraph(inline(c), ST["th"]) for c in rows[0]]]
    for r in rows[1:]:
        data.append([Paragraph(inline(c), ST["td"]) for c in r])

    # column widths proportional to the longest cell, with a floor
    lens = [max(len(re.sub(r"<[^>]+>", "", rows[i][j])) for i in range(len(rows))) for j in range(ncol)]
    lens = [max(6, min(l, 48)) for l in lens]
    total = sum(lens)
    widths = [TEXT_W * l / total for l in lens]
    if ncol == 8 and rows[0][0] == 'Frame':
        widths = [TEXT_W*f for f in (.24,.10,.10,.085,.08,.14,.125,.13)]

    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.9, NAVY),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, NAVY),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, NAVY),
        ("LINEBELOW", (0, 1), (-1, -2), 0.3, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    out = [Spacer(1, 4), t]
    if caption:
        out.append(Paragraph(inline(caption), ST["tcaption"]))
    else:
        out.append(Spacer(1, 10))
    return [KeepTogether(out)] if len(rows) <= 14 else out


def box(kind: str, title: str, paragraphs: list[str]) -> list:
    bg, frame = BOX_KINDS.get(kind, BOX_KINDS["plain"])
    inner = []
    if title:
        st = ParagraphStyle("bt", parent=ST["boxtitle"], textColor=frame)
        inner.append(Paragraph(inline(title.upper()), st))
    for p in paragraphs:
        inner.append(Paragraph(inline(p), ST["boxbody"]))
    t = Table([[inner]], colWidths=[TEXT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX", (0, 0), (-1, -1), 0.7, frame),
        ("LEFTPADDING", (0, 0), (-1, -1), 11),
        ("RIGHTPADDING", (0, 0), (-1, -1), 11),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return [Spacer(1, 6), KeepTogether([t]), Spacer(1, 10)]


class Heading(Paragraph):
    """A Paragraph that knows its TOC level and a bookmark key."""
    def __init__(self, text, style, level, key):
        super().__init__(text, style)
        self.toc_level = level
        self.toc_text = re.sub(r"<[^>]+>", "", text)
        self.bookmark_key = key


class RedRule(Spacer):
    """Short red rule under a chapter heading, in the house style."""
    def __init__(self):
        super().__init__(1, 9)

    def draw(self):
        self.canv.setStrokeColor(RED)
        self.canv.setLineWidth(1.1)
        self.canv.line(0, 5, 1.6 * cm, 5)


# ------------------------------------------------------------------ parser
def parse(text: str) -> tuple[dict, list]:
    meta: dict[str, str] = {}
    story: list = []
    lines = text.splitlines()
    i, n = 0, len(lines)
    para: list[str] = []
    chapter = 0
    section = 0
    subsection = 0
    key_counter = 0

    def flush_para():
        nonlocal para
        if para:
            story.append(Paragraph(inline(" ".join(s.strip() for s in para)), ST["body"]))
            para = []

    def new_key():
        nonlocal key_counter
        key_counter += 1
        return f"h{key_counter}"

    while i < n:
        raw = lines[i]
        line = raw.rstrip()
        stripped = line.strip()

        if stripped.startswith("!meta "):
            flush_para()
            k, _, v = stripped[6:].partition("=")
            meta[k.strip()] = v.strip()
            i += 1
            continue

        if not stripped:
            flush_para()
            i += 1
            continue

        if stripped == "!pagebreak":
            flush_para()
            story.append(PageBreak())
            i += 1
            continue

        if stripped == "!toc":
            flush_para()
            story.append(("TOC",))
            i += 1
            continue

        if stripped.startswith("### "):
            flush_para()
            subsection += 1
            label = f"{chapter}.{section}.{subsection}  {stripped[4:]}"
            story.append(Heading(inline(label), ST["h3"], 2, new_key()))
            i += 1
            continue
        if stripped.startswith("## "):
            flush_para()
            section += 1
            subsection = 0
            label = f"{chapter}.{section}  {stripped[3:]}"
            story.append(Heading(inline(label), ST["h2"], 1, new_key()))
            i += 1
            continue
        if stripped.startswith("# "):
            flush_para()
            chapter += 1
            section = 0
            subsection = 0
            label = f"{chapter}  {stripped[2:]}"
            story.append(Heading(inline(label), ST["h1"], 0, new_key()))
            story.append(RedRule())
            i += 1
            continue

        if stripped.startswith("- "):
            flush_para()
            story.append(Paragraph(inline(stripped[2:]), ST["bullet"], bulletText="\u25aa"))
            i += 1
            continue
        m = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if m:
            flush_para()
            story.append(Paragraph(inline(m.group(2)), ST["number"], bulletText=m.group(1) + "."))
            i += 1
            continue

        if stripped.startswith("> "):
            flush_para()
            story.append(Paragraph(inline(stripped[2:]), ST["quote"]))
            i += 1
            continue

        if stripped.startswith("!cmd "):
            flush_para()
            story.append(Paragraph(inline(stripped[5:]), ST["cmd"]))
            i += 1
            continue

        if stripped.startswith("!fig "):
            flush_para()
            parts = [p.strip() for p in stripped[5:].split("|")]
            path = parts[0]
            caption = parts[1] if len(parts) > 1 else ""
            width = float(parts[2]) if len(parts) > 2 and parts[2] else 1.0
            story.extend(figure(path, caption, width))
            i += 1
            continue

        if stripped.startswith("!box"):
            flush_para()
            head = stripped[4:].strip()
            kind, _, title = head.partition("|")
            kind = kind.strip() or "plain"
            title = title.strip()
            paras: list[str] = []
            buf: list[str] = []
            i += 1
            while i < n and lines[i].strip() != "!endbox":
                s = lines[i].strip()
                if not s:
                    if buf:
                        paras.append(" ".join(buf)); buf = []
                else:
                    buf.append(s)
                i += 1
            if buf:
                paras.append(" ".join(buf))
            story.extend(box(kind, title, paras))
            i += 1
            continue

        if stripped == "!table":
            flush_para()
            rows: list[list[str]] = []
            caption = None
            i += 1
            while i < n and lines[i].strip() != "!endtable":
                s = lines[i].strip()
                if s.startswith("!caption"):
                    caption = s[8:].strip()
                elif s.startswith("|"):
                    cells = [c.strip() for c in s.strip("|").split("|")]
                    if all(re.fullmatch(r"-{2,}:?|:?-{2,}:?", c) for c in cells):
                        pass  # separator row
                    else:
                        rows.append(cells)
                i += 1
            story.extend(table(rows, caption))
            i += 1
            continue

        para.append(line)
        i += 1

    flush_para()
    return meta, story


# ------------------------------------------------------------------ document template
class Report(BaseDocTemplate):
    def __init__(self, filename, meta, **kw):
        super().__init__(filename, pagesize=A4, leftMargin=MARGIN_L, rightMargin=MARGIN_R,
                         topMargin=MARGIN_T, bottomMargin=MARGIN_B,
                         title=meta.get("title", "HATI report"),
                         author=meta.get("author", ""), subject=meta.get("subtitle", ""), **kw)
        self.meta = meta
        frame = Frame(MARGIN_L, MARGIN_B, TEXT_W, PAGE_H - MARGIN_T - MARGIN_B, id="body",
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        cover = Frame(MARGIN_L, MARGIN_B, TEXT_W, PAGE_H - MARGIN_T - MARGIN_B, id="cover",
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([
            PageTemplate(id="Cover", frames=[cover], onPage=self._cover_page),
            # drawn at page end so the running head names the chapter on this page,
            # not the one that happened to be current when the page began
            PageTemplate(id="Body", frames=[frame], onPageEnd=self._body_page),
        ])
        self._current_chapter = ""

    def beforeDocument(self):
        # multiBuild runs several passes over the same object; the running head
        # must not inherit the last chapter of the previous pass
        self._current_chapter = ""

    # ---- page furniture
    def _cover_page(self, canv, doc):
        canv.saveState()
        canv.setFillColor(NAVY)
        canv.rect(0, PAGE_H - 9 * mm, PAGE_W, 9 * mm, stroke=0, fill=1)
        canv.setFillColor(RED)
        canv.rect(0, PAGE_H - 9 * mm - 1.4 * mm, PAGE_W, 1.4 * mm, stroke=0, fill=1)
        canv.setFillColor(SILVER)
        canv.setFont(SANS["regular"], 7.6)
        canv.drawCentredString(PAGE_W / 2, 1.55 * cm,
                               "HAZARD ASSESSMENT AND TERRAIN INTELLIGENCE")
        canv.restoreState()

    def _body_page(self, canv, doc):
        canv.saveState()
        # header
        logo = HERE / "fig" / "hati-logo.png"
        y = PAGE_H - MARGIN_T + 0.85 * cm
        if logo.exists():
            canv.drawImage(str(logo), MARGIN_L, y - 0.15 * cm, width=0.72 * cm, height=0.72 * cm,
                           mask="auto", preserveAspectRatio=True)
        canv.setFont(SANS["regular"], 7.8)
        canv.setFillColor(SILVER)
        canv.drawString(MARGIN_L + 0.95 * cm, y + 0.05 * cm, self.meta.get("docid", "HATI"))
        canv.drawRightString(PAGE_W - MARGIN_R, y + 0.05 * cm, self._current_chapter[:90])
        canv.setStrokeColor(RULE)
        canv.setLineWidth(0.5)
        canv.line(MARGIN_L, y - 0.28 * cm, PAGE_W - MARGIN_R, y - 0.28 * cm)
        # footer
        canv.setFont(SANS["regular"], 7.8)
        canv.setFillColor(SILVER)
        fy = MARGIN_B - 0.95 * cm
        canv.drawString(MARGIN_L, fy, f"HATI  \u00b7  {self.meta.get('title', '')}")
        canv.drawRightString(PAGE_W - MARGIN_R, fy, self.meta.get("date", ""))
        canv.setFillColor(NAVY)
        canv.setFont(SANS["bold"], 8.4)
        canv.drawCentredString(PAGE_W / 2, fy, str(doc.page))
        canv.restoreState()

    # ---- TOC + bookmarks
    def afterFlowable(self, flowable):
        if isinstance(flowable, Heading):
            level = flowable.toc_level
            if level == 0:
                self._current_chapter = flowable.toc_text
            self.canv.bookmarkPage(flowable.bookmark_key)
            self.canv.addOutlineEntry(flowable.toc_text, flowable.bookmark_key,
                                      level=level, closed=(level > 0))
            self.notify("TOCEntry", (level, flowable.toc_text, self.page, flowable.bookmark_key))


# ------------------------------------------------------------------ cover + toc
def cover(meta: dict) -> list:
    logo = HERE / "fig" / "hati-logo.png"
    out = [Spacer(1, 2.6 * cm)]
    if logo.exists():
        img = Image(str(logo), width=4.4 * cm, height=4.4 * cm)
        img.hAlign = "CENTER"
        out.append(img)
    out += [
        Spacer(1, 1.2 * cm),
        Paragraph(inline(meta.get("title", "")), ST["cover_title"]),
        Paragraph(inline(meta.get("subtitle", "")), ST["cover_sub"]),
        Spacer(1, 0.9 * cm),
        _rule_flowable(),
        Spacer(1, 0.9 * cm),
        Paragraph(inline(meta.get("author", "")), ST["cover_meta"]),
        Spacer(1, 0.15 * cm),
        Paragraph(inline(meta.get("affiliation", "")), ST["cover_meta_i"]),
        Spacer(1, 0.15 * cm),
        Paragraph(inline(meta.get("date", "")), ST["cover_meta"]),
        Spacer(1, 0.25 * cm),
        Paragraph(inline(meta.get("version", "")), ST["cover_meta_i"]),
        Spacer(1, 2.2 * cm),
        Paragraph(inline(meta.get("docid", "")), ST["cover_foot"]),
        NextPageTemplate("Body"),
        PageBreak(),
    ]
    return out


class _rule_flowable(Spacer):
    def __init__(self):
        super().__init__(1, 6)

    def draw(self):
        self.canv.setStrokeColor(RED)
        self.canv.setLineWidth(1.2)
        x0 = (TEXT_W - 8 * cm) / 2
        self.canv.line(x0, 3, x0 + 8 * cm, 3)


def toc_flowables() -> list:
    toc = TableOfContents()
    toc.levelStyles = [ST["toc1"], ST["toc2"]]
    toc.dotsMinLevel = 0
    return [Paragraph("Contents", ST["toc_title"]), toc, PageBreak()]


# ------------------------------------------------------------------ main
def build(src: Path, out: Path) -> None:
    meta, body = parse(src.read_text(encoding="utf-8"))
    meta.setdefault("date", date.today().strftime("%B %Y"))
    story = cover(meta) + toc_flowables()
    for item in body:
        if isinstance(item, tuple) and item[0] == "TOC":
            continue
        story.append(item)
    doc = Report(str(out), meta)
    doc.multiBuild(story)
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "report_content.txt"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "HATI_saturation_analysis.pdf"
    build(src, out)
