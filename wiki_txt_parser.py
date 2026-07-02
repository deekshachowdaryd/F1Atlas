"""
wiki_txt_parser.py
-------------------
Parses your scraped .txt article files into the same in-memory article
shape that ingest.py already knows how to chunk/index — so ingest.py
doesn't care whether the source was .json or .txt.

--------------------------------------------------------------------------
EXPECTED .txt FORMAT (based on your files)
--------------------------------------------------------------------------
ARTICLE  : Bahrain International Circuit
CATEGORY : CIRCUITS
URL      : https://en.wikipedia.org/wiki/Bahrain_International_Circuit
────────────────────────────────────────────────────────────
## Infobox: Bahrain International Circuit
...
The Bahrain International Circuit is a 5.412 km motorsport venue...

## History
...

### Construction and design
...

## References
[1] "..." ...

--------------------------------------------------------------------------
What this parser does
--------------------------------------------------------------------------
1. Pulls ARTICLE / CATEGORY / URL out of the header block.
2. Splits everything after the divider line into sections on `##`/`###`
   headings. Text before the first heading becomes an "Introduction"
   section.
3. Drops sections whose heading matches config.SKIP_SECTION_TITLES
   (References, External links, See also, ...) — citation dumps, no
   prose value for RAG.
4. Strips inline footnote markers like "[12]" and collapses whitespace.
5. Auto-tags each section with entity_ids by keyword-matching against
   config.ENTITY_MAP (same driver/team names the rest of the pipeline
   already knows about). No manual tagging needed.
6. Scans the whole article for 4-digit years to derive a season_range
   (min year, max year mentioned) for the year-filter feature.

The result is a dict shaped exactly like the JSON article schema, so
ingest.py's build_chunks() works on it unmodified.
"""
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

from config import ENTITY_MAP, SKIP_SECTION_TITLES, log

_HEADER_RE = re.compile(
    r"ARTICLE\s*:\s*(?P<title>.*?)\s*"
    r"CATEGORY\s*:\s*(?P<category>.*?)\s*"
    r"URL\s*:\s*(?P<url>\S+)",
    re.DOTALL,
)

# Divider line: a run of box-drawing dashes or plain hyphens (at least 8 long)
_DIVIDER_RE = re.compile(r"[─\-]{8,}")

# "## Heading" or "### Heading" on its own line
_HEADING_RE = re.compile(r"^(#{2,3})\s+(.+?)\s*$", re.MULTILINE)

# Footnote markers like "[12]" or "[1]"
_FOOTNOTE_RE = re.compile(r"\[\d+\]")

_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")


def _clean_text(text: str) -> str:
    text = _FOOTNOTE_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "article"


def _auto_tag_entities(text: str) -> List[str]:
    found = []
    text_lower = text.lower()
    for keyword, cid in ENTITY_MAP.items():
        if keyword in text_lower and cid not in found:
            found.append(cid)
    return found


def _extract_season_range(text: str) -> Optional[List[int]]:
    years = [int(y) for y in _YEAR_RE.findall(text)]
    if not years:
        return None
    return [min(years), max(years)]


def parse_txt_file(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8", errors="replace")

    header_match = _HEADER_RE.search(raw)
    if header_match:
        article_title = header_match.group("title").strip()
        category = header_match.group("category").strip()
        source_url = header_match.group("url").strip()
        header_end = header_match.end()
    else:
        log.warning(f"No ARTICLE/CATEGORY/URL header found in '{path.name}', using filename as title")
        article_title = path.stem.replace("_", " ").title()
        category = ""
        source_url = ""
        header_end = 0

    divider_match = _DIVIDER_RE.search(raw, header_end)
    body = raw[divider_match.end():] if divider_match else raw[header_end:]

    # Find all heading positions
    headings = list(_HEADING_RE.finditer(body))

    sections = []

    def _add_section(title: str, content: str):
        content = _clean_text(content)
        if not content:
            return
        if title.strip().lower() in SKIP_SECTION_TITLES:
            return
        sections.append({
            "section_title": title.strip(),
            "text": content,
            "entity_ids": _auto_tag_entities(content),
        })

    if not headings:
        _add_section("Introduction", body)
    else:
        # Text before the first heading
        intro = body[:headings[0].start()]
        _add_section("Introduction", intro)

        for i, h in enumerate(headings):
            title = h.group(2)
            start = h.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
            _add_section(title, body[start:end])

    article_id = _slugify(article_title)
    season_range = _extract_season_range(body)

    return {
        "article_id": article_id,
        "article_title": article_title,
        "category": category,
        "source_url": source_url,
        "season_range": season_range,
        "sections": sections,
    }


if __name__ == "__main__":
    import sys
    import json as _json

    if len(sys.argv) < 2:
        print("Usage: python wiki_txt_parser.py path/to/article.txt")
        sys.exit(1)

    parsed = parse_txt_file(Path(sys.argv[1]))
    print(_json.dumps(parsed, indent=2, ensure_ascii=False))
