"""
scrape_to_txt.py
----------------
F1 Historian AI — scrape Wikipedia → one txt file per article

Fetches all F1 articles via the Wikipedia "action=parse" API (full rendered
HTML) and converts each one to a structured plain-text file that PRESERVES:

    ✅ Infoboxes              (as "Key: Value" lines)
    ✅ Tables                 (as pipe-delimited rows, incl. championship /
                                race-result tables)
    ✅ Lists                  (bulleted / numbered)
    ✅ Section headings       (nested, with markdown-style #'s)
    ✅ References             (as a flat numbered list at the end)
    ✅ Statistics tables

It deliberately drops noise that isn't useful for RAG: edit-section links,
navboxes, "see also" sister-project boxes, citation-needed flags, hatnotes,
and raw wiki templates/markup.

Output layout (unchanged):

    f1_data/
        drivers/
            ayrton_senna.txt
        teams/
        seasons/
        circuits/
        rivalries/
        technical/
        general/

Usage:
    python scrape_to_txt.py
    python scrape_to_txt.py --out my_folder

Requirements:
    pip install requests beautifulsoup4
"""

import argparse
import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

# config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUTPUT_DIR    = Path("f1_data")
WIKI_API      = "https://en.wikipedia.org/w/api.php"
HEADERS       = {
    "User-Agent": "F1HistorianAI/1.0 (educational project; contact@example.com)"
}
REQUEST_DELAY = 5.0   # delay to be polite to wiki api
MAX_RETRIES   = 4
BACKOFF_BASE  = 20.0

# selectors to remove from html
STRIP_SELECTORS = [
    ".mw-editsection",       # edit links
    ".noprint",               # print hidden elements
    ".navbox",                 # navigation boxes
    ".vertical-navbox",
    ".sistersitebox",         # sister site links
    ".ambox",                  # article warning banners
    ".hatnote",                # disambiguation notes
    ".mw-empty-elt",
    ".reflist-lower-alpha",
    "sup.reference",          # citation markers
    ".mw-cite-backlink",
    ".thumb",                  # image thumbnails
    "style",
    "script",
]

# articles to scrape

ALL_ARTICLES = {
    "DRIVERS": [
        # historical
        "Ayrton Senna",
        "Alain Prost",
        "Nigel Mansell",
        "Nelson Piquet",
        "Niki Lauda",
        "James Hunt",
        "Jackie Stewart",
        "Jim Clark",
        "Juan Manuel Fangio",
        "Stirling Moss",
        "Michael Schumacher",
        "Damon Hill",
        "Mika Häkkinen",
        "David Coulthard",
        "Rubens Barrichello",
        "Nico Rosberg",
        "Lewis Hamilton",
        "Sebastian Vettel",
        "Fernando Alonso",
        "Kimi Räikkönen",
        "Jenson Button",
        "Mark Webber",
        "Max Verstappen",
        "Charles Leclerc",
        "Lando Norris",
        "George Russell",
        "Carlos Sainz Jr.",
        "Oscar Piastri",
        "Lance Stroll",
        "Pierre Gasly",
        "Esteban Ocon",
        "Alexander Albon",
        "Yuki Tsunoda",
        "Andrea Kimi Antonelli",
        "Oliver Bearman",
        "Gabriel Bortoleto",
        "Isack Hadjar",
        "Liam Lawson",
        "Franco Colapinto",
        "Nico Hülkenberg",
        "Sergio Pérez",
        "Valtteri Bottas",
        "Arvid Lindblad",
        "Jack Doohan",
    ],
    "TEAMS": [
        "McLaren",
        "Scuderia Ferrari",
        "Williams Racing",
        "Red Bull Racing",
        "Mercedes AMG Petronas Formula One Team",
        "Renault in Formula One",
        "Lotus F1",
        "Tyrrell Racing",
        "Brabham",
        "Benetton Formula",
        "Brawn GP",
        "Honda in Formula One",
        "Aston Martin in Formula One",
        "Racing Bulls",              
        "Alpine F1 Team",
        "Haas F1 Team",
        "Kick Sauber",               
        "Audi in Formula One",       
        "Cadillac F1 Team",          
    ],
    "SEASONS": [
        "1976 Formula One season",
        "1984 Formula One World Championship",
        "1986 Formula One World Championship",
        "1988 Formula One World Championship",
        "1989 Formula One World Championship",
        "1991 Formula One World Championship",
        "1992 Formula One World Championship",
        "1993 Formula One World Championship",
        "1994 Formula One World Championship",
        "1996 Formula One World Championship",
        "1999 Formula One World Championship",
        "2000 Formula One World Championship",
        "2001 Formula One World Championship",
        "2002 Formula One World Championship",
        "2003 Formula One World Championship",
        "2004 Formula One World Championship",
        "2005 Formula One World Championship",
        "2006 Formula One World Championship",
        "2007 Formula One World Championship",
        "2008 Formula One World Championship",
        "2009 Formula One World Championship",
        "2010 Formula One World Championship",
        "2011 Formula One World Championship",
        "2012 Formula One World Championship",
        "2013 Formula One World Championship",
        "2014 Formula One World Championship",
        "2015 Formula One World Championship",
        "2016 Formula One World Championship",
        "2017 Formula One World Championship",
        "2018 Formula One World Championship",
        "2019 Formula One World Championship",
        "2020 Formula One World Championship",
        "2021 Formula One World Championship",
        "2022 Formula One World Championship",
        "2023 Formula One World Championship",
        "2024 Formula One World Championship",
        "2025 Formula One World Championship",
        "2026 Formula One World Championship",
    ],
    "CIRCUITS": [
        "Circuit de Monaco",
        "Silverstone Circuit",
        "Monza Circuit",
        "Circuit de Spa-Francorchamps",
        "Suzuka International Racing Course",
        "Interlagos",
        "Circuit of the Americas",
        "Nürburgring",
        "Hockenheimring",
        "Bahrain International Circuit",
        "Yas Marina Circuit",
        "Baku City Circuit",
        "Jeddah Corniche Circuit",
        "Miami International Autodrome",
        "Las Vegas Strip Circuit",
        "Lusail International Circuit",
        "Marina Bay Street Circuit",
        "Zandvoort",
    ],
    "RIVALRIES": [
        "Prost–Senna rivalry",
        "Hill-Schumacher rivalry",
        "Hamilton–Rosberg rivalry",
        "Verstappen–Hamilton rivalry",
        "Norris–Piastri rivalry",
    ],
    "TECHNICAL": [
        "Formula One regulations",
        "Formula One car",
        "Kinetic energy recovery system",
        "Drag reduction system",
        "Formula One tyres",
        "Ground effect (cars)",
        "Formula One aerodynamics",
        "Formula One engines",
        "2026 Formula One regulations",
    ],
    "GENERAL": [
        "Formula One",
        "List of Formula One World Drivers' Champions",
        "List of Formula One World Constructors' Champions",
        "History of Formula One",
        "Formula One safety",
        "List of Formula One drivers",
        "List of Formula One constructors",
    ],
}

# wikipedia api fetch helper

def fetch_article_html(title: str) -> tuple[str, str]:
    """
    Fetch rendered HTML + canonical URL for a Wikipedia article via
    action=parse. Returns (html, url). Returns ("", "") on failure.
    """
    params = {
        "action":        "parse",
        "page":          title,
        "prop":          "text",
        "redirects":     True,
        "format":        "json",
        "formatversion": 2,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                WIKI_API, params=params, headers=HEADERS, timeout=30
            )

            if resp.status_code == 429:
                wait = BACKOFF_BASE ** attempt
                log.warning(f"  Rate limited, waiting {wait:.0f}s…")
                time.sleep(wait)
                continue

            if resp.status_code in (500, 502, 503, 504):
                wait = BACKOFF_BASE ** attempt
                log.warning(f"  Server error {resp.status_code}, waiting {wait:.0f}s…")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            if "error" in data:
                code = data["error"].get("code", "")
                if code in ("missingtitle",):
                    log.warning(f"  Not found on Wikipedia: '{title}'")
                    return "", ""
                log.warning(f"  API error for '{title}': {data['error']}")
                return "", ""

            parse = data.get("parse", {})
            html  = parse.get("text", "")
            page_title = parse.get("title", title)
            url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
            return html, url

        except requests.exceptions.Timeout:
            log.warning(f"  Timeout on attempt {attempt}/{MAX_RETRIES}")
            time.sleep(BACKOFF_BASE ** attempt)

        except requests.exceptions.ConnectionError as e:
            log.warning(f"  Connection error on attempt {attempt}/{MAX_RETRIES}: {e}")
            time.sleep(BACKOFF_BASE ** attempt)

        except requests.RequestException as e:
            log.error(f"  Non-retryable error for '{title}': {e}")
            return "", ""

    log.error(f"  Gave up on '{title}' after {MAX_RETRIES} attempts")
    return "", ""


# html parsing helpers

def _clean_cell_text(tag: Tag) -> str:
    """Get readable text from a table cell, collapsing whitespace."""
    text = tag.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _infobox_to_text(table: Tag) -> str:
    """Render an infobox table as 'Key: Value' lines."""
    lines = []
    caption = table.find("caption")
    if caption:
        cap_text = _clean_cell_text(caption)
        if cap_text:
            lines.append(f"## Infobox: {cap_text}")
    else:
        lines.append("## Infobox")

    for row in table.find_all("tr"):
        header_cell = row.find("th")
        data_cells = row.find_all("td")
        if header_cell and data_cells:
            key = _clean_cell_text(header_cell)
            val = " | ".join(_clean_cell_text(td) for td in data_cells if _clean_cell_text(td))
            if key and val:
                lines.append(f"{key}: {val}")
        elif header_cell and not data_cells:
            # infobox section divider
            key = _clean_cell_text(header_cell)
            if key:
                lines.append(f"-- {key} --")
        elif data_cells and not header_cell:
            val = " | ".join(_clean_cell_text(td) for td in data_cells if _clean_cell_text(td))
            if val:
                lines.append(val)
    return "\n".join(lines) + "\n"


def _table_to_text(table: Tag, label: str = "Table") -> str:
    """Render a generic (non-infobox) table as pipe-delimited rows."""
    rows_text = []
    caption = table.find("caption")
    title = _clean_cell_text(caption) if caption else label

    for row in table.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if not cells:
            continue
        cell_texts = [_clean_cell_text(c) for c in cells]
        cell_texts = [c for c in cell_texts if c != ""]
        if cell_texts:
            rows_text.append(" | ".join(cell_texts))

    if not rows_text:
        return ""

    out = [f"## {title}"]
    out.extend(rows_text)
    return "\n".join(out) + "\n"


def _list_to_text(list_tag: Tag, indent: int = 0) -> str:
    """Render <ul>/<ol> as bullet/numbered lines, recursing into nested lists."""
    lines = []
    ordered = list_tag.name == "ol"
    idx = 1
    for li in list_tag.find_all("li", recursive=False):
        # get text excluding nested list text
        nested_lists = li.find_all(["ul", "ol"], recursive=False)
        li_copy_text_parts = []
        for child in li.children:
            if isinstance(child, Tag) and child.name in ("ul", "ol"):
                continue
            if isinstance(child, NavigableString):
                li_copy_text_parts.append(str(child))
            elif isinstance(child, Tag):
                li_copy_text_parts.append(child.get_text(" ", strip=True))
        text = re.sub(r"\s+", " ", " ".join(li_copy_text_parts)).strip()
        bullet = f"{idx}." if ordered else "-"
        if text:
            lines.append(f"{'  ' * indent}{bullet} {text}")
        if ordered:
            idx += 1
        for nested in nested_lists:
            nested_text = _list_to_text(nested, indent + 1)
            if nested_text:
                lines.append(nested_text)
    return "\n".join(lines)


def extract_references(soup: BeautifulSoup) -> str:
    """Pull the References / citation list into a flat numbered block."""
    ref_lines = []
    ref_list = soup.select_one(".reflist, ol.references")
    if not ref_list:
        return ""
    items = ref_list.find_all("li")
    for i, li in enumerate(items, 1):
        text = li.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"^\^\s*", "", text)
        if text:
            ref_lines.append(f"[{i}] {text}")
    if not ref_lines:
        return ""
    return "## References\n" + "\n".join(ref_lines) + "\n"


def html_to_structured_text(html: str) -> str:
    """
    Convert full Wikipedia article HTML into structured plain text that
    preserves infoboxes, tables, lists, and section hierarchy.
    """
    soup = BeautifulSoup(html, "html.parser")

    # remove ui elements
    for selector in STRIP_SELECTORS:
        for el in soup.select(selector):
            el.decompose()

    body = soup.find(class_="mw-parser-output") or soup

    # extract infoboxes first
    infobox_blocks = []
    for table in body.select("table.infobox"):
        infobox_blocks.append(_infobox_to_text(table))
        table.decompose()

    # extract references separately
    references_text = extract_references(body)
    for ref_section in body.select(".reflist, ol.references"):
        ref_section.decompose()

    heading_tags = {"h2", "h3", "h4", "h5", "h6"}
    out_lines = []

    SKIP_HEADINGS = {"references", "external links", "see also", "notes"}

    skip_until_next_heading = False

    for el in body.find_all(recursive=False):
        _walk(el, out_lines, heading_tags, skip_state={"skip": False})

    body_text = "\n".join(out_lines)
    # collapse duplicate blank lines
    body_text = re.sub(r"\n{3,}", "\n\n", body_text).strip()

    parts = []
    if infobox_blocks:
        parts.append("\n\n".join(infobox_blocks))
    parts.append(body_text)
    if references_text:
        parts.append(references_text)

    return "\n\n".join(p for p in parts if p).strip()


def _walk(el: Tag, out: list, heading_tags: set, skip_state: dict) -> None:
    """Recursively walk top-level child elements and append rendered text."""
    if not isinstance(el, Tag):
        return

    name = el.name

    if name in heading_tags:
        text = el.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        if text.lower() in {"references", "external links", "see also", "notes"}:
            skip_state["skip"] = True
            return
        skip_state["skip"] = False
        level = int(name[1])
        out.append(f"\n{'#' * level} {text}\n")
        return

    if skip_state["skip"]:
        return

    if name == "p":
        text = el.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            out.append(text)
        return

    if name in ("ul", "ol"):
        list_text = _list_to_text(el)
        if list_text:
            out.append(list_text)
        return

    if name == "table":
        table_text = _table_to_text(el)
        if table_text:
            out.append(table_text)
        return

    if name in ("div", "section"):
        for child in el.find_all(recursive=False):
            _walk(child, out, heading_tags, skip_state)
        return

    # fallback for other containers
    if name in ("dl", "blockquote", "figure"):
        text = el.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            out.append(text)


# text cleaning

def clean(text: str) -> str:
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[nb \d+\]|\[note \d+\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


# main execution

def slugify(title: str) -> str:
    """Convert an article title to a safe filename: 'Ayrton Senna' → 'ayrton_senna'."""
    s = title.lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s-]+", "_", s)
    return s.strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape F1 Wikipedia articles to individual txt files")
    parser.add_argument(
        "--out", type=Path, default=OUTPUT_DIR,
        help=f"Output folder (default: {OUTPUT_DIR}/)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-fetch and overwrite files that already exist (replaces old data)"
    )
    args = parser.parse_args()

    total_articles = sum(len(v) for v in ALL_ARTICLES.values())
    print("\n" + "=" * 60)
    print("  F1 Historian AI — Wikipedia → TXT scraper")
    print(f"  {total_articles} articles  →  {args.out}/")
    print("=" * 60 + "\n")

    done        = 0
    fetched     = 0
    skipped     = 0
    failed      = []
    total_chars = 0

    for category, titles in ALL_ARTICLES.items():
        category_dir = args.out / category.lower()
        category_dir.mkdir(parents=True, exist_ok=True)

        for title in titles:
            done += 1
            filename = category_dir / f"{slugify(title)}.txt"

            if filename.exists() and not args.force:
                log.info(f"[{done}/{total_articles}] SKIP (exists): {title}")
                skipped += 1
                continue

            log.info(f"[{done}/{total_articles}] {title}")
            html, url = fetch_article_html(title)

            if not html:
                failed.append((category, title))
                time.sleep(REQUEST_DELAY)
                continue

            try:
                structured = html_to_structured_text(html)
            except Exception as e:
                log.error(f"  Failed to parse HTML for '{title}': {e}")
                failed.append((category, title))
                time.sleep(REQUEST_DELAY)
                continue

            cleaned      = clean(structured)
            total_chars += len(cleaned)
            fetched     += 1

            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"ARTICLE  : {title}\n")
                f.write(f"CATEGORY : {category}\n")
                f.write(f"URL      : {url}\n")
                f.write("─" * 60 + "\n\n")
                f.write(cleaned)
                f.write("\n")

            log.info(f"  → {filename}  ({len(cleaned):,} chars)")
            time.sleep(REQUEST_DELAY)

    total_kb = sum(
        f.stat().st_size for f in args.out.rglob("*.txt")
    ) / 1024

    print("\n" + "=" * 60)
    print(f"  Done.")
    print(f"  Written  : {fetched} new files")
    print(f"  Skipped  : {skipped} already existed")
    print(f"  Failed   : {len(failed)}")
    print(f"  Total KB : {total_kb:.0f} KB")
    print(f"  Folder   : {args.out}/")
    print("=" * 60)

    if failed:
        print(f"\nFailed articles ({len(failed)}):")
        for cat, t in failed:
            print(f"  [{cat}] {t}")
        print("\nThese may have slightly different Wikipedia titles.")
        print("Check and correct the title in ALL_ARTICLES, then re-run.")

    print(f"\nFolder structure:")
    for category_dir in sorted(args.out.iterdir()):
        if category_dir.is_dir():
            files = list(category_dir.glob("*.txt"))
            print(f"  {args.out.name}/{category_dir.name}/  ({len(files)} files)")

    print("\nNext: feed these files into your RAG pipeline.\n")


if __name__ == "__main__":
    main()