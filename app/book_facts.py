from __future__ import annotations

import re
from pathlib import Path
from bs4 import BeautifulSoup

from config import Config

_WORD_NUMS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16,
}

_BOOK_FACTS: dict | None = None


def _extract_chapters_from_html(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    all_tags = [c for c in soup.children if getattr(c, "name", None)]
    chapters = []

    for index, tag in enumerate(all_tags):
        if tag.name != "p":
            continue
        text = tag.get_text(" ", strip=True).lower()
        match = re.match(r"^chapter\s+([\w]+)$", text)
        if not match:
            continue
        token = match.group(1)
        if token not in _WORD_NUMS:
            continue

        title = ""
        for lookahead in range(index, min(index + 6, len(all_tags))):
            if all_tags[lookahead].name == "h1":
                title = all_tags[lookahead].get_text(" ", strip=True)
                break

        chapters.append({"number": _WORD_NUMS[token], "title": title})

    chapters.sort(key=lambda item: item["number"])
    return chapters


def get_book_facts() -> dict:
    global _BOOK_FACTS
    if _BOOK_FACTS is not None:
        return _BOOK_FACTS

    html_path = Path(Config.SOURCE_HTML_PATH)
    if not html_path.is_file() and Config.SOURCE_HTML_PATHS:
        html_path = Path(Config.SOURCE_HTML_PATHS[0])

    html = html_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    title = "The Climate Academy Student Book"
    opening_text = " ".join(
        tag.get_text(" ", strip=True)
        for tag in soup.find_all("p")[:20]
    )
    if "The Climate Academy was founded" in opening_text:
        title = "The Climate Academy Student Book"

    chapters = _extract_chapters_from_html(html)
    intro_text = ""
    intro_anchor = soup.find(string=re.compile(r"Climate change is a reality that is known through science", re.I))
    if intro_anchor is not None and getattr(intro_anchor, "parent", None) is not None:
        parts = []
        for sibling in intro_anchor.parent.find_all_next("p"):
            text = sibling.get_text(" ", strip=True)
            if text:
                parts.append(text)
            if len(parts) >= 4:
                break
        intro_text = " ".join(parts[:4])

    _BOOK_FACTS = {"title": title, "chapters": chapters, "intro_text": intro_text}
    return _BOOK_FACTS


def build_fact_passages(query: str) -> list[dict]:
    q = query.lower().strip()
    facts = get_book_facts()
    chapters = facts["chapters"]

    if not q:
        return []

    if any(phrase in q for phrase in ["title of the book", "book title", "what is the title", "name of the book"]):
        return [{
            "document": f"The title of the book is {facts['title']}.",
            "source_type": "book",
            "section_number": "0.0",
            "section_title": "Book Title",
            "distance": 0.0,
            "chapter_number": 0,
            "chapter_title": facts["title"],
        }]

    if "how many chapter" in q or "number of chapter" in q or "how many chapters" in q:
        return [{
            "document": f"The book has {len(chapters)} chapters.",
            "source_type": "book",
            "section_number": "0.1",
            "section_title": "Chapter Count",
            "distance": 0.0,
            "chapter_number": 0,
            "chapter_title": facts["title"],
        }]

    if any(phrase in q for phrase in ["list those chapters", "list the chapters", "what are the chapters", "chapter list", "name the chapters"]):
        chapter_lines = [f"Chapter {chapter['number']}: {chapter['title']}" for chapter in chapters]
        return [{
            "document": "\n".join(chapter_lines),
            "source_type": "book",
            "section_number": "0.2",
            "section_title": "Chapter List",
            "distance": 0.0,
            "chapter_number": 0,
            "chapter_title": facts["title"],
        }]

    if any(phrase in q for phrase in ["what is climate change", "define climate change", "climate change is"]):
        if facts["intro_text"]:
            return [{
                "document": facts["intro_text"],
                "source_type": "book",
                "section_number": "0.3",
                "section_title": "Introduction",
                "distance": 0.0,
                "chapter_number": 0,
                "chapter_title": "Introduction",
            }]

    return []
