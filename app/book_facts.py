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

_DEFAULT_BOOK_METADATA = {
    "title": "The Climate Academy Student Book",
    "author": "Matthew Pye",
    "edition_year": "2025",
    "license": "CC BY-NC-SA",
    "climate_academy_started": "2012",
    "chapters": [
        {"number": 1, "title": "The Absolute Basics"},
        {"number": 2, "title": "Mass Extinction Events"},
        {"number": 3, "title": "Spaceship Earth"},
        {"number": 4, "title": "Where are we now?"},
        {"number": 5, "title": "The United Nations?"},
        {"number": 6, "title": "Who is responsible?"},
        {"number": 7, "title": "The CUTx Percent Index"},
        {"number": 8, "title": "Tipping Points – Physical"},
        {"number": 9, "title": "Paradigm Shifts"},
        {"number": 10, "title": "Tipping Points – Social"},
        {"number": 11, "title": "The Psychology of Climate Change (1)"},
        {"number": 12, "title": "The Psychology of Climate Change (2)"},
        {"number": 13, "title": "The Paradox of Innovation"},
        {"number": 14, "title": "Climate Anxiety"},
        {"number": 15, "title": "The Climate Academy (1) The Piraeus"},
        {"number": 16, "title": "The Climate Academy (2) The Academy"},
    ],
}


def _normalize_query(query: str) -> str:
    text = query.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_book_fact_intent(query: str) -> str | None:
    """
    Return a coarse intent label for metadata-style questions.
    The query should already be translated to English when possible.
    """
    q = _normalize_query(query)
    if not q:
        return None

    title_patterns = [
        r"\bwhat(?:'s| is)? the (?:title|name) of (?:the )?(?:book|textbook|student book)\b",
        r"\bwhat(?:'s| is)? this book called\b",
        r"\bname of (?:the )?(?:book|textbook|student book)\b",
        r"\bbook title\b",
        r"\btextbook title\b",
    ]
    if any(re.search(pattern, q) for pattern in title_patterns):
        return "book_title"

    chapter_count_patterns = [
        r"\bhow many (?:chapters|sections|parts) (?:are|is) (?:there )?(?:in|in the) (?:the )?(?:book|textbook|student book)\b",
        r"\bhow many (?:chapters|sections|parts)\b",
        r"\bnumber of (?:chapters|sections|parts)\b",
        r"\bchapter count\b",
        r"\bsection count\b",
        r"\bcount of (?:chapters|sections|parts)\b",
    ]
    if any(re.search(pattern, q) for pattern in chapter_count_patterns):
        return "chapter_count"

    chapter_list_patterns = [
        r"\b(list|show|name|give me) (?:the )?(?:chapters|chapter names)\b",
        r"\bwhat are the chapters\b",
        r"\bchapter list\b",
        r"\bchapters list\b",
        r"\bchapter names\b",
    ]
    if any(re.search(pattern, q) for pattern in chapter_list_patterns):
        return "chapter_list"

    author_patterns = [
        r"\bwho wrote (?:the )?(?:book|student book|textbook)\b",
        r"\bwho is the author\b",
        r"\bauthor of (?:the )?(?:book|student book|textbook)\b",
    ]
    if any(re.search(pattern, q) for pattern in author_patterns):
        return "author"

    edition_patterns = [
        r"\bwhat year is (?:this )?(?:edition|book)\b",
        r"\bwhich year was (?:this )?(?:edition|book) published\b",
        r"\bedition year\b",
        r"\bpublication year\b",
    ]
    if any(re.search(pattern, q) for pattern in edition_patterns):
        return "edition_year"

    started_patterns = [
        r"\bwhen was climate academy started\b",
        r"\bwhen did climate academy start\b",
        r"\bwhen was the climate academy founded\b",
        r"\bclimate academy started\b",
        r"\bfounded in what year\b",
    ]
    if any(re.search(pattern, q) for pattern in started_patterns):
        return "climate_academy_started"

    return None


def is_chapter_summary_query(query: str) -> int | None:
    """
    Return a chapter number if the query asks for a chapter summary.
    Supports numeric and word-based chapter references.
    """
    q = _normalize_query(query)
    if not q:
        return None

    if not any(
        phrase in q
        for phrase in ["summary of chapter", "summarize chapter", "chapter summary", "summaries of chapter"]
    ):
        return None

    match = re.search(
        r"\bchapter\s+(?P<num>\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen)\b",
        q,
    )
    if not match:
        return None

    token = match.group("num")
    return int(token) if token.isdigit() else _WORD_NUMS.get(token)


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
    title = _DEFAULT_BOOK_METADATA["title"]
    chapters = _DEFAULT_BOOK_METADATA["chapters"]

    _BOOK_FACTS = {"title": title, "chapters": chapters}
    return _BOOK_FACTS


def build_fact_passages(query: str) -> list[dict]:
    q = _normalize_query(query)
    facts = get_book_facts()
    chapters = facts["chapters"]

    if not q:
        return []

    intent = detect_book_fact_intent(q)

    if intent == "book_title":
        return [{
            "document": f"The title of the book is {facts['title']}.",
            "source_type": "book",
            "section_number": "0.0",
            "section_title": "Book Title",
            "distance": 0.0,
            "chapter_number": 0,
            "chapter_title": facts["title"],
        }]

    if intent == "chapter_count":
        return [{
            "document": f"The book has {len(chapters)} chapters.",
            "source_type": "book",
            "section_number": "0.1",
            "section_title": "Chapter Count",
            "distance": 0.0,
            "chapter_number": 0,
            "chapter_title": facts["title"],
        }]

    if intent == "chapter_list":
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

    if intent == "author":
        return [{
            "document": f"The author of the book is {facts['author']}.",
            "source_type": "book",
            "section_number": "0.3",
            "section_title": "Author",
            "distance": 0.0,
            "chapter_number": 0,
            "chapter_title": facts["title"],
        }]

    if intent == "edition_year":
        return [{
            "document": f"The book edition year is {facts['edition_year']}.",
            "source_type": "book",
            "section_number": "0.4",
            "section_title": "Edition Year",
            "distance": 0.0,
            "chapter_number": 0,
            "chapter_title": facts["title"],
        }]

    if intent == "climate_academy_started":
        return [{
            "document": f"The Climate Academy was founded in {facts['climate_academy_started']}.",
            "source_type": "book",
            "section_number": "0.5",
            "section_title": "Climate Academy Founded",
            "distance": 0.0,
            "chapter_number": 0,
            "chapter_title": facts["title"],
        }]

    return []
