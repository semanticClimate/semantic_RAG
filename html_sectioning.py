"""
HTML book outline parsing and chunking for RAG.

Designed for flat HTML documents structured with heading tags (h1-h6)
as section boundaries, NOT nested <section> elements.

The parser walks all top-level elements in document order. Every time
a heading is encountered, a new section begins. All non-heading content
between two headings belongs to the most recent heading's section.

This matches the structure of climate_academy.html which uses:
  <h1> — chapter titles
  <h2> — Introduction / Main Text / Conclusion labels (skipped)
  <h3> — named subsections
  <h4> — named sub-subsections
  <p>, <ol>, <ul>, <table>, <blockquote>, <figure> — content elements
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from bs4 import BeautifulSoup, NavigableString, Tag

# Tags that signal a new section boundary
HEADING_TAGS = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}

# Maximum outline depth
MAX_OUTLINE_DEPTH = 6

# H1 IDs to skip entirely — table of contents and other non-content headings
SKIP_H1_IDS = {'contents', 'section', 'section-3', 'section-4', 'section-5'}

# Generic H2 labels that are structural dividers, not meaningful section titles.
# Text under these headings is kept but attributed to the parent H1 chapter.
SKIP_H2_TITLES = {
    'introduction', 'main text', 'conclusion', 'conclusions',
    'the bottom line', 'main text', 'overview'
}


@dataclass(frozen=True)
class SectionRecord:
    """One parsed section with its heading metadata and body text."""
    section_number: str
    title: str
    body: str
    level: int


def load_html_file(path: Path | str) -> str:
    p = Path(path)
    assert p.is_file(), f"HTML file not found: {p.resolve()}"
    return p.read_text(encoding="utf-8", errors="replace")


def _normalize_whitespace(text: str) -> str:
    """Clean up whitespace without destroying paragraph breaks."""
    text = text.replace("\u00a0", " ")          # non-breaking spaces
    text = re.sub(r"[ \t]+\n", "\n", text)      # trailing whitespace on lines
    text = re.sub(r"\n{3,}", "\n\n", text)      # more than 2 blank lines
    text = re.sub(r" {2,}", " ", text)          # multiple spaces
    return text.strip()


def _element_text(el: Tag) -> str:
    """Extract clean text from a content element."""
    return _normalize_whitespace(el.get_text(separator=" ", strip=True))


def _bump_counters(counters: List[int], level: int) -> str:
    """Increment counter at this level, reset all deeper levels, return dotted number."""
    idx = level - 1
    counters[idx] += 1
    for j in range(level, MAX_OUTLINE_DEPTH):
        counters[j] = 0
    return ".".join(str(counters[i]) for i in range(level))


def _should_skip_heading(tag: Tag) -> bool:
    """
    Return True for headings that should not start a new section.

    Skips:
    - Empty headings (alt-text IDs from images, etc.)
    - Table of contents H1
    - Generic structural H2 labels (Introduction, Main Text, Conclusion)
    """
    title = re.sub(r"\s+", " ", tag.get_text(strip=True)).strip()
    tag_id = tag.get("id", "")
    level = int(tag.name[1])

    if not title:
        return True

    if level == 1 and any(skip in tag_id for skip in SKIP_H1_IDS):
        return True

    if level == 2 and title.lower() in SKIP_H2_TITLES:
        return True

    return False


def parse_book_html(html: str) -> List[SectionRecord]:
    """
    Parse a flat heading-structured HTML document into SectionRecord objects.

    Walks all top-level elements in document order. Headings act as section
    boundary markers. Body content accumulates between headings.

    Returns a list of SectionRecord objects, one per meaningful heading,
    with decimal section numbers reflecting the heading hierarchy.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Collect all top-level elements — this book has no body/html wrapper
    all_elements = [
        el for el in soup.children
        if isinstance(el, Tag)
    ]

    # Skip footnotes section entirely
    footnote_ids = set()
    for el in all_elements:
        if el.name == 'section' and 'footnotes' in el.get('class', []):
            footnote_ids.add(id(el))

    counters = [0] * MAX_OUTLINE_DEPTH
    records: List[SectionRecord] = []

    current_title: Optional[str] = None
    current_level: Optional[int] = None
    current_number: Optional[str] = None
    current_body_parts: List[str] = []

    def flush():
        """Save the accumulated section if it has content."""
        if not current_title or not current_body_parts:
            return
        body = _normalize_whitespace("\n".join(
            part for part in current_body_parts if part.strip()
        ))
        if body:
            records.append(SectionRecord(
                section_number=current_number,
                title=current_title,
                body=body,
                level=current_level,
            ))

    for el in all_elements:
        # Skip footnote section
        if id(el) in footnote_ids:
            continue

        if el.name in HEADING_TAGS:
            if _should_skip_heading(el):
                # Structural heading — don't start new section,
                # but keep collecting body under current section
                continue

            # New meaningful heading — flush the previous section first
            flush()

            title = re.sub(r"\s+", " ", el.get_text(strip=True)).strip()
            level = int(el.name[1])
            number = _bump_counters(counters, level)

            current_title = title
            current_level = level
            current_number = number
            current_body_parts = []

        else:
            # Content element — collect text if we are inside a section
            if current_title is not None:
                text = _element_text(el)
                if text:
                    current_body_parts.append(text)

    # Don't forget the last section
    flush()

    return records


def word_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping word-count windows.

    chunk_size and overlap are in words, not tokens.
    Overlap ensures sentences at chunk boundaries appear in both adjacent chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    words = text.split()
    chunks: List[str] = []
    step = chunk_size - overlap
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i: i + chunk_size]))
        i += step
    return chunks


@dataclass(frozen=True)
class IndexedChunk:
    """One embedding unit: a chunk of text with its section metadata."""
    document: str           # formatted string sent to the embedder and LLM
    section_number: str     # e.g. "3.2.1"
    section_title: str      # e.g. "The Greenhouse Effect"
    chunk_index: int        # position within the section (0-based)


def records_to_indexed_chunks(
    records: Iterable[SectionRecord],
    chunk_size: int,
    chunk_overlap: int,
) -> List[IndexedChunk]:
    """
    Convert SectionRecord objects into IndexedChunk objects ready for embedding.

    Each chunk is prefixed with a section header in the format:
        [§ 3.2 — The Greenhouse Effect]
    This header is included in the embedded text so the embedding captures
    the section context, and is shown to the LLM as a citation reference.
    """
    out: List[IndexedChunk] = []
    for rec in records:
        chunks = word_chunks(rec.body, chunk_size, chunk_overlap)
        for idx, part in enumerate(chunks):
            header = f"[§ {rec.section_number}"
            if rec.title:
                header += f" — {rec.title}"
            header += "]"
            document = f"{header}\n{part}"
            out.append(IndexedChunk(
                document=document,
                section_number=rec.section_number,
                section_title=rec.title,
                chunk_index=idx,
            ))
    return out


def parse_html_path_to_chunks(
    path: Path | str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[IndexedChunk]:
    """
    Full pipeline: load HTML file → parse sections → produce indexed chunks.
    Called by ingest.py.
    """
    html = load_html_file(path)
    records = parse_book_html(html)
    return records_to_indexed_chunks(records, chunk_size, chunk_overlap)