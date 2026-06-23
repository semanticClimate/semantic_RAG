"""
HTML parsing and chunking for two distinct source types:

  BOOK  (climate_academy.html)
  ─────────────────────────────────────────────────────────────────────────
  Structure: completely flat HTML — all elements are direct children of the
  document root with NO <section> wrappers.

  Chapter boundaries are detected by a <p> tag whose text matches
  "Chapter One" … "Chapter Sixteen", followed within a few siblings by
  an <h1> giving the chapter title.

  Within a chapter the hierarchy is:
    <h1>  Chapter title           (level 1)
    <h2>  Section heading         (level 2 – e.g. "Introduction", "Main Text")
    <h4>  Sub-topic heading       (level 3 – named h4 in the source)
    <p>   Body paragraphs

  We collect every paragraph that falls between headings and attach it to
  the nearest enclosing heading, producing one SectionRecord per logical
  block of text.

  ENCYCLOPEDIA  (climate_filtered.html)
  ─────────────────────────────────────────────────────────────────────────
  Structure: proper HTML with a <div id="entries"> container.  Each entry
  is a <div class="entry" data-term="…"> containing:
    .term          – the term name
    .synonyms      – optional synonyms line
    .description   – one or more <p> tags with the definition

  Each entry becomes exactly ONE SectionRecord (never split across records).
  Long entries are chunked at the chunk stage, not here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from bs4 import BeautifulSoup, Tag

# ─────────────────────────────────────────────────────────────────────────────
#  Shared data structures
# ─────────────────────────────────────────────────────────────────────────────

HEADING_TAGS = tuple(f"h{i}" for i in range(1, 7))

CHAPTER_WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16,
}


@dataclass(frozen=True)
class SectionRecord:
    """One logical chunk of source text ready for embedding."""
    section_number: str   # e.g. "3.2" for book, "carbon cycle" for encyclopedia
    title: str            # heading / term name
    body: str             # plain text body
    level: int            # 1–3 for book sections; 1 for encyclopedia entries
    source_type: str      # "book" or "encyclopedia"
    chapter_number: int   # 1-16 for book; 0 for encyclopedia
    chapter_title: str    # chapter name for book; "" for encyclopedia


# ─────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_html_file(path: Path | str) -> str:
    p = Path(path)
    assert p.is_file(), f"HTML file not found: {p.resolve()}"
    return p.read_text(encoding="utf-8", errors="replace")


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _direct_child_tags(tag: Tag) -> List[Tag]:
    return [c for c in tag.children if isinstance(c, Tag)]


# ─────────────────────────────────────────────────────────────────────────────
#  BOOK PARSER
# ─────────────────────────────────────────────────────────────────────────────

def _is_chapter_marker(tag: Tag) -> Optional[int]:
    """
    Return chapter number (1-16) if this <p> tag is 'Chapter One' … 'Chapter Sixteen'.
    Otherwise return None.
    """
    if tag.name != "p":
        return None
    txt = tag.get_text(strip=True).lower()
    m = re.match(r"^chapter\s+(\w+)$", txt)
    if m:
        return CHAPTER_WORD_TO_NUM.get(m.group(1))
    return None


def _tag_text(tag: Tag) -> str:
    """Plain text of a tag, excluding image alt texts."""
    # Remove img tags before getting text to skip alt text noise
    import copy
    clone = copy.copy(tag)
    return _normalize_whitespace(tag.get_text(separator="\n", strip=True))


def parse_book_html(html: str) -> List[SectionRecord]:
    """
    Parse climate_academy.html into SectionRecords.

    The document is completely flat — no <section> nesting.
    We scan top-level siblings and group them into chapters, then into
    sections within each chapter.
    """
    soup = BeautifulSoup(html, "html.parser")
    all_tags: List[Tag] = [c for c in soup.children if isinstance(c, Tag)]

    # ── Step 1: locate chapter boundaries ──────────────────────────────────
    # A chapter starts at the <p>Chapter X</p> marker tag.
    # We find the index of the following <h1> as the actual chapter title tag.

    chapter_spans: List[Tuple[int, int, int, str]] = []  # (start_idx, end_idx, ch_num, ch_title)

    chapter_marker_positions: List[Tuple[int, int]] = []  # (marker_idx, chapter_number)
    for i, tag in enumerate(all_tags):
        ch_num = _is_chapter_marker(tag)
        if ch_num is not None:
            chapter_marker_positions.append((i, ch_num))

    for pos, (marker_idx, ch_num) in enumerate(chapter_marker_positions):
        # find the h1 title within the next 5 siblings
        ch_title = ""
        ch_h1_idx = marker_idx
        for j in range(marker_idx, min(marker_idx + 5, len(all_tags))):
            if all_tags[j].name == "h1":
                raw = all_tags[j].get_text(separator=" ", strip=True)
                ch_title = re.sub(r"\s+", " ", raw).strip()
                ch_h1_idx = j
                break

        # end of this chapter = start of next chapter marker (or end of doc)
        if pos + 1 < len(chapter_marker_positions):
            end_idx = chapter_marker_positions[pos + 1][0]
        else:
            end_idx = len(all_tags)

        chapter_spans.append((ch_h1_idx, end_idx, ch_num, ch_title))

    # ── Step 2: parse each chapter into sections ───────────────────────────
    records: List[SectionRecord] = []

    for (start_idx, end_idx, ch_num, ch_title) in chapter_spans:
        chapter_tags = all_tags[start_idx:end_idx]
        records.extend(
            _parse_chapter_sections(chapter_tags, ch_num, ch_title)
        )

    # ── Step 3: parse the Introduction (before Chapter One) ────────────────
    if chapter_marker_positions:
        intro_end = chapter_marker_positions[0][0]
        intro_tags = all_tags[:intro_end]
        intro_records = _parse_introduction(intro_tags)
        records = intro_records + records

    return records


def _parse_introduction(tags: List[Tag]) -> List[SectionRecord]:
    """
    Parse the preamble / introduction section before Chapter One.
    Treat the whole block as chapter 0, section 1.
    """
    # Collect all text after the h1#introduction heading
    collecting = False
    body_parts: List[str] = []
    intro_title = "Introduction"

    for tag in tags:
        if tag.name == "h1" and tag.get("id", "") == "introduction":
            collecting = True
            t = tag.get_text(separator=" ", strip=True)
            if t:
                intro_title = t
            continue
        if collecting:
            text = _normalize_whitespace(tag.get_text(separator="\n", strip=True))
            if text:
                body_parts.append(text)

    body = _normalize_whitespace("\n\n".join(body_parts))
    if not body:
        return []

    return [SectionRecord(
        section_number="0.1",
        title=intro_title,
        body=body,
        level=1,
        source_type="book",
        chapter_number=0,
        chapter_title="Introduction",
    )]


def _parse_chapter_sections(chapter_tags: List[Tag], ch_num: int, ch_title: str) -> List[SectionRecord]:
    """
    Parse a single chapter's flat tag list into SectionRecords.

    Hierarchy recognised:
      h1  → chapter heading (level 1)  — starts the chapter, produces one record
      h2  → section heading (level 2)  — e.g. "Introduction", "Main Text"
      h4  → sub-topic heading (level 3)
      p / table / ol / ul / blockquote → body content

    The chapter summary table (immediately after the h1) is included in the
    chapter-level record.
    """
    records: List[SectionRecord] = []

    # State machine: track current heading context
    current_level: int = 1
    current_title: str = ch_title
    current_section_num: str = str(ch_num)
    current_body_parts: List[str] = []

    # section counters: [h2_count, h4_count]
    h2_count = 0
    h4_count = 0

    def flush(level: int, title: str, section_num: str, body_parts: List[str]):
        body = _normalize_whitespace("\n\n".join(body_parts))
        if body:
            records.append(SectionRecord(
                section_number=section_num,
                title=title,
                body=body,
                level=level,
                source_type="book",
                chapter_number=ch_num,
                chapter_title=ch_title,
            ))

    for tag in chapter_tags:
        if tag.name == "h1":
            # Chapter heading — reset everything, start level-1 record
            flush(current_level, current_title, current_section_num, current_body_parts)
            current_level = 1
            current_title = re.sub(r"\s+", " ", tag.get_text(separator=" ", strip=True)).strip() or ch_title
            current_section_num = str(ch_num)
            current_body_parts = []
            h2_count = 0
            h4_count = 0

        elif tag.name == "h2":
            flush(current_level, current_title, current_section_num, current_body_parts)
            h2_count += 1
            h4_count = 0
            current_level = 2
            current_title = tag.get_text(separator=" ", strip=True)
            current_section_num = f"{ch_num}.{h2_count}"
            current_body_parts = []

        elif tag.name == "h4":
            flush(current_level, current_title, current_section_num, current_body_parts)
            h4_count += 1
            current_level = 3
            current_title = tag.get_text(separator=" ", strip=True)
            current_section_num = f"{ch_num}.{h2_count}.{h4_count}"
            current_body_parts = []

        elif tag.name in ("h3", "h5", "h6"):
            # Treat like h4 (sub-topic)
            flush(current_level, current_title, current_section_num, current_body_parts)
            h4_count += 1
            current_level = 3
            current_title = tag.get_text(separator=" ", strip=True)
            current_section_num = f"{ch_num}.{h2_count}.{h4_count}"
            current_body_parts = []

        else:
            # Body content: p, table, ol, ul, blockquote, div, figure, etc.
            text = _normalize_whitespace(tag.get_text(separator="\n", strip=True))
            if text:
                current_body_parts.append(text)

    # flush the last section
    flush(current_level, current_title, current_section_num, current_body_parts)

    return records


# ─────────────────────────────────────────────────────────────────────────────
#  ENCYCLOPEDIA PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_encyclopedia_html(html: str) -> List[SectionRecord]:
    """
    Parse climate_filtered.html into SectionRecords.

    Each <div class="entry" data-term="…"> becomes one record.
    The term name is the title; the description text is the body.
    Synonyms (when present) are prepended to the body so they are
    searchable.
    """
    soup = BeautifulSoup(html, "html.parser")
    entries = soup.select("div.entry")
    records: List[SectionRecord] = []

    for entry in entries:
        term_tag = entry.select_one(".term")
        syn_tag  = entry.select_one(".synonyms")
        desc_tag = entry.select_one(".description")

        if not term_tag:
            continue

        term = _normalize_whitespace(term_tag.get_text(separator=" ", strip=True))
        if not term:
            continue

        body_parts: List[str] = []

        if syn_tag:
            syn_text = _normalize_whitespace(syn_tag.get_text(separator=" ", strip=True))
            if syn_text:
                body_parts.append(f"Also known as: {syn_text}")

        if desc_tag:
            desc_text = _normalize_whitespace(desc_tag.get_text(separator="\n", strip=True))
            if desc_text:
                body_parts.append(desc_text)

        body = _normalize_whitespace("\n\n".join(body_parts))
        if not body:
            continue

        records.append(SectionRecord(
            section_number=term.lower(),   # term slug used as section id
            title=term,
            body=body,
            level=1,
            source_type="encyclopedia",
            chapter_number=0,
            chapter_title="",
        ))

    return records


# ─────────────────────────────────────────────────────────────────────────────
#  Auto-detect source type
# ─────────────────────────────────────────────────────────────────────────────

def detect_source_type(html: str) -> str:
    """
    Return 'encyclopedia' if the HTML contains the encyclopedia entry structure,
    'book' otherwise.
    """
    # Fast check: encyclopedia has a div#entries with div.entry children
    if 'class="entry"' in html and 'data-term=' in html:
        return "encyclopedia"
    return "book"


def parse_html(html: str) -> List[SectionRecord]:
    """Parse any supported HTML, auto-detecting the source type."""
    source_type = detect_source_type(html)
    if source_type == "encyclopedia":
        return parse_encyclopedia_html(html)
    return parse_book_html(html)


# ─────────────────────────────────────────────────────────────────────────────
#  Chunking
# ─────────────────────────────────────────────────────────────────────────────

def word_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")
    chunks: List[str] = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i: i + chunk_size]))
        i += chunk_size - overlap
    return chunks


def _sentence_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def encyclopedia_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Sentence-aware chunking: tries to keep sentences intact.
    Falls back to word_chunks for very long sentences.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paragraphs:
        return []

    chunks: List[str] = []
    step = chunk_size - overlap
    tail_words: List[str] = []

    for para in paragraphs:
        sentences = _sentence_split(para) or [para]
        cur_words: List[str] = tail_words.copy()

        for sent in sentences:
            sent_words = sent.split()
            if len(sent_words) > chunk_size:
                if cur_words:
                    chunks.append(" ".join(cur_words))
                    cur_words = []
                long_parts = word_chunks(sent, chunk_size, overlap)
                chunks.extend(long_parts[:-1])
                cur_words = long_parts[-1].split() if long_parts else []
                continue
            if len(cur_words) + len(sent_words) <= chunk_size:
                cur_words.extend(sent_words)
            else:
                if cur_words:
                    chunks.append(" ".join(cur_words))
                    cur_words = cur_words[-overlap:] if overlap > 0 else []
                cur_words.extend(sent_words)
                if len(cur_words) > chunk_size:
                    chunks.append(" ".join(cur_words[:chunk_size]))
                    cur_words = cur_words[step:]

        if cur_words:
            chunks.append(" ".join(cur_words))
            tail_words = cur_words[-overlap:] if overlap > 0 else []
        else:
            tail_words = []

    return [c for c in chunks if c.strip()]


def book_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Book-friendly chunking that prefers paragraph boundaries.

    Books in this project are already sectioned before chunking, so the goal
    here is to preserve the local flow of each section instead of slicing it
    into rigid word windows. Very short sections stay intact; longer ones are
    grouped by paragraphs with a small overlap for continuity.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    words = text.split()
    if len(words) <= chunk_size:
        return [text.strip()] if text.strip() else []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paragraphs:
        return word_chunks(text, chunk_size, overlap)

    chunks: List[str] = []
    step = chunk_size - overlap
    cur_parts: List[str] = []
    cur_count = 0

    def flush_current():
        nonlocal cur_parts, cur_count
        if cur_parts:
            chunk = _normalize_whitespace("\n\n".join(cur_parts))
            if chunk:
                chunks.append(chunk)

    for para in paragraphs:
        para_words = para.split()
        para_count = len(para_words)

        if para_count > chunk_size:
            flush_current()
            chunks.extend(word_chunks(para, chunk_size, overlap))
            # Start next chunk with overlap from the last word_chunk
            if overlap > 0 and chunks:
                last_words = chunks[-1].split()[-overlap:]
                cur_parts = [" ".join(last_words)] if last_words else []
                cur_count = len(last_words)
            else:
                cur_parts = []
                cur_count = 0
            continue

        if cur_count and cur_count + para_count > chunk_size:
            flush_current()
            # Start next chunk with overlap from the flushed chunk
            if overlap > 0 and chunks:
                last_words = chunks[-1].split()[-overlap:]
                cur_parts = [" ".join(last_words), para] if last_words else [para]
                cur_count = len(last_words) + para_count
            else:
                cur_parts = [para]
                cur_count = para_count
        else:
            cur_parts.append(para)
            cur_count += para_count

    flush_current()

    return [c for c in chunks if c.strip()]


# ─────────────────────────────────────────────────────────────────────────────
#  IndexedChunk — final unit stored in ChromaDB
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IndexedChunk:
    document: str          # text sent to the embedder and stored as document
    section_number: str
    section_title: str
    chunk_index: int
    source_type: str       # "book" | "encyclopedia"
    chapter_number: int    # 1-16 for book; 0 for encyclopedia
    chapter_title: str     # chapter name for book; "" for encyclopedia


def records_to_indexed_chunks(
    records: Iterable[SectionRecord],
    chunk_size: int,
    chunk_overlap: int,
    chunk_mode: str = "default",
) -> List[IndexedChunk]:
    """
    Convert SectionRecords to IndexedChunks.

    chunk_mode:
      "default"      – word-level sliding window (fast, good for book prose)
      "encyclopedia" – sentence-aware chunking (better for definition text)
      "auto"         – uses encyclopedia chunking for encyclopedia records,
                       word chunking for book records  ← RECOMMENDED
    """
    out: List[IndexedChunk] = []

    for rec in records:
        # Choose chunking strategy
        if chunk_mode == "auto":
            use_enc_chunking = (rec.source_type == "encyclopedia")
            use_book_chunking = (rec.source_type == "book")
        else:
            use_enc_chunking = (chunk_mode == "encyclopedia")
            use_book_chunking = False

        parts = (
            encyclopedia_chunks(rec.body, chunk_size, chunk_overlap)
            if use_enc_chunking
            else book_chunks(rec.body, chunk_size, chunk_overlap)
            if use_book_chunking
            else word_chunks(rec.body, chunk_size, chunk_overlap)
        )

        for idx, part in enumerate(parts):
            # Build a context header so the LLM always knows where this chunk is from
            if rec.source_type == "book":
                header = f"[Book | Ch.{rec.chapter_number} {rec.chapter_title} | {rec.section_number} {rec.title}]"
            else:
                header = f"[Encyclopedia | {rec.title}]"

            doc = f"{header}\n{part}"

            out.append(IndexedChunk(
                document=doc,
                section_number=rec.section_number,
                section_title=rec.title,
                chunk_index=idx,
                source_type=rec.source_type,
                chapter_number=rec.chapter_number,
                chapter_title=rec.chapter_title,
            ))

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_html_path_to_chunks(
    path: Path | str,
    chunk_size: int,
    chunk_overlap: int,
    chunk_mode: str = "auto",
) -> List[IndexedChunk]:
    html = load_html_file(path)
    records = parse_html(html)
    return records_to_indexed_chunks(records, chunk_size, chunk_overlap, chunk_mode=chunk_mode)


# ─────────────────────────────────────────────────────────────────────────────
#  Legacy shim — keeps old callers working
# ─────────────────────────────────────────────────────────────────────────────

def parse_book_html_legacy(html: str) -> List["SectionRecord"]:
    """Backward-compat alias."""
    return parse_html(html)
