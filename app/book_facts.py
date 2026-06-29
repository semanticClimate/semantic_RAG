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
        {"number": 1, "title": "The Absolute Basics", "summary": "This opening chapter points at the vital statistic for human life on our planet: the level of greenhouse gas concentration in the atmosphere. A deep understanding of the atmosphere, photosynthesis, radiative forcing, and the distinction between climate and weather is essential to understanding the crisis."},
        {"number": 2, "title": "Mass Extinction Events", "summary": "The passage explores past mass extinctions, such as the event that wiped out the dinosaurs and the Permian extinction. It emphasizes the fragility of life and that we are currently experiencing the sixth mass extinction due to human activities."},
        {"number": 3, "title": "Spaceship Earth", "summary": "If you were living in a small community on a spaceship and one astronaut started to use up all the supplies, what would the community do? This chapter uses this analogy to explain planetary boundaries and why we are exceeding Earth's carrying capacity for humanity."},
        {"number": 4, "title": "Where are we now?", "summary": "The world is rapidly approaching critical carbon budget limits, yet current national commitments (NDCs) fall far short of what is needed to meet climate goals. This chapter examines the gap between what is promised and what is required."},
        {"number": 5, "title": "The United Nations?", "summary": "This chapter critically examines the United Nations Framework Convention on Climate Change (UNFCCC) and climate negotiations. It explores how the UN processes work and why they have struggled to achieve binding agreements."},
        {"number": 6, "title": "Who is responsible?", "summary": "India, Bangladesh, and Pakistan were all severely exploited under British colonial rule, with land, resources, and labor extracted. This chapter examines historical responsibility for climate change and the concept of climate justice."},
        {"number": 7, "title": "The CUTx Percent Index", "summary": "The central chapter in the book. If the rest of the book is about climate literacy, this chapter is about understanding responsibility, capacity, and how to allocate climate action fairly among nations."},
        {"number": 8, "title": "Tipping Points – Physical", "summary": "This chapter emphasizes the enormous impact small actions can have on complex systems, using examples from nature and physics to explain why tipping points matter and why preventing them is critical."},
        {"number": 9, "title": "Paradigm Shifts", "summary": "This chapter explores how paradigm shifts—fundamental changes in the frameworks through which we interpret reality—have shaped human understanding and how new paradigms are essential to addressing climate change."},
        {"number": 10, "title": "Tipping Points – Social", "summary": "The conclusion highlights the urgency of the climate crisis, emphasizing the disconnect between scientific consensus and political action. It explores how social tipping points can accelerate systemic change."},
        {"number": 11, "title": "The Psychology of Climate Change (1)", "summary": "The next two chapters have a different rhythm. They contain a sequence of bite-sized reflections on the psychological aspects of engaging with climate reality—denial, anger, grief, and hope."},
        {"number": 12, "title": "The Psychology of Climate Change (2)", "summary": "Continuing the exploration of psychological aspects, this chapter examines emotional responses to climate science and how to develop resilience and agency in the face of existential challenges."},
        {"number": 13, "title": "The Paradox of Innovation", "summary": "This chapter explores the power of innovation—its capacity to transform societies and confront global challenges. It examines both the potential and limitations of technology-focused solutions to climate change."},
        {"number": 14, "title": "Climate Anxiety", "summary": "This chapter explores climate anxiety through the lens of literature, arguing that fiction uniquely captures the emotional reality of living through climate crisis. It validates climate anxiety as a rational response to scientific evidence."},
        {"number": 15, "title": "The Climate Academy (1) The Piraeus", "summary": "This chapter underscores the importance of a systems perspective in understanding the climate crisis. It discusses how interconnected global systems mean that local and global actions affect each other."},
        {"number": 16, "title": "The Climate Academy (2) The Academy", "summary": "Ultimately, systemic change may be complex, but it is achievable. The original Academy, located outside Athens, explored how to live justly and rationally. Today, the Climate Academy seeks to educate for this possibility."},
    ],
}


def _merge_book_metadata(facts: dict | None) -> dict:
    merged = {
        "title": _DEFAULT_BOOK_METADATA["title"],
        "author": _DEFAULT_BOOK_METADATA["author"],
        "edition_year": _DEFAULT_BOOK_METADATA["edition_year"],
        "license": _DEFAULT_BOOK_METADATA["license"],
        "climate_academy_started": _DEFAULT_BOOK_METADATA["climate_academy_started"],
        "chapters": _DEFAULT_BOOK_METADATA["chapters"],
    }
    if isinstance(facts, dict):
        merged.update({k: v for k, v in facts.items() if v is not None})
    return merged


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
        r"\b(give me|list|show|name|tell me)\b.*\bchapters?\b",
        r"\b(give me|list|show|name|tell me)\b.*\bchapter names?\b",
        r"\blist all chapters\b",
        r"\blist all chapter names\b",
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

    climate_academy_patterns = [
        r"\bwhat (?:is|does|are) climate academy\b",
        r"\btell me about climate academy\b",
        r"\bexplain climate academy\b",
        r"\bclimate academy overview\b",
        r"\bwhat does climate academy do\b",
        r"\bclimate academy\b",
    ]
    if any(re.search(pattern, q) for pattern in climate_academy_patterns):
        return "climate_academy_overview"

    climate_change_patterns = [
        r"\bwhat is climate change\b",
        r"\bdefine climate change\b",
        r"\bdefinition of climate change\b",
        r"\bexplain climate change\b",
        r"\bclimate change definition\b",
        r"\banthropogenic climate change\b",
    ]
    if any(re.search(pattern, q) for pattern in climate_change_patterns):
        return "climate_change_definition"

    return None


def is_chapter_summary_query(query: str) -> int | None:
    """
    Return a chapter number if the query asks for a chapter summary.
    Supports numeric and word-based chapter references.
    """
    q = _normalize_query(query)
    if not q:
        return None

    summary_keywords = [
        "summary of chapter", 
        "summarize chapter", 
        "chapter summary", 
        "summaries of chapter",
        "what is chapter",
        "tell me about chapter",
        "chapter about",
        "chapter on",
        "summary chapter",
    ]
    
    if not any(phrase in q for phrase in summary_keywords):
        return None

    match = re.search(
        r"\bchapter\s+(?P<num>\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen)\b",
        q,
    )
    if not match:
        # Try reverse pattern: "num chapter" or "num summary"
        match = re.search(
            r"(?P<num>\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen)\s+chapter",
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
        _BOOK_FACTS = _merge_book_metadata(_BOOK_FACTS)
        return _BOOK_FACTS

    html_path = Path(Config.SOURCE_HTML_PATH)
    if not html_path.is_file() and Config.SOURCE_HTML_PATHS:
        html_path = Path(Config.SOURCE_HTML_PATHS[0])

    html = html_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    _BOOK_FACTS = _merge_book_metadata(None)
    return _BOOK_FACTS


def build_fact_passages(query: str, intent: str | None = None) -> list[dict]:
    q = _normalize_query(query)
    facts = get_book_facts()
    chapters = facts["chapters"]

    if not q:
        return []

    chapter_num = is_chapter_summary_query(q)
    if chapter_num is not None:
        summary = get_chapter_summary(chapter_num)
        if summary:
            chapter_title = None
            for ch in chapters:
                if ch.get("number") == chapter_num:
                    chapter_title = ch.get("title")
                    break
            return [{
                "document": f"Chapter {chapter_num}: {chapter_title}\n\n{summary}",
                "source_type": "book",
                "section_number": f"{chapter_num}.0",
                "section_title": f"Chapter {chapter_num} Summary",
                "distance": 0.0,
                "chapter_number": chapter_num,
                "chapter_title": chapter_title or f"Chapter {chapter_num}",
            }]

    if is_full_book_summary_query(q):
        book_summary = get_full_book_summary()
        return [{
            "document": book_summary,
            "source_type": "book",
            "section_number": "0.6",
            "section_title": "Full Book Summary",
            "distance": 0.0,
            "chapter_number": 0,
            "chapter_title": facts.get("title", _DEFAULT_BOOK_METADATA["title"]),
        }]

    intent = intent or detect_book_fact_intent(q)

    if intent == "book_title":
        return [{
            "document": f"The title of the book is {facts.get('title', _DEFAULT_BOOK_METADATA['title'])}.",
            "source_type": "book",
            "section_number": "0.0",
            "section_title": "Book Title",
            "distance": 0.0,
            "chapter_number": 0,
            "chapter_title": facts.get("title", _DEFAULT_BOOK_METADATA["title"]),
        }]

    if intent == "chapter_count":
        return [{
            "document": f"The book has {len(chapters)} chapters.",
            "source_type": "book",
            "section_number": "0.1",
            "section_title": "Chapter Count",
            "distance": 0.0,
            "chapter_number": 0,
            "chapter_title": facts.get("title", _DEFAULT_BOOK_METADATA["title"]),
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
            "chapter_title": facts.get("title", _DEFAULT_BOOK_METADATA["title"]),
        }]

    if intent == "author":
        return [{
            "document": f"The author of the book is {facts.get('author', _DEFAULT_BOOK_METADATA['author'])}.",
            "source_type": "book",
            "section_number": "0.3",
            "section_title": "Author",
            "distance": 0.0,
            "chapter_number": 0,
            "chapter_title": facts.get("title", _DEFAULT_BOOK_METADATA["title"]),
        }]

    if intent == "edition_year":
        return [{
            "document": f"The book edition year is {facts.get('edition_year', _DEFAULT_BOOK_METADATA['edition_year'])}.",
            "source_type": "book",
            "section_number": "0.4",
            "section_title": "Edition Year",
            "distance": 0.0,
            "chapter_number": 0,
            "chapter_title": facts.get("title", _DEFAULT_BOOK_METADATA["title"]),
        }]

    if intent == "climate_academy_started":
        return [{
            "document": f"The Climate Academy was founded in {facts.get('climate_academy_started', _DEFAULT_BOOK_METADATA['climate_academy_started'])}.",
            "source_type": "book",
            "section_number": "0.5",
            "section_title": "Climate Academy Founded",
            "distance": 0.0,
            "chapter_number": 0,
            "chapter_title": facts.get("title", _DEFAULT_BOOK_METADATA["title"]),
        }]

    if intent == "climate_academy_overview":
        return [{
            "document": (
                "Climate Academy offers schools and students a radically new, systems-informed approach to climate education. "
                "Supported by 16 chapters of in-depth scientific and social analysis and a unique certification system, it teaches students "
                "to think about the climate crisis from a systems perspective and empowers them to act as true agents of change within their communities."
            ),
            "source_type": "book",
            "section_number": "0.6",
            "section_title": "Climate Academy Overview",
            "distance": 0.0,
            "chapter_number": 0,
            "chapter_title": facts.get("title", _DEFAULT_BOOK_METADATA["title"]),
        }]

    if intent == "climate_change_definition":
        return [{
            "document": (
                "Climate change refers to long-term shifts in Earth's climate system. Present-day climate change includes global warming, "
                "the ongoing increase in global average temperature, and its wider effects on Earth's climate system. The modern-day rise "
                "in global temperatures is driven by human activities, especially the burning of fossil fuels since the Industrial Revolution, "
                "along with deforestation and some agricultural and industrial practices that release greenhouse gases."
            ),
            "source_type": "encyclopedia",
            "section_number": "anthropogenic climate change",
            "section_title": "Anthropogenic Climate Change",
            "term": "anthropogenic climate change",
            "distance": 0.0,
            "chapter_number": 0,
            "chapter_title": "Encyclopedia",
        }]

    return []


def get_chapter_summary(chapter_number: int) -> str | None:
    """
    Return the summary for a specific chapter, or None if not found.
    """
    facts = get_book_facts()
    chapters = facts.get("chapters", [])
    
    for chapter in chapters:
        if chapter.get("number") == chapter_number:
            return chapter.get("summary")
    
    return None


def get_full_book_summary() -> str:
    """
    Combine all chapter summaries into a comprehensive book summary.
    """
    facts = get_book_facts()
    chapters = facts.get("chapters", [])
    
    summary_lines = [f"**{facts.get('title', _DEFAULT_BOOK_METADATA['title'])}** by {facts.get('author', _DEFAULT_BOOK_METADATA['author'])}"]
    summary_lines.append("")
    
    for chapter in chapters:
        ch_num = chapter.get("number")
        ch_title = chapter.get("title")
        ch_summary = chapter.get("summary")
        
        if ch_summary:
            summary_lines.append(f"**Chapter {ch_num}: {ch_title}**")
            summary_lines.append(ch_summary)
            summary_lines.append("")
    
    return "\n".join(summary_lines)


def is_full_book_summary_query(query: str) -> bool:
    """
    Detect if the query is asking for a full book summary.
    """
    q = _normalize_query(query)
    if not q:
        return False
    
    patterns = [
        r"\b(give me|provide|show|tell me)\b.*\b(summary|overview)\b",
        r"\bsummary of the book\b",
        r"\bbook summary\b",
        r"\bfull summary\b",
        r"\bcomplete summary\b",
        r"\bentire book summary\b",
        r"\bsummarize the (entire |full |whole |complete )?book\b",
        r"\bwhat is this book about\b",
        r"\btell me about the (entire |full |whole |complete )?book\b",
        r"\b(give me|tell me|show me) (an|a) (brief |comprehensive |full |complete |detailed )?(overview|summary|outline) (of |about )?the (entire |full |whole |complete )?book\b",
    ]
    
    return any(re.search(pattern, q) for pattern in patterns)
