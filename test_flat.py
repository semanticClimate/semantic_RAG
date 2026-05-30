from bs4 import BeautifulSoup
from html_sectioning import load_html_file, find_book_root, _direct_child_tags, _bump_counters, _format_section_number, _normalize_whitespace, SectionRecord

soup = BeautifulSoup(load_html_file("input/climate_academy.html"), "html.parser")
root = find_book_root(soup)
records = []
counters = [0] * 6
current_rec = None
current_body_parts = []
hh = ["h1", "h2", "h3", "h4", "h5", "h6"]

for child in _direct_child_tags(root):
    if child.name == "section" and child.get("id") == "footnotes":
        continue
    if child.name in hh:
        if current_rec:
            body = _normalize_whitespace("\n".join(current_body_parts))
            if body:
                records.append(SectionRecord(
                    section_number=current_rec["number"],
                    title=current_rec["title"],
                    body=body,
                    level=current_rec["level"]
                ))
        level = int(child.name[1])
        _bump_counters(counters, level)
        number = _format_section_number(counters, level)
        title = child.get_text(separator=" ", strip=True)
        current_rec = {
            "number": number,
            "title": title,
            "level": level
        }
        current_body_parts = []
    else:
        text = child.get_text(separator=" ", strip=True)
        if text:
            current_body_parts.append(text)

if current_rec:
    body = _normalize_whitespace("\n".join(current_body_parts))
    if body:
        records.append(SectionRecord(
            section_number=current_rec["number"],
            title=current_rec["title"],
            body=body,
            level=current_rec["level"]
        ))

print(f"Parsed records: {len(records)}")
for r in records[:15]:
    print(f"- Section {r.section_number}: {r.title} ({len(r.body)} chars)")
