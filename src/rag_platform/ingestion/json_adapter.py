import json
import html
import re
from typing import Any, Dict, List, Iterable


def _clean_text(s: str) -> str:
    # decode HTML entities like &amp; &nbsp;
    s = html.unescape(s)
    # replace non-breaking spaces
    s = s.replace("\u00a0", " ")
    # collapse whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def record_to_text_and_metadata(rec: Dict[str, Any]) -> Dict[str, Any]:
    blocks = []
    for item in (rec.get("ContentList") or []):
        if isinstance(item, dict):
            c = item.get("Content")
            if isinstance(c, str) and c.strip():
                blocks.append(c)

    # optional: include top-level Content if not empty
    top_content = rec.get("Content")
    if isinstance(top_content, str) and top_content.strip():
        blocks.append(top_content)

    text = _clean_text("\n\n".join(blocks))

    metadata = {
        "uri": rec.get("URI"),
        "title": rec.get("Title"),
        "language": rec.get("Language"),
        "search_categories": rec.get("SearchCategoryList") or [],
        "file_uri_list": rec.get("File_URI_List") or [],
    }

    # keep only JSON-serializable metadata
    return {"text": text, "metadata": metadata}


def load_json_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Your file might be a list at the top level
    if isinstance(data, list):
        records = data
    # Or wrapped in some object
    elif isinstance(data, dict) and "records" in data and isinstance(data["records"], list):
        records = data["records"]
    else:
        raise ValueError("Unsupported JSON structure: expected a list of records")

    out = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        out.append(record_to_text_and_metadata(rec))
    return out