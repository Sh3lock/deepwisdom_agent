import json
from pathlib import Path

from langchain_core.tools import tool


def _iter_docs(root: Path):
    exts = {".txt", ".md", ".markdown"}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def _make_snippet(text: str, idx: int, window: int = 80) -> str:
    start = max(0, idx - window)
    end = min(len(text), idx + window)
    snippet = text[start:end].replace("\n", " ").replace("\r", " ")
    return " ".join(snippet.split())


@tool
def local_search(query: str, top_k: int = 3) -> str:
    """Search docs/ for keyword matches and return top_k snippets with sources."""
    docs_dir = Path(__file__).resolve().parent / "docs"
    if not docs_dir.exists():
        return json.dumps([], ensure_ascii=True)

    query = (query or "").strip()
    if not query:
        return json.dumps([], ensure_ascii=True)

    tokens = [t for t in query.lower().split() if t]
    if not tokens:
        tokens = [query.lower()]

    results = []
    for path in _iter_docs(docs_dir):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lower = text.lower()
        score = sum(lower.count(t) for t in tokens)
        if score <= 0:
            continue
        first_idx = min((lower.find(t) for t in tokens if t in lower), default=0)
        snippet = _make_snippet(text, first_idx)
        results.append(
            {
                "source": str(path.relative_to(docs_dir)),
                "snippet": snippet,
                "score": score,
            }
        )

    results.sort(key=lambda r: r["score"], reverse=True)
    trimmed = [
        {"source": r["source"], "snippet": r["snippet"]}
        for r in results[: max(1, top_k)]
    ]
    return json.dumps(trimmed, ensure_ascii=True)