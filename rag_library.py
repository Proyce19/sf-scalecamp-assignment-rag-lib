from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import difflib

# -------------------------------
# 0) DEFAULT DATASET CONTENT (tiny)
# -------------------------------
DEFAULT_BOOKS = [
    {"title": "The Hitchhiker's Guide to the Galaxy", "author": "Douglas Adams",
     "summary": "A comedic science fiction adventure following Arthur Dent after Earth is destroyed to make way for a hyperspace bypass.",
     "tags": ["sci-fi", "comedy", "space", "earth destroyed", "arthur dent"]},
    {"title": "Pride and Prejudice", "author": "Jane Austen",
     "summary": "Elizabeth Bennet navigates love, class, and misunderstandings with the proud Mr. Darcy.",
     "tags": ["romance", "regency", "manners", "society", "class", "marriage"]},
    {"title": "1984", "author": "George Orwell",
     "summary": "A dystopian tale where Winston Smith resists a totalitarian regime that rewrites truth.",
     "tags": ["dystopia", "totalitarian", "surveillance", "rebellion", "big brother"]},
    {"title": "To Kill a Mockingbird", "author": "Harper Lee",
     "summary": "Through Scout Finch's eyes, the novel explores racial injustice and moral growth in the American South.",
     "tags": ["racism", "justice", "coming-of-age", "south", "courtroom"]},
    {"title": "The Hobbit", "author": "J.R.R. Tolkien",
     "summary": "Bilbo Baggins joins dwarves on a quest to reclaim their homeland from a dragon.",
     "tags": ["fantasy", "dragon", "dwarves", "quest", "bilbo"]},
    {"title": "The Great Gatsby", "author": "F. Scott Fitzgerald",
     "summary": "Nick Carraway tells of the enigmatic millionaire Jay Gatsby and his tragic dream.",
     "tags": ["wealth", "millionaire", "fortune", "american dream", "jazz age", "romance"]},
    {"title": "Moby-Dick", "author": "Herman Melville",
     "summary": "Captain Ahab obsessively hunts the white whale, Moby Dick.",
     "tags": ["whale", "sea", "revenge", "obsession", "ship"]},
    {"title": "Harry Potter and the Chamber of Secrets", "author": "J.K. Rowling",
     "summary": "Harry faces a monster hidden within Hogwarts.",
     "tags": ["fantasy", "harry potter", "hogwarts", "monster", "mystery", "magic"]},
    {"title": "Harry Potter and the Sorcerer's Stone", "author": "J.K. Rowling",
     "summary": "A boy discovers he is a wizard and attends Hogwarts School.",
     "tags": ["fantasy", "harry potter", "hogwarts", "magic", "wizard", "friendship"]},
    {"title": "Frankenstein", "author": "Mary Shelley",
     "summary": "Victor Frankenstein creates a monster with tragic consequences.",
     "tags": ["gothic", "science", "creation", "monster", "tragedy", "horror"]},
]


# -------------------------------
# 1) DATA LOADING
# -------------------------------
def ensure_dataset(path: str) -> None:
    """Create a default dataset JSON if it doesn't exist."""
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_BOOKS, f, indent=2, ensure_ascii=False)


def load_books(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a list of {title, author, summary, [tags]} objects")
    cleaned = []
    for i, item in enumerate(data):
        if not all(k in item for k in ("title", "author", "summary")):
            raise ValueError(f"Entry {i} missing one of required keys: title/author/summary")
        tags = item.get("tags", []) or []
        if not isinstance(tags, list):
            raise ValueError(f"Entry {i} has non-list 'tags'")
        cleaned.append({
            "title": str(item.get("title", "")),
            "author": str(item.get("author", "")),
            "summary": str(item.get("summary", "")),
            "tags": [str(t) for t in tags],
        })
    return cleaned


# -------------------------------
# 2) RETRIEVER (keywords + synonyms + fuzzy + co-occurrence bonus)
# -------------------------------
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "is", "are", "to", "of", "in", "on", "for", "with",
    "as", "at", "by", "from", "this", "that", "it", "its", "their", "his", "her", "be", "into",
    "about", "after", "where", "who", "whom", "which", "while", "until", "than", "then", "so"
}
_tokenizer = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _tokenizer.findall(text or "")]


def keywords(text: str) -> List[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


# Synonym map to help vague queries
SYNONYMS = {
    "money": ["wealth", "rich", "riches", "millionaire", "fortune", "affluent"],
    "fortune": ["wealth", "rich", "riches", "millionaire", "affluent"],
    "wealth": ["fortune", "rich", "riches", "millionaire", "affluent"],
    "rich": ["wealth", "fortune", "riches", "millionaire", "affluent"],
    "riches": ["wealth", "fortune", "rich", "millionaire", "affluent"],
    "millionaire": ["wealth", "fortune", "rich", "riches", "affluent"],
    "affluent": ["wealth", "fortune", "rich", "riches", "millionaire"],
    "guy": ["man", "boy", "gentleman", "person", "teen", "teenager"],
    "girl": ["woman", "lady", "person"],
    "political": ["politics", "power", "intrigue", "factions"],
    "desert": ["sand", "arid", "arrakis"],
    "dragon": ["smaug"],
    "mars": ["martian"],
    "space": ["planet", "astronaut"],
    "dystopia": ["dystopian", "totalitarian"],
    "monster": ["beast", "creature"],
    "hogwarts": ["school", "castle"]
}


def expand_query_terms(q_terms: List[str]) -> List[str]:
    out = set(q_terms)
    for t in list(out):
        for s in SYNONYMS.get(t, []):
            out.add(s)
    return list(out)


@dataclass
class BookKw:
    book: Dict[str, str]
    kw_title: set
    kw_author: set
    kw_summary: set
    kw_tags: set


def precompute_keywords(books: List[Dict[str, str]]) -> List[BookKw]:
    out: List[BookKw] = []
    for b in books:
        tag_text = " ".join(b.get("tags", []) or [])
        out.append(BookKw(
            book=b,
            kw_title=set(keywords(b.get("title", ""))),
            kw_author=set(keywords(b.get("author", ""))),
            kw_summary=set(keywords(b.get("summary", ""))),
            kw_tags=set(keywords(tag_text)),
        ))
    return out


# --- Improved scoring with co-occurrence and pair boosts ---
PAIR_BONUSES = [
    ("monster", "hogwarts"),
    ("desert", "spice"),
    ("american", "dream"),
    ("ring", "middle-earth"),
    ("hobbit", "dragon"),
    ("totalitarian", "surveillance"),
    ("racial", "justice"),
    ("whale", "sea"),
    ("revenge", "ghost"),
    ("tragedy", "love"),
    ("wizard", "hogwarts"),
    ("wizard", "magic"),
    ("knight", "windmills"),
    ("marlin", "fisherman"),
    ("wwii", "nazi"),
    ("post-apocalyptic", "survival"),
    ("mars", "astronaut"),
    ("prophecy", "arrakis"),
    ("revenge", "monster"),
    ("love", "class"),
]


def score_query_to_book(q_terms: List[str], item: BookKw) -> float:
    if not q_terms:
        return 0.0

    # Base fielded score
    score = 4.5 * sum(1 for t in q_terms if t in item.kw_tags)
    score += 2.0 * sum(1 for t in q_terms if t in item.kw_title)
    score += 1.5 * sum(1 for t in q_terms if t in item.kw_author)
    score += 1.0 * sum(1 for t in q_terms if t in item.kw_summary)

    # Co-occurrence bonuses (more than one query term in same field)
    def bonus(field_set):
        return 0.75 if len(field_set.intersection(q_terms)) >= 2 else 0.0

    score += bonus(item.kw_tags)
    score += 0.5 * bonus(item.kw_title)
    score += 0.25 * bonus(item.kw_summary)

    # Pair bonuses if both terms appear anywhere in title/summary/tags
    hay = item.kw_title | item.kw_summary | item.kw_tags
    for a, b in PAIR_BONUSES:
        if a in hay and b in hay:
            score += 2.75

    # Normalize by query length to avoid bias toward longer queries
    return score / (len(q_terms) ** 0.5)


def retrieve(query: str, index: List[BookKw], k: int = 3, adaptive: bool = False) -> List[Tuple[Dict[str, str], float]]:
    """Retrieve top-k books for a query with synonyms + fuzzy fallback.
       Anti-position-bias: return results ASC by score so BEST is LAST in the context.
    """
    q_base = keywords(query)
    q_terms = expand_query_terms(q_base)

    scored = [(it.book, score_query_to_book(q_terms, it)) for it in index]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Adaptive top-k: if the top margin is large, use k=1
    if adaptive and len(scored) > 1:
        top, second = scored[0][1], scored[1][1]
        if top > 0 and (top - second) / (top + 1e-9) >= 0.30:
            k = 1

    top = [pair for pair in scored[:k] if pair[1] > 0]

    # Fallback: try loose fuzzy if nothing matched
    if not top and q_terms:
        def loose_hit(book: Dict[str, str]) -> int:
            hay = (
                    book.get("title", "") + " " +
                    book.get("author", "") + " " +
                    book.get("summary", "") + " " +
                    " ".join(book.get("tags", []))
            ).lower()
            hay_words = set(re.findall(r"[a-z0-9]+", hay))
            hits = 0
            for t in q_terms:
                if t in hay:
                    hits += 1
                    continue
                for w in hay_words:
                    if len(t) > 3 and difflib.SequenceMatcher(None, t, w).ratio() >= 0.85:
                        hits += 1
                        break
            return hits

        scored_loose = [(it.book, loose_hit(it.book)) for it in index]
        scored_loose.sort(key=lambda x: x[1], reverse=True)
        if scored_loose and scored_loose[0][1] > 0:
            top = scored_loose[:1]

    # IMPORTANT: reverse so BEST is LAST (counter position bias)
    top_sorted_for_llm = sorted(top, key=lambda x: x[1])  # ascending; best at the end
    return top_sorted_for_llm


# -------------------------------
# 3) GENERATOR (Transformers)
# -------------------------------
IDENTIFY_PROMPT_TEMPLATE = (
    "You are a library assistant. Use ONLY the context snippets.\n"
    "Pick the SINGLE snippet with the HIGHEST score and answer with its exact book title + [#].\n"
    "If nothing answers, say: \"I don't know based on the catalog.\"\n\n"
    "Question:\n{question}\n\n"
    "Context (each has a score; higher = more relevant):\n{context}\n\n"
    "Answer (Title [#]):\n"
)

EXPLAIN_PROMPT_TEMPLATE = (
    "You are a library assistant. Use ONLY the context snippets.\n"
    "Write ONE concise sentence that answers the question and include the title + [#] of the single MOST relevant snippet.\n"
    "If the snippets don't contain the answer, say: \"I don't know based on the catalog.\"\n\n"
    "Question:\n{question}\n\n"
    "Context (each has a score; higher = more relevant):\n{context}\n\n"
    "Answer (one sentence, include title + [#]):\n"
)


def build_context(snippets: List[Tuple[Dict[str, str], float]]) -> str:
    lines = []
    best_i = None
    best_s = -1e9
    for i, (b, s) in enumerate(snippets, start=1):
        if s > best_s:
            best_s, best_i = s, i
        tags = b.get("tags") or []
        tag_str = f"\nTags: {', '.join(tags)}" if tags else ""
        lines.append(
            f"[{i}] (score={s:.3f}) Title: {b['title']} | Author: {b['author']}\n"
            f"Summary: {b['summary']}{tag_str}"
        )
    if best_i is not None:
        lines.append(f"\nMost relevant snippet: [{best_i}]")
    return "\n\n".join(lines)


# Heuristics to choose identify vs explain
import re as _re

_DEF_ID_PATTERNS = [
    r"\bwhich\s+book\b",
    r"\bwhat\s+book\b",
    r"\bname\s+of\s+the\s+book\b",
    r"\brecommend\s+a\s+book\b",
    r"\bbook\s+about\b",
]


def _is_identify_intent(q: str) -> bool:
    ql = (q or "").strip().lower()
    return any(_re.search(p, ql) for p in _DEF_ID_PATTERNS)


def _is_explain_intent(q: str) -> bool:
    return not _is_identify_intent(q)


def _enforce_title_only(raw_answer: str, snippets: List[Tuple[Dict[str, str], float]]) -> str:
    ans = (raw_answer or "").strip()
    titles = [b["title"] for b, _ in snippets]

    # Exact match
    if ans in titles:
        return ans

    # Fuzzy match (e.g., "Book Thief [3]")
    best = difflib.get_close_matches(ans, titles, n=1, cutoff=0.8)
    if best:
        return best[0]

    # Fallback: use most relevant snippet’s title (last in the sorted list)
    return titles[-1]


def detect_device() -> Tuple[str, str]:
    """Return (device_kind, dtype_str). device_kind in {mps,cuda,cpu}."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps", "float16"
        if torch.cuda.is_available():
            return "cuda", "float16"
        return "cpu", "float32"
    except Exception:
        return "cpu", "float32"


class Generator:
    def __init__(self, model_id: str, device_kind: str, dtype_str: str, local_model_path: Optional[str] = None):
        self.model_id = model_id
        self.device_kind = device_kind
        self.dtype_str = dtype_str
        self.local_model_path = local_model_path
        self._pipe = None

    def load(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
        import torch

        source = self.local_model_path if self.local_model_path else self.model_id
        tokenizer = AutoTokenizer.from_pretrained(source)

        model_kwargs = {}
        if self.device_kind == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs.update({
                    "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
                    "device_map": "auto",
                })
            except Exception:
                model_kwargs.update({"torch_dtype": getattr(torch, self.dtype_str)})
        else:
            model_kwargs.update({"torch_dtype": getattr(torch, self.dtype_str)})

        model = AutoModelForSeq2SeqLM.from_pretrained(source, **model_kwargs)

        if self.device_kind == "mps":
            model.to("mps")

        self._pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

        # Warmup
        _ = self._pipe("Answer briefly. Question: ping? Context: none", max_new_tokens=8, temperature=0.0,
                       do_sample=False)

    def generate(self, prompt: str, max_new_tokens: int = 96) -> str:
        if self._pipe is None:
            self.load()
        out = self._pipe(prompt, max_new_tokens=max_new_tokens, temperature=0.0, do_sample=False)
        return out[0]["generated_text"].strip()


# -------------------------------
# 4) RAG PUBLIC API
# -------------------------------
# def rag_answer(
#         question: str,
#         index: List[BookKw],
#         gen: Generator,
#         topk: int = 3,
#         show_context: bool = False,
#         adaptive_topk: bool = False
# ) -> str:
#     retrieved = retrieve(question, index, k=topk, adaptive=adaptive_topk)
#     if not retrieved:
#         return "I couldn't find relevant books in the catalog to answer that."
#
#     context = build_context(retrieved)
#     if show_context:
#         print("\n--- Retrieved Context ---\n" + context + "\n-------------------------\n")
#
#     prompt = (IDENTIFY_PROMPT_TEMPLATE if _is_identify_intent(question) else EXPLAIN_PROMPT_TEMPLATE) \
#         .format(question=question.strip(), context=context)
#
#     return gen.generate(prompt, max_new_tokens=96)

def rag_answer(
        question: str,
        index: List[BookKw],
        gen: Generator,
        topk: int = 3,
        show_context: bool = False,
        adaptive_topk: bool = False,
        add_citation: bool = False
) -> str:
    retrieved = retrieve(question, index, k=topk, adaptive=adaptive_topk)
    if not retrieved:
        return "I couldn't find relevant books in the catalog to answer that."

    context = build_context(retrieved)
    if show_context:
        print("\n--- Retrieved Context ---\n" + context + "\n-------------------------\n")

    identify = _is_identify_intent(question)
    prompt_template = IDENTIFY_PROMPT_TEMPLATE if identify else EXPLAIN_PROMPT_TEMPLATE
    prompt = prompt_template.format(question=question.strip(), context=context)

    max_new = 48 if identify else 96
    raw = gen.generate(prompt, max_new_tokens=max_new)

    if identify:
        # Force the answer to be one of the retrieved titles
        title = _enforce_title_only(raw, retrieved)
        if add_citation:
            # most relevant snippet = last in retrieved (sorted by score)
            best_idx = len(retrieved)
            return f"{title} [{best_idx}]"
        return title

    # explain intent → return the model's answer as-is
    return raw


# -------------------------------
# 5) CLI / MAIN
# -------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tiny RAG library chatbot (anti-position-bias, co-occurrence boosts)")
    p.add_argument("--data", default="books.json", help="Path to dataset JSON (title, author, summary, [tags])")
    # Accept ANY Hugging Face model id
    group = p.add_mutually_exclusive_group()
    group.add_argument("--model", default="google/flan-t5-small",
                       help="Hugging Face model id (e.g., google/flan-t5-base, facebook/bart-base, t5-small)")
    group.add_argument("--local-model-path", default=None,
                       help="Path to a locally saved model folder (overrides --model)")
    p.add_argument("--topk", type=int, default=3, help="Max book snippets to include as context")
    p.add_argument("--adaptive-topk", action="store_true",
                   help="If top score >> second, auto use k=1 for sharper answers")
    p.add_argument("--demo", action="store_true", help="Run demo questions, then start interactive chat")
    p.add_argument("--show-context", action="store_true", help="Print the retrieved context snippets before generating")
    p.add_argument("--offline", action="store_true", help="Force Transformers to run in offline mode (no internet)")
    return p.parse_args()


def main():
    args = parse_args()

    # Offline mode for transformers
    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    ensure_dataset(args.data)
    books = load_books(args.data)
    index = precompute_keywords(books)

    device_kind, dtype_str = detect_device()

    print("\n" + "=" * 72)
    print(f" Device     : {device_kind:<10} | Precision: {dtype_str:<8}")
    print(f" Model      : {args.local_model_path or args.model}")
    print(f" Data       : {args.data}")
    print(f" Top-k      : {args.topk} {'(adaptive on)' if args.adaptive_topk else ''}")
    print("=" * 72 + "\n")

    gen = Generator(args.model, device_kind, dtype_str, local_model_path=args.local_model_path)
    gen.load()

    if args.demo:
        demo_questions = [
            "What is The Hitchhiker's Guide to the Galaxy about?",
            "Which book has a hobbit who travels with dwarves and faces a dragon?",
            "monster hidden in Hogwarts?",
            "What book talks about wealth and the American Dream?",
            "Tell me about the society in 1984.",
        ]
        print("— Demo Q&A —")
        for q in demo_questions:
            print("-" * 72)
            print(f"Q: {q}")
            ans = rag_answer(q, index, gen, topk=args.topk, show_context=args.show_context,
                             adaptive_topk=args.adaptive_topk)
            print(f"A: {ans}")
        print("-" * 72)

    try:
        print("\nEnter your questions (Ctrl+C to exit).\n")
        while True:
            q = input("You: ").strip()
            if not q:
                continue
            ans = rag_answer(q, index, gen, topk=args.topk, show_context=args.show_context,
                             adaptive_topk=args.adaptive_topk)
            print("Bot:", ans)
    except (KeyboardInterrupt, EOFError):
        print("\nExit!\n")


if __name__ == "__main__":
    main()
