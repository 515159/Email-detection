from __future__ import annotations

import html
import re


HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
NON_LETTER_RE = re.compile(r"[^a-zA-Z\s]")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize noisy email content before vectorization."""
    text = html.unescape(str(text)).lower()
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = NON_LETTER_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()
