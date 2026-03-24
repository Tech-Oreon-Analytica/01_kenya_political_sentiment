"""
text_cleaning.py
----------------
Shared reusable text preprocessing utilities for sentiment analysis projects.
Handles standard NLP cleaning: URLs, mentions, hashtags, punctuation, whitespace.

Usage:
    from shared.utils.text_cleaning import clean_text, batch_clean
"""

import re
import string


# ── Regex patterns compiled once for efficiency ───────────────────────────────
_URL_PATTERN       = re.compile(r"http\S+|www\.\S+", re.IGNORECASE)
_MENTION_PATTERN   = re.compile(r"@\w+")
_HASHTAG_CLEAN     = re.compile(r"#(\w+)")      # keep the word, remove the '#'
_NUMERIC_PATTERN   = re.compile(r"\b\d+\b")
_EXTRA_SPACE       = re.compile(r"\s{2,}")
_EMOJI_PATTERN     = re.compile(
    "[\U00010000-\U0010ffff]", flags=re.UNICODE
)


def clean_text(
    text: str,
    remove_mentions : bool = True,
    remove_urls     : bool = True,
    expand_hashtags : bool = True,
    remove_numbers  : bool = False,
    lowercase       : bool = True,
) -> str:
    """
    Clean a single text string.

    Parameters
    ----------
    text            : Raw input string.
    remove_mentions : Strip @user handles.
    remove_urls     : Strip http(s) and www links.
    expand_hashtags : Replace #Topic → Topic (preserves topic signal for VADER).
    remove_numbers  : Optionally strip standalone numeric tokens.
    lowercase       : Convert to lowercase.

    Returns
    -------
    Cleaned string. Returns empty string if input is not a valid string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Remove URLs
    if remove_urls:
        text = _URL_PATTERN.sub("", text)

    # Remove @mentions
    if remove_mentions:
        text = _MENTION_PATTERN.sub("", text)

    # Expand or strip hashtags
    if expand_hashtags:
        text = _HASHTAG_CLEAN.sub(r"\1", text)   # #Elections → Elections
    else:
        text = _HASHTAG_CLEAN.sub("", text)

    # Remove emojis (preserves VADER's ability to read ASCII emoticons like :) )
    text = _EMOJI_PATTERN.sub("", text)

    # Optionally remove standalone numbers
    if remove_numbers:
        text = _NUMERIC_PATTERN.sub("", text)

    # Lowercase
    if lowercase:
        text = text.lower()

    # Normalise whitespace
    text = _EXTRA_SPACE.sub(" ", text).strip()

    return text


def batch_clean(texts, **kwargs) -> list:
    """Apply clean_text to an iterable of strings. Returns list of cleaned strings."""
    return [clean_text(t, **kwargs) for t in texts]
