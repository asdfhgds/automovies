"""Cinematic subtitles.

Replaces paragraph captions with short, readable chunks (2-3 words), each with
word-level timings. Word timestamps are distributed evenly across the narration
window when real timestamps are unavailable — the chunker never invents
precision it doesn't have, but it does guarantee *short* readable captions.
"""
from typing import Dict, List, Optional

MAX_LINE_WORDS = 3
MAX_LINE_CHARS = 32

PUNCT_BOUNDARIES = {".", ",", "!", "?", ";", ":", "—"}


def split_into_captions(text: str, max_words: int = MAX_LINE_WORDS,
                        max_chars: int = MAX_LINE_CHARS) -> List[str]:
    """Split narration into short caption chunks.

    Prefers punctuation boundaries; falls back to word-count caps. Empty chunks
    are dropped; a single word longer than max_chars is kept whole.
    """
    if not text or not text.strip():
        return []
    words = text.split()
    chunks = []
    current = []
    for word in words:
        candidate = current + [word]
        line = " ".join(candidate)
        if len(candidate) > max_words or (line and len(line) > max_chars):
            _flush(chunks, current)
            current = [word]
        else:
            current = candidate
    _flush(chunks, current)
    return chunks


def _flush(chunks: List[str], current: List[str]):
    if current:
        # never split a token mid-word for display; keep lines short
        chunks.append(" ".join(current).strip())


def caption_word_timings(captions: List[str], start_sec: float,
                         duration_sec: float) -> List[dict]:
    """Distribute word timings evenly across a narration window.

    Returns::

        [{"text", "start_sec", "end_sec", "words": [{"word","start_sec","end_sec"}]}]
    """
    total_words = sum(len(c.split()) for c in captions)
    out = []
    t = start_sec
    for cap in captions:
        words = cap.split()
        if not words:
            continue
        cap_dur = duration_sec * (len(words) / max(1, total_words))
        step = cap_dur / len(words)
        word_list = []
        for i, w in enumerate(words):
            w_start = t + i * step
            word_list.append({
                "word": w,
                "start_sec": round(w_start, 3),
                "end_sec": round(w_start + step, 3),
            })
            if i == len(words) - 1:
                cap_end = w_start + step
        out.append({
            "text": cap,
            "start_sec": round(t, 3),
            "end_sec": round(cap_end, 3),
            "words": word_list,
        })
        t = cap_end
    return out


def merge_with_real_word_timestamps(text: str, words: List[dict],
                                    start_sec: float, duration_sec: float,
                                    max_words: int = MAX_LINE_WORDS) -> List[dict]:
    """Merge real word timestamps (transcript-style) into short captions.

    ``words``: ``[{"word"/"text", "start_sec"/"start", "end_sec"/"end"}]``.
    Falls back to even distribution when word timestamps are missing/empty.
    """
    normalized = []
    for w in words or []:
        word = w.get("word") or w.get("text") or ""
        s = w.get("start_sec", w.get("start"))
        e = w.get("end_sec", w.get("end"))
        if not word or s is None or e is None:
            continue
        normalized.append({"word": word, "start": float(s), "end": float(e)})
    if not normalized:
        captions = split_into_captions(text, max_words=max_words)
        return caption_word_timings(captions, start_sec, duration_sec)

    # group real words into captions preserving their own timings
    captions = []
    current = []
    for w in normalized:
        current.append(w)
        if len(current) >= max_words:
            _flush_timed(captions, current)
            current = []
    _flush_timed(captions, current)
    return captions


def _flush_timed(captions: List[dict], current: List[dict]):
    if not current:
        return
    captions.append({
        "text": " ".join(w["word"] for w in current),
        "start_sec": round(current[0]["start"], 3),
        "end_sec": round(current[-1]["end"], 3),
        "words": [
            {"word": w["word"], "start_sec": round(w["start"], 3),
             "end_sec": round(w["end"], 3)}
            for w in current
        ],
    })


def captions_to_srt_lines(captions: List[dict], offset_sec: float = 0.0) -> List[str]:
    """Render caption dicts to SRT blocks (single-line, short, uppercase)."""
    blocks = []
    for i, cap in enumerate(captions, start=1):
        s = offset_sec + cap.get("start_sec", 0.0)
        e = offset_sec + cap.get("end_sec", 0.0)
        blocks.append(f"{i}\n{_fmt_ts(s)} --> {_fmt_ts(e)}\n{cap.get('text', '').upper()}\n")
    return blocks


def _fmt_ts(sec: float) -> str:
    sec = max(0.0, sec)
    ms = int(round((sec - int(sec)) * 1000))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"