import re
from typing import Dict, List, Optional, Tuple

from transformers import StoppingCriteria


PARAGRAPH_HEADER = re.compile(r"(?:^|\n)\s*Paragraph:\s*", re.I)
EVENT_HEADER = re.compile(r"(?:^|\n)\s*Event Sentences:\s*", re.I)
COMPLETE_SENTENCE = re.compile(r".*?[.!?](?=\s|$)", re.S)


def normalize_sentence(text: str) -> str:
    return " ".join(text.split()).casefold()


def complete_sentences(text: str) -> List[str]:
    return [" ".join(match.group(0).split()) for match in COMPLETE_SENTENCE.finditer(text or "")]


def _unique(sentences: List[str], limit: int) -> Tuple[List[str], int]:
    kept: List[str] = []
    seen = set()
    duplicate_count = 0
    for sentence in sentences:
        key = normalize_sentence(sentence)
        if not key or key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        kept.append(sentence)
        if len(kept) >= limit:
            break
    return kept, duplicate_count


def split_sections(text: str) -> Tuple[str, Optional[str], bool, bool]:
    text = (text or "").strip()
    paragraph_match = PARAGRAPH_HEADER.search(text)
    event_match = EVENT_HEADER.search(text)
    has_paragraph = paragraph_match is not None
    has_events = event_match is not None

    paragraph_start = paragraph_match.end() if paragraph_match else 0
    paragraph_end = event_match.start() if event_match else len(text)
    paragraph = text[paragraph_start:paragraph_end].strip()
    events = text[event_match.end():].strip() if event_match else None
    return paragraph, events, has_paragraph, has_events


def event_sentences(text: str) -> List[str]:
    sentences: List[str] = []
    for raw_line in (text or "").splitlines():
        line = re.sub(r"^\s*(?:[-*]\s+|\d+[.)]\s*)", "", raw_line).strip()
        if line.casefold() in {"paragraph:", "event sentences:"}:
            continue
        if not line or line.casefold() in {"none", "none."}:
            continue
        complete = complete_sentences(line)
        if complete:
            sentences.extend(complete)
        elif line:
            sentences.append(" ".join(line.split()))
    return sentences


def protect_output(
    primary_text: str,
    retry_text: Optional[str],
    paragraph_max_sentences: int,
    event_max_sentences: int,
) -> Tuple[str, Dict]:
    paragraph, primary_events, has_paragraph, has_events = split_sections(primary_text)
    paragraph_items, paragraph_duplicates = _unique(
        complete_sentences(paragraph), paragraph_max_sentences
    )
    if not paragraph_items and paragraph.strip():
        paragraph_items = [" ".join(paragraph.split())]

    event_source = "primary"
    raw_events = primary_events
    if not has_events:
        event_source = "retry" if retry_text else "missing"
        if retry_text:
            _, retry_events, _, retry_has_events = split_sections(retry_text)
            raw_events = retry_events if retry_has_events else retry_text.strip()

    event_items, event_duplicates = _unique(
        event_sentences(raw_events or ""), event_max_sentences
    )
    paragraph_text = " ".join(paragraph_items).strip() or "None."
    events_text = "\n".join(event_items).strip() or "None."
    protected = f"Paragraph:\n{paragraph_text}\nEvent Sentences:\n{events_text}"
    return protected, {
        "primary_had_paragraph_header": has_paragraph,
        "primary_had_event_header": has_events,
        "event_source": event_source,
        "paragraph_sentence_count": len(paragraph_items),
        "event_sentence_count": len(event_items),
        "paragraph_duplicates_removed": paragraph_duplicates,
        "event_duplicates_removed": event_duplicates,
    }


def needs_event_retry(primary_text: str) -> bool:
    return EVENT_HEADER.search(primary_text or "") is None


class ParagraphEventsStoppingCriteria(StoppingCriteria):
    """Stops batch-size-one generation after a completed loop or section limit."""

    def __init__(
        self,
        tokenizer,
        prompt_length: int,
        paragraph_max_sentences: int,
        event_max_sentences: int,
        event_only: bool = False,
    ):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.paragraph_max_sentences = paragraph_max_sentences
        self.event_max_sentences = event_max_sentences
        self.event_only = event_only
        self.stop_reason: Optional[str] = None

    @staticmethod
    def _has_duplicate(sentences: List[str]) -> bool:
        keys = [normalize_sentence(sentence) for sentence in sentences]
        return len(keys) != len(set(keys))

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids[0, self.prompt_length:]
        text = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if self.event_only:
            _, event_text, _, has_events = split_sections(text)
            sentences = event_sentences(event_text if has_events else text)
            if self._has_duplicate(sentences):
                self.stop_reason = "event_sentence_loop"
                return True
            if len(sentences) >= self.event_max_sentences:
                self.stop_reason = "event_sentence_limit"
                return True
            return False

        paragraph, events, _, has_events = split_sections(text)
        if has_events:
            sentences = event_sentences(events or "")
            if self._has_duplicate(sentences):
                self.stop_reason = "event_sentence_loop"
                return True
            if len(sentences) >= self.event_max_sentences:
                self.stop_reason = "event_sentence_limit"
                return True
            return False

        sentences = complete_sentences(paragraph)
        if self._has_duplicate(sentences):
            self.stop_reason = "paragraph_sentence_loop"
            return True
        # Allow exactly the requested number so the model can still emit the
        # Event Sentences header; stop only when it starts an extra sentence.
        if len(sentences) > self.paragraph_max_sentences:
            self.stop_reason = "paragraph_sentence_limit"
            return True
        return False
