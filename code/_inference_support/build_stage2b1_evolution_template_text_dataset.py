import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SYSTEM_PROMPT = 'You are a helpful assistant for geomorphology knowledge extraction.'
ROLE_FIELDS = ('input_ids', 'output_ids', 'location_ids', 'time_ids', 'mechanism_ids')
LOCAL_REL_TYPES = ('hasAttribute', 'hasLocation', 'hasTime')
REL_ORDER = {'hasAttribute': 0, 'hasLocation': 1, 'hasTime': 2}
SENT_BOUNDARY = re.compile(r'.*?(?:[.!?](?:\s+|$)|$)', re.S)

USER_PROMPT_TEMPLATE = '''You are given a geomorphology passage annotated with sentence tags and entity tags.

Your task is to extract all geomorphic evolution events from the passage.

Follow these rules strictly:
1. For each evolution event, first copy the exact supporting evidence from the input text.
2. Then fill the evolution template.
3. Use only the sentence tags [Sx]...[/Sx] and entity tags [Ex]...[/Ex] that already appear in the input text.
4. Do not invent any new entity, sentence, time, location, mechanism, attribute, or state.
5. If a slot is missing, always write [None].
6. In the structured template, output only entity ids such as [E1], [E2], [E3]. Do not output surface text such as "initial surface" or "weathering" inside the template fields.
7. If multiple input entities exist, separate them with commas.
8. If multiple output entities exist, separate them with commas.
9. If multiple mechanisms exist, separate them with commas.
10. If multiple local states exist, separate them with semicolons.
11. If two mentions have different entity ids, treat them as different entities even if their surface forms are identical.
12. Every evolution event must contain at least one input entity and one output entity. If either is missing, do not output that event.
13. Output an evolution event only when the text explicitly states that one or more input entities transform into, generate, form, or produce one or more output entities.
14. Do not output events for background descriptions, domain descriptions, process summaries, mechanism listings, driving factors, or partial process fragments without an explicit input-output transformation.
15. Keep the output format exactly the same as the template below.
16. Do not output explanations or notes.
17. If there are no geomorphic evolution events in the input text, output exactly: [None]

Output format:

Evolution Event 1:
Evidence: [copy the exact supporting sentence(s) with sentence tags and entity tags]
Global Core Evolution: At [time], in [location], [input entity or input entities] change into [output entity or output entities] through [mechanism or mechanisms].
Local Entity State: [entity] has attribute [attribute]; [entity] is located at [local location]; [entity] is at [local time].

Evolution Event 2:
Evidence: [copy the exact supporting sentence(s) with sentence tags and entity tags]
Global Core Evolution: At [time], in [location], [input entity or input entities] change into [output entity or output entities] through [mechanism or mechanisms].
Local Entity State: [entity] has attribute [attribute]; [entity] is located at [local location]; [entity] is at [local time].

If there is no local entity state, output exactly:
Local Entity State: [None].

If there is no time or location in the global core evolution, write:
At [None], in [None], ...

Input text:
{tagged_text}'''


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def dump_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_entities_and_text(user_text: str) -> Tuple[List[Dict], str]:
    start_marker = 'Provided Entities JSON:\n'
    input_marker = 'Input Text:'
    start = user_text.find(start_marker)
    if start == -1:
        raise ValueError('Failed to locate Provided Entities JSON block.')
    start += len(start_marker)
    input_pos = user_text.find(input_marker, start)
    if input_pos == -1:
        raise ValueError('Failed to locate Input Text block.')
    entities = json.loads(user_text[start:input_pos].strip()).get('entities', [])
    input_text = user_text[input_pos + len(input_marker):].strip()
    return entities, input_text


def find_fallback_spans(input_text: str, entity_text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    if not entity_text:
        return spans
    patterns = [
        re.compile(r'\b' + re.escape(entity_text) + r'\b'),
        re.compile(re.escape(entity_text)),
        re.compile(re.escape(entity_text) + r'\w*', re.IGNORECASE),
    ]
    for pattern in patterns:
        spans = [(match.start(), match.end()) for match in pattern.finditer(input_text)]
        if spans:
            return spans
    return spans


def normalize_entity(entity: Dict, input_text: str) -> Dict:
    spans = []
    if 'start_offset' in entity and 'end_offset' in entity:
        spans.append((int(entity['start_offset']), int(entity['end_offset'])))
    if 'span' in entity and isinstance(entity['span'], list) and len(entity['span']) == 2:
        spans.append((int(entity['span'][0]), int(entity['span'][1])))
    for cand in entity.get('offset_candidates', []):
        spans.append((int(cand['start_offset']), int(cand['end_offset'])))
    # dedupe while keeping order
    unique_spans = []
    seen = set()
    for span in spans:
        if span not in seen:
            seen.add(span)
            unique_spans.append(span)
    if not unique_spans:
        unique_spans = find_fallback_spans(input_text, str(entity.get('text', '')).strip())
    if not unique_spans:
        raise ValueError(f'Entity {entity.get("id")} is missing span grounding.')
    return {
        'id': int(entity['id']),
        'text': str(entity.get('text', '')),
        'label': str(entity.get('label', '')),
        'spans': unique_spans,
        'anchor_start': min(span[0] for span in unique_spans),
        'anchor_end': min(span[1] for span in unique_spans),
    }


def build_sentence_spans(text: str) -> List[Tuple[int, int]]:
    spans = []
    pos = 0
    for match in SENT_BOUNDARY.finditer(text):
        chunk = match.group(0)
        if not chunk:
            continue
        start = pos
        end = pos + len(chunk)
        if chunk.strip():
            spans.append((start, end))
        pos = end
        if pos >= len(text):
            break
    if not spans:
        spans.append((0, len(text)))
    return spans


def assign_sentence(start_offset: int, sentence_spans: Sequence[Tuple[int, int]]) -> int:
    for idx, (s0, s1) in enumerate(sentence_spans):
        if s0 <= start_offset < s1:
            return idx
    return len(sentence_spans) - 1


def render_tagged_sentence(sentence_text: str, sentence_start: int, mentions: Sequence[Tuple[int, int, int]]) -> str:
    # mentions: (start, end, entity_id) in absolute offsets
    mentions = sorted(mentions, key=lambda item: (item[0], item[1], item[2]))
    parts = []
    cursor = 0
    for start, end, entity_id in mentions:
        rel_start = start - sentence_start
        rel_end = end - sentence_start
        if rel_start < cursor:
            continue
        parts.append(sentence_text[cursor:rel_start])
        parts.append(f'[E{entity_id}]')
        parts.append(sentence_text[rel_start:rel_end])
        parts.append(f'[/E{entity_id}]')
        cursor = rel_end
    parts.append(sentence_text[cursor:])
    return ''.join(parts)


def build_tagged_text(input_text: str, entities: List[Dict]) -> Tuple[str, List[str], Dict[int, List[int]], List[Tuple[int, int]]]:
    sentence_spans = build_sentence_spans(input_text)
    sentence_mentions: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    entity_to_sentences: Dict[int, set] = defaultdict(set)

    for entity in entities:
        for start, end in entity['spans']:
            sent_idx = assign_sentence(start, sentence_spans)
            sentence_mentions[sent_idx].append((start, end, entity['id']))
            entity_to_sentences[entity['id']].add(sent_idx)

    tagged_sentences = []
    for idx, (s0, s1) in enumerate(sentence_spans):
        sent_text = input_text[s0:s1]
        tagged_body = render_tagged_sentence(sent_text, s0, sentence_mentions[idx])
        tagged_sentences.append(f'[S{idx + 1}]' + tagged_body + f'[/S{idx + 1}]')

    return ''.join(tagged_sentences), tagged_sentences, {k: sorted(v) for k, v in entity_to_sentences.items()}, sentence_spans


def parse_structure(assistant_text: str) -> Dict:
    return json.loads(assistant_text)


def format_id_list(values: Sequence[int]) -> str:
    if not values:
        return '[None]'
    return ','.join(f'[E{int(v)}]' for v in values)


def min_entity_anchor(entity_id: int, entity_map: Dict[int, Dict]) -> int:
    return entity_map[entity_id]['anchor_start']


def build_local_state_clauses(
    evolution: Dict,
    global_relations: Sequence[Dict],
    entity_to_sentences: Dict[int, List[int]],
    entity_map: Dict[int, Dict],
) -> Tuple[str, List[int]]:
    member_ids = set()
    for field in ROLE_FIELDS:
        member_ids.update(int(v) for v in evolution.get(field, []))
    global_location_ids = {int(v) for v in evolution.get('location_ids', [])}
    global_time_ids = {int(v) for v in evolution.get('time_ids', [])}

    kept = []
    supporting_entity_ids = []
    for rel in global_relations:
        rel_type = rel.get('type')
        if rel_type not in LOCAL_REL_TYPES:
            continue
        from_id = int(rel['from_id'])
        to_id = int(rel['to_id'])
        if from_id not in member_ids:
            continue
        if rel_type == 'hasLocation' and to_id in global_location_ids:
            continue
        if rel_type == 'hasTime' and to_id in global_time_ids:
            continue
        kept.append((from_id, to_id, rel_type))
        supporting_entity_ids.append(to_id)

    kept.sort(
        key=lambda item: (
            entity_to_sentences.get(item[0], [10**6])[0],
            min_entity_anchor(item[0], entity_map),
            REL_ORDER[item[2]],
            min_entity_anchor(item[1], entity_map),
        )
    )

    clauses = []
    for from_id, to_id, rel_type in kept:
        if rel_type == 'hasAttribute':
            clauses.append(f'[E{from_id}] has attribute [E{to_id}]')
        elif rel_type == 'hasLocation':
            clauses.append(f'[E{from_id}] is located at [E{to_id}]')
        elif rel_type == 'hasTime':
            clauses.append(f'[E{from_id}] is at [E{to_id}]')
    if not clauses:
        return '[None].', []
    return '; '.join(clauses) + '.', supporting_entity_ids


def evolution_sort_key(evolution: Dict, entity_to_sentences: Dict[int, List[int]], entity_map: Dict[int, Dict]) -> Tuple[int, int]:
    member_ids = []
    for field in ROLE_FIELDS:
        member_ids.extend(int(v) for v in evolution.get(field, []))
    if not member_ids:
        return (10**6, 10**6)
    min_sent = min(entity_to_sentences.get(entity_id, [10**6])[0] for entity_id in member_ids)
    min_offset = min(min_entity_anchor(entity_id, entity_map) for entity_id in member_ids)
    return (min_sent, min_offset)


def choose_evidence_window(support_ids: Sequence[int], entity_to_sentences: Dict[int, List[int]]) -> List[int]:
    support_ids = [entity_id for entity_id in support_ids if entity_id in entity_to_sentences]
    if not support_ids:
        return []
    all_sentence_ids = sorted({sent for entity_id in support_ids for sent in entity_to_sentences[entity_id]})
    best = None
    for left in all_sentence_ids:
        for right in all_sentence_ids:
            if right < left:
                continue
            covered = True
            for entity_id in support_ids:
                candidates = entity_to_sentences[entity_id]
                if not any(left <= sent <= right for sent in candidates):
                    covered = False
                    break
            if not covered:
                continue
            score = (right - left, left)
            if best is None or score < best[0]:
                best = (score, list(range(left, right + 1)))
    if best is not None:
        return best[1]
    return all_sentence_ids


def build_evidence(evolution: Dict, tagged_sentences: Sequence[str], entity_to_sentences: Dict[int, List[int]], extra_entity_ids: Sequence[int]) -> str:
    support_ids = set(extra_entity_ids)
    for field in ROLE_FIELDS:
        support_ids.update(int(v) for v in evolution.get(field, []))
    sentence_ids = choose_evidence_window(sorted(support_ids), entity_to_sentences)
    if not sentence_ids:
        return '[None]'
    return ''.join(tagged_sentences[idx] for idx in sentence_ids)


def build_assistant_output(
    structure: Dict,
    tagged_sentences: Sequence[str],
    entity_to_sentences: Dict[int, List[int]],
    entity_map: Dict[int, Dict],
) -> Tuple[str, int, int]:
    evolutions = structure.get('evolutions', [])
    global_relations = structure.get('global_relations', [])
    complete = []
    partial = 0
    for evolution in evolutions:
        has_input = bool(evolution.get('input_ids'))
        has_output = bool(evolution.get('output_ids'))
        if has_input and has_output:
            complete.append(evolution)
        else:
            partial += 1

    if not complete:
        return '[None]', 0, partial

    complete.sort(key=lambda evo: evolution_sort_key(evo, entity_to_sentences, entity_map))
    chunks = []
    for idx, evolution in enumerate(complete, start=1):
        local_state_text, extra_entity_ids = build_local_state_clauses(evolution, global_relations, entity_to_sentences, entity_map)
        evidence = build_evidence(evolution, tagged_sentences, entity_to_sentences, extra_entity_ids)
        chunks.append(
            f'Evolution Event {idx}:\n'
            f'Evidence: {evidence}\n'
            f'Global Core Evolution: At {format_id_list(evolution.get("time_ids", []))}, in {format_id_list(evolution.get("location_ids", []))}, {format_id_list(evolution.get("input_ids", []))} change into {format_id_list(evolution.get("output_ids", []))} through {format_id_list(evolution.get("mechanism_ids", []))}.\n'
            f'Local Entity State: {local_state_text}'
        )
    return '\n\n'.join(chunks), len(complete), partial


def build_dataset(rows: List[Dict]) -> Tuple[List[Dict], Dict]:
    dataset = []
    stats = Counter()
    for row in rows:
        entities_raw, input_text = extract_entities_and_text(row['messages'][1]['content'])
        entities = [normalize_entity(entity, input_text) for entity in entities_raw]
        entity_map = {entity['id']: entity for entity in entities}
        tagged_text, tagged_sentences, entity_to_sentences, _ = build_tagged_text(input_text, entities)
        structure = parse_structure(row['messages'][2]['content'])
        assistant_text, complete_events, partial_events = build_assistant_output(structure, tagged_sentences, entity_to_sentences, entity_map)

        dataset.append(
            {
                'messages': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': USER_PROMPT_TEMPLATE.format(tagged_text=tagged_text)},
                    {'role': 'assistant', 'content': assistant_text},
                ]
            }
        )
        stats['rows'] += 1
        stats['complete_events'] += complete_events
        stats['partial_events_skipped'] += partial_events
        stats['rows_none'] += int(assistant_text == '[None]')
        stats['rows_with_events'] += int(assistant_text != '[None]')
        stats['max_tagged_text_chars'] = max(stats['max_tagged_text_chars'], len(tagged_text))
        stats['max_output_chars'] = max(stats['max_output_chars'], len(assistant_text))
    return dataset, dict(stats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-input', required=True)
    parser.add_argument('--val-input', required=True)
    parser.add_argument('--train-output', required=True)
    parser.add_argument('--val-output', required=True)
    parser.add_argument('--metadata-output', required=True)
    args = parser.parse_args()

    train_rows = load_json(Path(args.train_input))
    val_rows = load_json(Path(args.val_input))

    train_dataset, train_stats = build_dataset(train_rows)
    val_dataset, val_stats = build_dataset(val_rows)

    dump_json(Path(args.train_output), train_dataset)
    dump_json(Path(args.val_output), val_dataset)
    dump_json(
        Path(args.metadata_output),
        {
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'train': train_stats,
            'val': val_stats,
            'description': 'Text-only evolution-event template dataset with tagged sentences/entities and English controlled-template outputs.',
        },
    )

    print(json.dumps({
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'train': train_stats,
        'val': val_stats,
        'train_output': args.train_output,
        'val_output': args.val_output,
        'metadata_output': args.metadata_output,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
