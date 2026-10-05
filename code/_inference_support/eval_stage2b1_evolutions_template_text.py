import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

BASE_PATH = Path(__file__).resolve().parent / 'eval_stage2_model.py'
BUILD_PATH = Path(__file__).resolve().parent / 'build_stage2b1_evolution_template_text_dataset.py'

ROLE_FIELDS = ('input_ids', 'output_ids', 'location_ids', 'time_ids', 'mechanism_ids')
EVENT_BLOCK_PATTERN = re.compile(
    r'Evolution Event\s+\d+\s*:\s*(.*?)(?=(?:\n\s*Evolution Event\s+\d+\s*:)|\Z)',
    re.S,
)
CORE_LINE_WITH_TIME_PATTERN = re.compile(
    r'Global Core Evolution:\s*At\s*(.*?),\s*in\s*(.*?),\s*(.*?)\s*change into\s*(.*?)\s*through\s*(.*?)\.\s*$',
    re.S | re.I,
)
CORE_LINE_NO_TIME_PATTERN = re.compile(
    r'Global Core Evolution:\s*(?:In|Within|At)\s*(.*?),\s*(.*?)\s*change into\s*(.*?)\s*through\s*(.*?)\.\s*$',
    re.S | re.I,
)
ENTITY_ID_PATTERN = re.compile(r'\[E(\d+)\]')
EVENT_HEADER_ANY_PATTERN = re.compile(r'(^|\n)\s*Evolution Event\s+[^:\n：]*[：:]\s*', re.I)
CYRILLIC_E_PATTERN = re.compile(r'\[([Ее])\s*(\d+)\s*\]')
NONE_TOKEN_PATTERN = re.compile(r'^\[\s*none\s*\]\.?$', re.I)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Failed to load module from {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module(BASE_PATH, 'eval_stage2_model_base')
builder = load_module(BUILD_PATH, 'build_stage2b1_template_dataset')
PROMPT_MODES = ('standard',)
MERGE_OBJECT_MECHANISM = False
ALLOW_PARTIAL_CORE_EVENTS = False


def normalize_prediction_text(pred_text: str) -> str:
    text = (pred_text or '').strip()
    if not text:
        return text

    text = text.replace('：', ':')
    text = text.replace('，', ',')
    text = text.replace('。', '.')
    text = CYRILLIC_E_PATTERN.sub(r'[E\2]', text)
    text = re.sub(r'\[\s*[Nn]one\s*\]', '[None]', text)
    text = re.sub(r'Local Entity State:\s*None\.', 'Local Entity State: [None].', text, flags=re.I)
    text = re.sub(r'Local Entity State:\s*\[None\](?!\.)', 'Local Entity State: [None].', text, flags=re.I)
    text = re.sub(r'Global Core Evolution:\s*At\s*\[None\]\s*,\s*in\s*', 'Global Core Evolution: At [None], in ', text, flags=re.I)
    text = re.sub(r'Global Core Evolution:\s*At(?!\s)', 'Global Core Evolution: At ', text, flags=re.I)
    text = re.sub(r'Global Core Evolution:\s*In(?!\s)', 'Global Core Evolution: In ', text, flags=re.I)
    text = re.sub(r'Global Core Evolution:\s*Within(?!\s)', 'Global Core Evolution: Within ', text, flags=re.I)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)

    header_counter = 0

    def _renumber(match: re.Match) -> str:
        nonlocal header_counter
        header_counter += 1
        prefix = match.group(1) or ''
        return f'{prefix}Evolution Event {header_counter}: '

    text = EVENT_HEADER_ANY_PATTERN.sub(_renumber, text)
    return text.strip()


def parse_id_list(text: str) -> Optional[List[int]]:
    text = str(text).strip()
    if not text:
        return []
    if NONE_TOKEN_PATTERN.match(text):
        return []
    ids = [int(v) for v in ENTITY_ID_PATTERN.findall(text)]
    if not ids:
        return None
    return sorted(set(ids))


def has_valid_core_roles(evo: Dict) -> bool:
    has_input = bool(evo.get('input_ids'))
    has_output = bool(evo.get('output_ids'))
    has_mechanism = bool(evo.get('mechanism_ids'))
    if ALLOW_PARTIAL_CORE_EVENTS:
        return sum([has_input, has_output, has_mechanism]) >= 2
    return has_input and has_output


def canonical_label(label: str) -> str:
    norm = str(label or '').strip()
    if MERGE_OBJECT_MECHANISM and norm in {'Object', 'Mechanism'}:
        return 'Object'
    return norm


def parse_prediction(pred_text: str, strict_output: bool = False) -> Tuple[Optional[Dict], Optional[str]]:
    stripped = normalize_prediction_text(pred_text)
    if not stripped:
        return None, 'empty_output'
    if NONE_TOKEN_PATTERN.match(stripped):
        return {'evolutions': []}, None

    blocks = EVENT_BLOCK_PATTERN.findall(stripped)
    if not blocks:
        return None, 'missing_evolution_blocks'

    evolutions = []
    skipped_reasons = []
    for block in blocks:
        lines = [line.rstrip() for line in block.strip().splitlines() if line.strip()]
        core_lines = [line for line in lines if line.startswith('Global Core Evolution:')]
        if len(core_lines) != 1:
            skipped_reasons.append('missing_or_multiple_global_core_lines')
            continue
        core_line = core_lines[0].strip()
        match = CORE_LINE_WITH_TIME_PATTERN.match(core_line)
        if match is not None:
            time_text, location_text, input_text, output_text, mechanism_text = match.groups()
        else:
            match = CORE_LINE_NO_TIME_PATTERN.match(core_line)
            if match is None:
                skipped_reasons.append('invalid_global_core_format')
                continue
            location_text, input_text, output_text, mechanism_text = match.groups()
            time_text = '[None]'

        evo = {}
        invalid_field = None
        for field, raw_text in (
            ('time_ids', time_text),
            ('location_ids', location_text),
            ('input_ids', input_text),
            ('output_ids', output_text),
            ('mechanism_ids', mechanism_text),
        ):
            ids = parse_id_list(raw_text)
            if ids is None:
                invalid_field = f'invalid_{field}'
                break
            if ids:
                evo[field] = ids
        if invalid_field is not None:
            skipped_reasons.append(invalid_field)
            continue
        if not has_valid_core_roles(evo):
            if ALLOW_PARTIAL_CORE_EVENTS:
                skipped_reasons.append('each_evolution_must_have_at_least_two_core_roles')
            else:
                skipped_reasons.append('each_evolution_must_have_input_and_output')
            continue
        evolutions.append(evo)

    if not evolutions:
        if skipped_reasons:
            return None, skipped_reasons[0]
        return None, 'missing_valid_evolution_blocks'
    return {'evolutions': evolutions}, None


def prompt_mode_instructions(prompt_mode: str) -> str:
    snippets = []
    modes = [mode.strip() for mode in str(prompt_mode or 'standard').split(',') if mode.strip()]
    for mode in modes:
        if mode == 'standard':
            continue
        if mode == 'force_events':
            snippets.append(
                '\n\nAdditional extraction policy:\n'
                '1) If the passage explicitly contains verbs such as "utilizes", "generating", "drives", "forms", or "produces", do not output [None] unless there is truly no explicit input-to-output event.\n'
                '2) Prefer extracting the clearest explicit event rather than returning [None].\n'
                '3) When unsure, output fewer high-confidence events, but still extract the explicit input-output transformations that are directly stated.\n'
            )
        elif mode == 'format_lock':
            snippets.append(
                '\n\nAdditional formatting policy:\n'
                '1) Use ASCII punctuation only. Use ":" not "：".\n'
                '2) Event headers must be exactly: Evolution Event 1:, Evolution Event 2:, etc.\n'
                '3) Use only ASCII entity and sentence tags, such as [E1], [E2], [S1], [S2]. Never use other characters like [Е1] or #E1#.\n'
                '4) Global Core Evolution must exactly follow this pattern: At [time], in [location], [input ids] change into [output ids] through [mechanism ids].\n'
                '5) If there is no local state, write exactly: Local Entity State: [None].\n'
            )
        elif mode == 'summary_bridge':
            snippets.append(
                '\n\nAdditional extraction policy for summary-style passages:\n'
                '1) If a sentence summarizes a transformation using broad nouns or grouped outputs, extract the main explicit transformation instead of returning [None].\n'
                '2) Ignore pure background definitions, but keep any sentence that explicitly states an input, an output, and a driving mechanism.\n'
                '3) Prefer the top-level evolution chain over fine-grained subevents when the passage is summary-like.\n'
            )
        elif mode == 'precision_guard':
            snippets.append(
                '\n\nAdditional precision policy:\n'
                '1) Do not convert every sentence into an evolution event. Output [None] for a sentence unless it explicitly states a concrete input-to-output transformation.\n'
                '2) Do not extract events from taxonomy, category comparisons, domain descriptions, framework summaries, process listings, or background statements.\n'
                '3) If one sentence lists several outputs produced by the same input, location, time, and mechanism, keep them in one event instead of splitting them into multiple near-duplicate events.\n'
                '4) Do not treat a mechanism word, function word, relation word, heading word, or attribute-only phrase as an input or output entity.\n'
                '5) Do not infer missing input or output entities from nearby sentences. If the current evidence sentence does not explicitly support a full transformation, do not output that event.\n'
                '6) Prefer fewer high-confidence events over many speculative events.\n'
                '7) If a sentence mainly describes states, labels, categories, co-occurrence, or comparison rather than change, do not extract an evolution event from it.\n'
            )
        elif mode == 'slot_guard':
            snippets.append(
                '\n\nAdditional slot assignment policy:\n'
                '1) Do not place entities labeled Location, Time, or Attribute into input_ids or output_ids.\n'
                '2) Use Location entities only in location_ids or local location states.\n'
                '3) Use Time entities only in time_ids or local time states.\n'
                '4) Use Attribute entities only in local attribute states, never as core transformed inputs or outputs.\n'
                '5) If a candidate event would require using only Location, Time, or Attribute entities as inputs or outputs, do not output that event.\n'
                '6) Prefer core entities that denote geomorphic objects, materials, landforms, or process participants for input_ids and output_ids.\n'
            )
        elif mode == 'event_boundary_guard':
            snippets.append(
                '\n\nAdditional event boundary policy:\n'
                '1) Extract at most one main evolution event from one evidence sentence unless that same sentence explicitly states two separate and complete transformation chains.\n'
                '2) If one sentence contains one input-to-output transformation with several outputs, keep those outputs in one event instead of splitting them into multiple events.\n'
                '3) Do not build one event by stitching together unrelated fragments from multiple sentences.\n'
                '4) Use multiple evidence sentences for one event only when they are consecutive and clearly describe the same single transformation process.\n'
                '5) If sentence A gives only a mechanism or only a background condition, and sentence B gives a different transformation, do not merge them into one event.\n'
                '6) Do not turn taxonomy, composition, enumeration, framework description, or domain comparison into evolution events.\n'
                '7) Prefer one well-bounded event with complete local evidence over several overlapping events built from the same sentence region.\n'
            )
    return ''.join(snippets)


def build_qwen_messages(system_text: str, user_text: str, strict_output: bool = False, retry_reason: Optional[str] = None):
    cleaned_user = user_text.strip()
    if strict_output:
        cleaned_user += (
            '\n\nStrict output constraints:\n'
            '1) Output either exactly [None] or one or more Evolution Event blocks.\n'
            '2) Each event must contain exactly one line starting with "Global Core Evolution:".\n'
            '3) In structured fields, use only ids such as [E1], [E2], [E3].\n'
            '4) If a slot is missing, write [None].\n'
            '5) Do not invent ids that do not appear in the input text.\n'
            '6) Do not output explanations, markdown, or comments.\n'
        )
    cleaned_user += prompt_mode_instructions(','.join(PROMPT_MODES))
    if retry_reason:
        cleaned_user += f'\n\nPrevious output was invalid ({retry_reason}). Regenerate and strictly follow the constraints.'
    return [
        {'role': 'system', 'content': [{'type': 'text', 'text': system_text}]},
        {'role': 'user', 'content': [{'type': 'text', 'text': cleaned_user}]},
    ]


def load_gold(path: str, gold_entity_source_path: Optional[str] = None) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    gold_entity_rows = data
    if gold_entity_source_path:
        with open(gold_entity_source_path, 'r', encoding='utf-8') as f:
            gold_entity_rows = json.load(f)
    if len(gold_entity_rows) != len(data):
        raise RuntimeError('gold_entity_source_path must have the same number of samples as gold-path')

    samples = []
    for i, item in enumerate(data):
        gold_entity_item = gold_entity_rows[i]
        raw_user = item['messages'][1]['content']
        entities_raw, input_text = builder.extract_entities_and_text(raw_user)
        entities = [builder.normalize_entity(entity, input_text) for entity in entities_raw]
        tagged_text, _, _, _ = builder.build_tagged_text(input_text, entities)
        gold_obj, _ = parse_prediction_from_gold_json(item['messages'][2]['content'])
        gold_entities_raw, _ = builder.extract_entities_and_text(gold_entity_item['messages'][1]['content'])
        samples.append(
            {
                'index': i,
                'system': builder.SYSTEM_PROMPT,
                'user': builder.USER_PROMPT_TEMPLATE.format(tagged_text=tagged_text),
                'gold_raw': item['messages'][2]['content'],
                'gold': gold_obj,
                'pred_entities': entities_raw,
                'gold_entities': gold_entities_raw,
            }
        )
    return samples


def parse_prediction_from_gold_json(assistant_text: str) -> Tuple[Dict, Optional[str]]:
    pred_obj = base.extract_json_object(assistant_text)
    if pred_obj is None:
        return {'evolutions': []}, 'invalid_gold_json'
    evolutions = pred_obj.get('evolutions', [])
    norm_evos = []
    for evo in evolutions:
        if not isinstance(evo, dict):
            continue
        norm = {}
        for field in ROLE_FIELDS:
            vals = evo.get(field, [])
            if not isinstance(vals, list):
                continue
            cleaned = []
            for v in vals:
                try:
                    cleaned.append(int(v))
                except Exception:
                    continue
            if cleaned:
                norm[field] = sorted(set(cleaned))
        if has_valid_core_roles(norm):
            norm_evos.append(norm)
    return {'evolutions': norm_evos}, None


def entity_map_by_id(entities: List[Dict]) -> Dict[int, Dict]:
    out = {}
    for e in entities:
        try:
            out[int(e['id'])] = {'text': str(e.get('text', '')).strip(), 'label': str(e.get('label', '')).strip()}
        except Exception:
            continue
    return out


def role_signature(evo: Dict, entity_map: Dict[int, Dict]) -> Dict[str, List[Tuple[str, str]]]:
    sig = {}
    for field in ROLE_FIELDS:
        vals = evo.get(field, [])
        items = []
        for vid in vals:
            ent = entity_map.get(int(vid))
            if ent:
                items.append((ent['text'], canonical_label(ent['label'])))
        if items:
            sig[field] = sorted(set(items))
    return sig


def evo_similarity(pred_sig: Dict, gold_sig: Dict, sim, tau: float) -> float:
    scores = []
    used_fields = 0
    for field in ROLE_FIELDS:
        p_items = pred_sig.get(field, [])
        g_items = gold_sig.get(field, [])
        if not p_items and not g_items:
            continue
        used_fields += 1
        if not p_items or not g_items:
            scores.append(0.0)
            continue
        matched = 0
        used_g = set()
        for pt, pl in p_items:
            best_j = None
            best_s = 0.0
            for j, (gt, gl) in enumerate(g_items):
                if j in used_g or pl != gl:
                    continue
                s = sim.score(pt, gt)
                if s > best_s:
                    best_s = s
                    best_j = j
            if best_j is not None and best_s >= tau:
                used_g.add(best_j)
                matched += 1
        denom = max(len(p_items), len(g_items))
        scores.append(matched / denom if denom else 0.0)
    if used_fields == 0:
        return 0.0
    return sum(scores) / used_fields


def evolution_soft_match(pred_obj: Dict, gold_obj: Dict, pred_entities: List[Dict], gold_entities: List[Dict], sim, tau: float) -> int:
    pred_evos = pred_obj.get('evolutions', [])
    gold_evos = gold_obj.get('evolutions', [])
    if not pred_evos or not gold_evos:
        return 0
    pmap = entity_map_by_id(pred_entities)
    gmap = entity_map_by_id(gold_entities)
    matrix = []
    for pe in pred_evos:
        prow = []
        psig = role_signature(pe, pmap)
        for ge in gold_evos:
            gsig = role_signature(ge, gmap)
            prow.append(evo_similarity(psig, gsig, sim, tau))
        matrix.append(prow)
    matched = base.best_bipartite_matches(matrix)
    return sum(1 for _, _, s in matched if s >= tau)


def compute_structure_metrics(pred_objs: List[Optional[Dict]], pred_entities_by_sample: List[List[Dict]]) -> Dict[str, float]:
    total = len(pred_objs)
    format_valid = 0
    id_valid = 0
    for obj, ents in zip(pred_objs, pred_entities_by_sample):
        if obj is None:
            continue
        format_valid += 1
        valid_ids = {int(e['id']) for e in ents if 'id' in e}
        ok = True
        for evo in obj.get('evolutions', []):
            for field in ROLE_FIELDS:
                for vid in evo.get(field, []):
                    if int(vid) not in valid_ids:
                        ok = False
        if ok:
            id_valid += 1
    return {
        'format_valid_rate': format_valid / total if total else 0.0,
        'id_valid_rate': id_valid / format_valid if format_valid else 0.0,
    }


def run_inference(
    samples: List[Dict],
    model_path: str,
    adapter_path: str,
    max_new_tokens: int,
    temperature: float,
    strict_output: bool,
    retry_on_invalid: int,
    retry_max_new_tokens: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> List[str]:
    print('[INFO] Loading stage2 model...')
    model, processor, model_type = base.load_vl_model_and_processor(
        model_path=model_path,
        adapter_path=adapter_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    print(f"[INFO] Loaded VL model_type={model_type or 'unknown'}")

    first_param = next(model.parameters())
    target_device = first_param.device

    outputs = []
    retried = 0
    repaired = 0
    unresolved_invalid = 0

    if temperature is None or temperature <= 0:
        gen_kwargs = {
            'max_new_tokens': max_new_tokens,
            'do_sample': False,
            'repetition_penalty': repetition_penalty,
        }
    else:
        gen_kwargs = {
            'max_new_tokens': max_new_tokens,
            'do_sample': True,
            'temperature': temperature,
            'top_p': 0.9,
            'repetition_penalty': repetition_penalty,
        }
    if no_repeat_ngram_size > 0:
        gen_kwargs['no_repeat_ngram_size'] = no_repeat_ngram_size

    for i, s in enumerate(samples):
        final_text = ''
        retry_reason = None
        made_retry = False
        valid_after_retry = False

        for attempt in range(max(0, retry_on_invalid) + 1):
            messages = build_qwen_messages(
                s['system'],
                s['user'],
                strict_output=strict_output,
                retry_reason=retry_reason if attempt > 0 else None,
            )
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], return_tensors='pt', padding=True)
            inputs = {k: v.to(target_device) for k, v in inputs.items()}

            call_gen_kwargs = dict(gen_kwargs)
            if attempt > 0 and retry_max_new_tokens > 0:
                call_gen_kwargs['max_new_tokens'] = min(max_new_tokens, retry_max_new_tokens)

            with torch.no_grad():
                generated = model.generate(**inputs, **call_gen_kwargs)

            trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated)]
            pred_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            final_text = pred_text

            if not strict_output:
                break

            _, retry_reason = parse_prediction(pred_text, strict_output=True)
            if retry_reason is None:
                if made_retry:
                    valid_after_retry = True
                break

            if attempt < max(0, retry_on_invalid):
                made_retry = True
                continue
            unresolved_invalid += 1

        if made_retry:
            retried += 1
        if valid_after_retry:
            repaired += 1

        outputs.append(final_text)
        if (i + 1) % 5 == 0 or (i + 1) == len(samples):
            print(f'[INFO] Inference progress: {i + 1}/{len(samples)}')

    if strict_output and retry_on_invalid > 0:
        print(
            '[INFO] Strict retry stats: '
            f'retried_samples={retried}, repaired_after_retry={repaired}, still_invalid={unresolved_invalid}'
        )
    return outputs


def save_predictions(path: str, samples: List[Dict], pred_texts: List[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for s, p in zip(samples, pred_texts):
        rows.append({'index': s['index'], 'prediction': p})
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def load_predictions(path: str) -> Dict[int, str]:
    with open(path, 'r', encoding='utf-8') as f:
        rows = json.load(f)
    out = {}
    for row in rows:
        out[int(row['index'])] = row.get('prediction', '')
    return out


def evaluate(samples: List[Dict], pred_texts: List[str], tau: float, sim_model: str) -> Dict:
    sim = base.SemanticSimilarity(sim_model)
    evo_metric = base.MetricCounter()
    pred_objs = []
    pred_entities_by_sample = []
    per_sample = []

    for s, pred_text in zip(samples, pred_texts):
        pred_obj, invalid_reason = parse_prediction(pred_text, strict_output=True)
        pred_objs.append(pred_obj)
        pred_entities = s['pred_entities']
        pred_entities_by_sample.append(pred_entities)
        gold_obj = s['gold']
        gold_entities = s['gold_entities']
        if pred_obj is None:
            evo_metric.fn += len(gold_obj['evolutions'])
            per_sample.append({'index': s['index'], 'format_valid': False, 'invalid_reason': invalid_reason})
            continue
        evo_tp = evolution_soft_match(pred_obj, gold_obj, pred_entities, gold_entities, sim, tau)
        evo_metric.tp += evo_tp
        evo_metric.fp += max(0, len(pred_obj['evolutions']) - evo_tp)
        evo_metric.fn += max(0, len(gold_obj['evolutions']) - evo_tp)
        per_sample.append(
            {
                'index': s['index'],
                'format_valid': True,
                'pred_evolution_count': len(pred_obj['evolutions']),
                'gold_evolution_count': len(gold_obj['evolutions']),
                'evolution_tp': evo_tp,
            }
        )

    structure = compute_structure_metrics(pred_objs, pred_entities_by_sample)
    return {
        'tau': tau,
        'evolution_soft': {
            'precision': evo_metric.precision(),
            'recall': evo_metric.recall(),
            'f1': evo_metric.f1(),
            'tp': evo_metric.tp,
            'fp': evo_metric.fp,
            'fn': evo_metric.fn,
        },
        'structure': structure,
        'num_samples': len(samples),
        'per_sample': per_sample,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Stage2B-1 template-text evaluation')
    parser.add_argument('--gold-path', type=str, required=True)
    parser.add_argument('--pred-path', type=str, required=True)
    parser.add_argument('--result-path', type=str, required=True)
    parser.add_argument('--gold-entity-source-path', type=str, default='')
    parser.add_argument('--run-inference', action='store_true')
    parser.add_argument('--model-path', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--adapter-path', type=str, default='')
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--sim-model', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--max-new-tokens', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--repetition-penalty', type=float, default=1.05)
    parser.add_argument('--no-repeat-ngram-size', type=int, default=0)
    parser.add_argument('--strict-output', action='store_true')
    parser.add_argument('--retry-on-invalid', type=int, default=0)
    parser.add_argument('--retry-max-new-tokens', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--prompt-mode', type=str, default='standard')
    parser.add_argument('--merge-object-mechanism', action='store_true')
    parser.add_argument('--allow-partial-core-events', action='store_true')
    return parser.parse_args()


def main():
    global PROMPT_MODES, MERGE_OBJECT_MECHANISM, ALLOW_PARTIAL_CORE_EVENTS
    args = parse_args()
    PROMPT_MODES = tuple(mode.strip() for mode in str(args.prompt_mode or 'standard').split(',') if mode.strip()) or ('standard',)
    MERGE_OBJECT_MECHANISM = bool(args.merge_object_mechanism)
    ALLOW_PARTIAL_CORE_EVENTS = bool(args.allow_partial_core_events)
    samples = load_gold(args.gold_path, gold_entity_source_path=args.gold_entity_source_path or None)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    if args.run_inference:
        pred_texts = run_inference(
            samples=samples,
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            strict_output=args.strict_output,
            retry_on_invalid=args.retry_on_invalid,
            retry_max_new_tokens=args.retry_max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        save_predictions(args.pred_path, samples, pred_texts)
        print(f'[INFO] Predictions saved to: {args.pred_path}')
    else:
        pred_map = load_predictions(args.pred_path)
        pred_texts = [pred_map.get(s['index'], '') for s in samples]

    result = evaluate(samples, pred_texts, tau=args.tau, sim_model=args.sim_model)
    Path(args.result_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print('\n===== Evaluation Summary =====')
    print(f"Samples: {result['num_samples']}")
    print(f"Evolution Soft F1: {result['evolution_soft']['f1']:.4f}")
    print(f"Template valid rate: {result['structure']['format_valid_rate']:.4f}")
    print(f"ID valid rate: {result['structure']['id_valid_rate']:.4f}")
    print(f"Results saved to: {args.result_path}")


if __name__ == '__main__':
    main()
