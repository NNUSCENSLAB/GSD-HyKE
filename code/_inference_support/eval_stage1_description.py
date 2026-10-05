import argparse
import json
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from peft import PeftModel
from PIL import Image
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, StoppingCriteriaList

from stage1_paragraph_events_guard import (
    ParagraphEventsStoppingCriteria,
    needs_event_retry,
    protect_output,
)

try:
    from transformers import AutoModelForVision2Seq, MllamaForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
except Exception:
    AutoModelForVision2Seq = None
    MllamaForConditionalGeneration = None
    Qwen2_5_VLForConditionalGeneration = None

try:
    from transformers import InternVLForConditionalGeneration
except Exception:
    InternVLForConditionalGeneration = None


WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
PARAGRAPH_SECTION_PATTERN = re.compile(r"Paragraph:\s*(.*?)(?:\n\s*Event Sentences:\s*|\Z)", re.S | re.I)
EVENT_SECTION_PATTERN = re.compile(r"Event Sentences:\s*(.*)\Z", re.S | re.I)
OLD_DESCRIPTION_TRAILER_PATTERN = re.compile(
    r"\n*Return only the descriptive paragraph, without headings or commentary\.\s*\Z",
    re.I,
)
TRAINED_PARAGRAPH_EVENTS_SUFFIX = (
    "\n\nReturn exactly two sections with these headers:\n"
    "Paragraph:\n"
    "Event Sentences:\n\n"
    "In Paragraph, write one coherent, source-grounded description of the figure. "
    "In Event Sentences, include only explicit input-to-output geomorphic changes supported by the image, caption, "
    "or Related Text, with exactly one atomic event per line. Preserve precise entity, mechanism, location, and time "
    "terms. Do not turn static descriptions, panel labels, legends, arrows, classifications, or spatial adjacency into "
    "events. Do not infer missing roles or add outside knowledge. Do not use bullets, numbering, JSON, or role labels. "
    "When no explicit geomorphic change is supported, write None. after the Event Sentences header."
)
TRAINED_PARAGRAPH_EVENTS_MECHANISM_GATE_SUFFIX = (
    "\n\nReturn exactly two sections with these headers:\n"
    "Paragraph:\n"
    "Event Sentences:\n\n"
    "In Paragraph, write one or two short, source-grounded sentences describing the figure. "
    "In Event Sentences, write only explicit geomorphic evolution events, with exactly one atomic event per line and "
    "at most eight event sentences. "
    "An event sentence is permitted only when the same source statement explicitly provides all three: "
    "(a) an input material, landform, or prior state; (b) a resulting material, landform, or transformed state; "
    "and (c) the process or mechanism linking them. Preserve the exact entity, mechanism, location, and time terms. "
    "Use this abstract pattern when it matches the source: At [time/location], [input] is transformed into [output] "
    "through [explicit process]. A statement such as 'At [time], panel [number] shows [label]' is static and must be "
    "omitted from Event Sentences, even if it has a time or a named object. Also omit labels, legends, panel titles, "
    "lists, spatial adjacency, composition, persistence, hosting, and state descriptions when no explicit process links "
    "an input to an output. Do not infer a missing mechanism or invent a bridge state merely to complete an event. "
    "Prefer omission over a partial event. Do not use bullets, numbering, JSON, or role labels. When no qualifying "
    "event is supported, write None. after the Event Sentences header."
)
CONSERVATIVE_APPENDIX = (
    "\n\nAdditional writing constraints:\n"
    "1) Keep the output as natural continuous prose in a single paragraph; do not use bullets, slot labels, or JSON.\n"
    "2) Stay close to the original geomorphology terminology from the image and contextual snippets; avoid replacing precise terms with broader paraphrases.\n"
    "3) Preserve distinct geomorphic processes or interactions separately; do not merge multiple events into one vague summary.\n"
    "4) Preserve important spatial, directional, temporal, and quantitative details when they are supported by the image or snippets.\n"
    "5) Explicitly mention key landforms, materials, mechanisms, and their interactions when supported.\n"
    "6) Do not add unsupported facts.\n"
)
CONSERVATIVE_BALANCED_APPENDIX = (
    "\n\nAdditional writing constraints:\n"
    "1) Keep the output as natural continuous prose in a single paragraph; do not use bullets, slot labels, or JSON.\n"
    "2) Stay close to the original geomorphology terminology from the image and contextual snippets; avoid replacing precise terms with broader paraphrases.\n"
    "3) Preserve distinct geomorphic processes or interactions separately; do not merge multiple events into one vague summary.\n"
    "4) Preserve the original spatial order, directional cues, and mechanism combinations whenever they are supported by the image or snippets.\n"
    "5) When supported, keep the relation structure explicit: location, process or mechanism, input material or landform, and resulting output landform or state.\n"
    "6) Do not skip important intermediate states in multi-step geomorphic processes.\n"
    "7) Preserve important spatial, directional, temporal, and quantitative details when they are supported by the image or snippets.\n"
    "8) Explicitly mention key landforms, materials, mechanisms, and their interactions when supported.\n"
    "9) Do not add unsupported facts, outside background knowledge, or inferred causal links that are not grounded in the image or snippets.\n"
)
PARAGRAPH_EVENTS_APPENDIX = (
    "\n\nAdditional writing constraints:\n"
    "1) Write the output in two sections exactly.\n"
    "2) First section header must be: Paragraph:\n"
    "3) Second section header must be: Event Sentences:\n"
    "4) In Paragraph, write one coherent natural paragraph describing the image.\n"
    "5) In Event Sentences, write 6 to 12 short natural sentences, one distinct geomorphic process, interaction, or form relation per sentence.\n"
    "6) Do not use bullets, numbering, JSON, or slot labels.\n"
    "7) Preserve technical geomorphology terms, quantities, directions, locations, and process-form relations whenever supported.\n"
    "8) Avoid merging multiple events into one vague sentence.\n"
    "9) Do not add unsupported facts.\n"
)
RELATION_PRESERVING_APPENDIX = (
    "\n\nAdditional writing constraints:\n"
    "1) Keep the output as natural continuous prose in a single paragraph; do not use bullets, slot labels, or JSON.\n"
    "2) Prioritize semantic fidelity over stylistic elegance; stay close to the original geomorphology terminology from the image and contextual snippets.\n"
    "3) Preserve each distinct process-form relation separately; do not compress several events into one summary sentence.\n"
    "4) Whenever supported, explicitly preserve the structure of location, mechanism or process, input material or landform, and resulting output landform or state.\n"
    "5) Preserve directional, spatial, temporal, and quantitative details when they appear in the image or snippets.\n"
    "6) Preserve technical trigger phrases such as utilizes, generating, characterized by, bounded by, composed of, containing, uplift, subsidence, erosion, deposition, and drainage when they are supported.\n"
    "7) Prefer repeating precise entity names over replacing them with pronouns or vague category words.\n"
    "8) Do not add unsupported facts.\n"
)
RELATION_PRESERVING_EVENTS_APPENDIX = (
    "\n\nAdditional writing constraints:\n"
    "1) Write the output in two sections exactly.\n"
    "2) First section header must be: Paragraph:\n"
    "3) Second section header must be: Event Sentences:\n"
    "4) In Paragraph, write one coherent natural paragraph describing the image with high semantic fidelity.\n"
    "5) In Event Sentences, write 8 to 16 short natural sentences, one distinct geomorphic process, interaction, composition relation, or form relation per sentence.\n"
    "6) Each event sentence should preserve, when supported, the location, mechanism or process, input entity, and output entity.\n"
    "7) Keep technical geomorphology terms unchanged when possible; avoid vague paraphrases and pronouns.\n"
    "8) Preserve relation phrases such as utilizes, generating, characterized by, bounded by, composed of, containing, uplift, drainage, erosion, deposition, and fragmentation when supported.\n"
    "9) Avoid merging multiple events into one sentence.\n"
    "10) Do not use bullets, numbering, JSON, or slot labels.\n"
    "11) Do not add unsupported facts.\n"
)
EVENT_CHAIN_STRICT_APPENDIX = (
    "\n\nAdditional writing constraints:\n"
    "1) Write the output in two sections exactly.\n"
    "2) First section header must be: Paragraph:\n"
    "3) Second section header must be: Event Sentences:\n"
    "4) In Paragraph, write one short coherent paragraph with high semantic fidelity and no extra background knowledge.\n"
    "5) In Event Sentences, write 6 to 14 short natural sentences, with exactly one atomic geomorphic relation per sentence.\n"
    "6) Preserve the original order of events, spatial layout, and mechanism combinations whenever supported.\n"
    "7) Keep exact named entities and technical geomorphology terms whenever supported; do not replace them with broader paraphrases or pronouns.\n"
    "8) Do not merge multiple mechanisms or multiple outputs into one vague summary sentence.\n"
    "9) Do not skip intermediate states in multi-step processes.\n"
    "10) When the evidence supports it, prefer relation patterns such as: At or Within [location], [process or mechanism] utilizes [input], generating [output].\n"
    "11) For non-process relations, preserve explicit forms such as: [entity] is characterized by [attribute], [entity] is bounded by [boundary], [entity] is composed of [parts], or [entity] contains [contents].\n"
    "12) Do not add unsupported examples, explanations, world knowledge, or inferred causal links.\n"
    "13) Do not use bullets, numbering, JSON, or slot labels inside the two sections.\n"
)
EVENT_CHAIN_PRECISION_GUARD_APPENDIX = (
    "\n\nAdditional writing constraints:\n"
    "1) Write the output in two sections exactly.\n"
    "2) First section header must be: Paragraph:\n"
    "3) Second section header must be: Event Sentences:\n"
    "4) In Paragraph, write one short coherent paragraph with high semantic fidelity and no extra background knowledge.\n"
    "5) In Event Sentences, write only high-confidence atomic geomorphic relations, with exactly one relation per sentence.\n"
    "6) Prefer omission over invention: if a candidate relation is uncertain, diagrammatic only, classificatory only, or purely descriptive, do not write it as an event sentence.\n"
    "7) Keep exact named entities, exact numeric values, units, thresholds, and directional terms whenever supported; do not replace them with vague summaries such as minimum, maximum, intermediate, drainage network, dome structure, or water flow.\n"
    "8) Do not merge multiple outputs, multiple mechanisms, or multiple stages into one sentence unless the source explicitly states them in one single atomic relation.\n"
    "9) Do not convert taxonomy, legend labels, composition lists, scale hierarchies, or channel-pattern comparisons into causal event sentences.\n"
    "10) Do not convert attribute statements such as characterized by, bounded by, composed of, or contains into generated causal chains.\n"
    "11) When the evidence clearly supports a process relation, prefer patterns such as: At or Within [location], [process or mechanism] utilizes [input], generating [output].\n"
    "12) For quantitative figures, preserve each explicit case separately; do not collapse exact measurements into rankings or trend summaries.\n"
    "13) Preserve original event order and intermediate states whenever clearly supported.\n"
    "14) Do not add unsupported facts, explanations, examples, or inferred links.\n"
    "15) Do not use bullets, numbering, JSON, or slot labels inside the two sections.\n"
)
EVENT_CHAIN_STAGE2_GUARD_APPENDIX = (
    "\n\nAdditional writing constraints:\n"
    "1) Write the output in two sections exactly.\n"
    "2) First section header must be: Paragraph:\n"
    "3) Second section header must be: Event Sentences:\n"
    "4) In Paragraph, write only 2 to 4 short sentences that summarize the figure with high fidelity and no background knowledge.\n"
    "5) In Event Sentences, write 8 to 20 short natural sentences, with exactly one atomic geomorphic event per sentence.\n"
    "6) Each event sentence should preserve, whenever supported, the structure: location, mechanism or process, input entity, and output entity.\n"
    "7) When the evidence clearly supports a process relation, prefer patterns such as: At or Within [location], [mechanism or process] utilizes [input], generating [output].\n"
    "8) Keep exact named entities, exact technical terms, exact numeric values, exact units, exact directional words, and exact temporal expressions whenever supported.\n"
    "9) Prefer copying precise terms from the image or snippets over replacing them with broad paraphrases, summaries, rankings, or generic words.\n"
    "10) Do not write overview sentences such as the diagram illustrates, the figure shows, distinct types, categorized into, based on, or similar summary framing inside Event Sentences.\n"
    "11) Do not convert taxonomy, legend labels, type names, scale hierarchies, example lists, composition lists, or comparison statements into causal events unless the source explicitly states an input-to-output transformation.\n"
    "12) Do not rewrite exact quantitative cases as rankings or trend summaries such as highest, lowest, maximum, minimum, stronger, weaker, more, less, or followed by, unless those ranking words are explicitly written in the source.\n"
    "13) For quantitative diagrams, preserve each explicit case separately. Do not collapse multiple cases into one sentence.\n"
    "14) Do not merge distant stages, distant regions, or multiple alternative pathways into one event sentence.\n"
    "15) Do not skip important intermediate states in multi-step processes when they are explicitly shown.\n"
    "16) If a sentence would have no explicit output entity, do not write it as an event sentence.\n"
    "17) Prefer omission over invention. If a candidate event is uncertain, classificatory only, descriptive only, or explanatory only, leave it out.\n"
    "18) Do not add unsupported facts, outside background knowledge, inferred causal links, or rewritten scientific explanations.\n"
    "19) Do not use bullets, numbering, JSON, or slot labels inside the two sections.\n"
)
EVENT_CHAIN_VERBATIM_GUARD_APPENDIX = (
    "\n\nAdditional writing constraints:\n"
    "1) Write the output in two sections exactly.\n"
    "2) First section header must be: Paragraph:\n"
    "3) Second section header must be: Event Sentences:\n"
    "4) In Paragraph, write only 1 to 3 short sentences that summarize the figure with high fidelity and no background knowledge.\n"
    "5) In Event Sentences, write 8 to 24 short natural sentences, with exactly one atomic geomorphic event per sentence.\n"
    "6) Treat Event Sentences as extraction-friendly event skeletons rather than polished prose.\n"
    "7) Preserve the original event order and keep one source case per sentence whenever the source gives separate cases, rows, stages, regions, scales, pathways, or directional variants.\n"
    "8) When the evidence clearly supports a process relation, prefer patterns such as: At or Within [location], [mechanism or process] utilizes [input], generating [output].\n"
    "9) Copy precise entity names, process names, quantities, units, thresholds, directional terms, and temporal expressions from the source whenever supported.\n"
    "10) Do not replace precise entities with broad paraphrases or generic placeholders such as landform, water bodies, topographic domes, profile, curve, interaction, feature, system, output, or shape unless that exact phrase is in the source.\n"
    "11) Do not rewrite explicit quantitative cases as comparisons, rankings, or trend summaries such as upper curve, lower curve, middle curve, maximum, minimum, stronger, weaker, more, less, or similar analytical wording unless those exact words are written in the source.\n"
    "12) For quantitative figures, preserve each explicit case separately. Do not compress multiple transportation modes, slopes, stages, passes, regions, or outputs into one summary sentence.\n"
    "13) Do not merge distant stages, distant regions, multiple alternative pathways, or multi-step chains into one sentence when the source presents them as separate events.\n"
    "14) Do not skip important intermediate states in multi-step processes when they are explicitly shown.\n"
    "15) Do not convert taxonomy, legend labels, scale hierarchies, type names, example lists, composition lists, or explanatory background into causal event sentences unless the source explicitly states an input-to-output transformation.\n"
    "16) Do not add explanatory rewrites, scientific interpretation, or inferred causal links. Prefer omission over invention.\n"
    "17) If a candidate sentence would not preserve at least an explicit input and an explicit output entity, do not write it as an event sentence.\n"
    "18) Avoid pronouns and avoid referring back with it, they, these, those, former, latter, or similar shorthand when a precise noun phrase can be repeated.\n"
    "19) Do not use bullets, numbering, JSON, or slot labels inside the two sections.\n"
)
EVENT_CHAIN_VERBATIM_GUARD_V2_APPENDIX = (
    "\n\nAdditional writing constraints:\n"
    "1) Write the output in two sections exactly.\n"
    "2) First section header must be: Paragraph:\n"
    "3) Second section header must be: Event Sentences:\n"
    "4) In Paragraph, write only 1 to 2 short sentences with no background knowledge and no interpretation.\n"
    "5) In Event Sentences, write only direct extraction-friendly event skeletons. Use 6 to 18 short sentences, with exactly one atomic geomorphic event per sentence.\n"
    "6) Keep the original event order. Keep separate source cases separate. Do not merge different regions, stages, scales, pathways, directions, or example cases into one sentence.\n"
    "7) When the evidence clearly supports a process relation, prefer patterns such as: At or Within [location], [mechanism or process] utilizes [input], generating [output].\n"
    "8) Copy precise entity names, process names, quantities, units, thresholds, directional terms, and temporal expressions whenever supported.\n"
    "9) Repeat exact noun phrases instead of summarizing them. Avoid pronouns and avoid shorthand such as it, they, these, those, former, latter, or similar references.\n"
    "10) Do not replace precise entities with broad paraphrases or generic placeholders such as landform, feature, system, output, shape, interaction, profile, curve, water bodies, drainage everywhere, topographic domes, stable land surfaces, or internal drainage structures unless that exact phrase is explicitly written in the source.\n"
    "11) For quantitative figures, each sentence must preserve one explicit case with its original actor, direction, and exact values. Do not rewrite quantitative cases as comparisons, rankings, or trend summaries such as upper curve, lower curve, middle curve, highest, lowest, surpasses, greater than, less than, maximum, minimum, increase to, decrease to, or similar analytical wording unless those exact words are written in the source.\n"
    "12) If you cannot preserve an explicit output entity and the exact case-specific content, omit the sentence.\n"
    "13) Do not introduce new named locations, new materials, new transport modes, new processes, or new mechanism names that are not explicitly grounded in the source.\n"
    "14) Do not convert legend labels, taxonomies, scale hierarchies, example lists, composition lists, overview summaries, or explanatory background into causal event sentences unless the source explicitly states an input-to-output transformation.\n"
    "15) Do not collapse map-level regional examples into continental summaries. Prefer one region-specific sentence over one broad synthesis sentence.\n"
    "16) Do not skip important intermediate states in multi-step chains when they are explicitly shown.\n"
    "17) Prefer omission over invention. If a candidate sentence would require interpretation, interpolation, or comparison wording, leave it out.\n"
    "18) Do not use bullets, numbering, JSON, or slot labels inside the two sections.\n"
)
EVENT_CHAIN_SOFT_GUARD_APPENDIX = (
    "\n\nAdditional writing constraints:\n"
    "1) Write the output in two sections exactly.\n"
    "2) First section header must be: Paragraph:\n"
    "3) Second section header must be: Event Sentences:\n"
    "4) In Paragraph, write only 1 to 2 short sentences with high fidelity and no background knowledge.\n"
    "5) In Event Sentences, write only explicit event-supporting sentences. Prefer 4 to 12 short natural sentences, and use fewer when the figure supports fewer clear events.\n"
    "6) Keep the wording close to the source terminology. Prefer natural sentences, not rigid templates.\n"
    "7) Keep one explicit source case per sentence when the source presents separate regions, stages, scales, directions, or pathways separately.\n"
    "8) Only write an event sentence when the source clearly supports an input entity and an output entity or transformed state.\n"
    "9) Each event sentence must stay close to explicit source noun phrases. Do not create hidden intermediate carriers such as force, mass, residue, debris, depth change, profile change, or similar bridge states unless they are explicitly named in the source.\n"
    "10) Do not turn one real event into a multi-step causal chain by feeding one generated output into a new invented sentence unless the source explicitly states that next step.\n"
    "11) Preserve explicit location, time, direction, quantity, and mechanism phrases when they are clearly tied to the same event sentence.\n"
    "12) Do not write overview, taxonomy, legend, scale hierarchy, category, framework, composition list, or comparison sentences as event sentences.\n"
    "13) Do not rewrite exact cases as rankings, comparisons, or summaries such as highest, lowest, upper curve, lower curve, typical pattern, or similar analytical wording unless those exact words are explicitly written in the source.\n"
    "14) Do not merge distant stages, distant regions, or multiple alternative pathways into one sentence.\n"
    "15) Do not invent intermediate states, new mechanism names, or new generalized entities. Prefer omission over invention.\n"
    "16) Avoid pronouns and vague placeholders such as landform, feature, system, curve, profile, output, process, or structure when a more precise noun phrase is supported.\n"
    "17) Do not use bullets, numbering, JSON, or slot labels inside the two sections.\n"
)
QUANT_RELATION_PRESERVING_APPENDIX = (
    "\n\nAdditional writing constraints:\n"
    "1) Keep the output as natural continuous prose in a single paragraph; do not use bullets, slot labels, or JSON.\n"
    "2) Prioritize semantic fidelity over stylistic elegance; stay close to the original geomorphology terminology from the image and contextual snippets.\n"
    "3) Preserve each distinct process-form relation separately; do not compress several events into one vague summary sentence.\n"
    "4) Whenever supported, explicitly preserve the structure of location, mechanism or process, input material or landform, and resulting output landform or state.\n"
    "5) Preserve directional, spatial, temporal, and quantitative details when they appear in the image or snippets.\n"
    "6) Preserve technical trigger phrases such as utilizes, generating, characterized by, bounded by, composed of, containing, uplift, subsidence, erosion, deposition, and drainage when they are supported.\n"
    "7) Prefer repeating precise entity names over replacing them with pronouns or vague category words.\n"
    "8) Preserve all quantitative expressions exactly when supported, including area ranges, lifespan values, units, inequality signs, thresholds, and magnitude markers.\n"
    "9) If the image or contextual snippets mention expressions such as Area ..., Lifespan ..., km², years, >, <, >=, <=, or numeric ranges, keep them explicitly in the output rather than paraphrasing them.\n"
    "10) Do not replace precise quantitative expressions with vague summaries such as small, large, short-term, or long-term unless the precise expression is also preserved.\n"
    "11) Do not add unsupported facts.\n"
)

PRECISION_GUARD_APPENDIX = (
    "\n\nAdditional constraints:\n"
    "1) Describe only process relations explicitly supported by the figure, caption, or Related Text.\n"
    "2) Do not convert panel labels, legends, categories, leader lines, spatial adjacency, or visual similarity into transformation events.\n"
    "3) Write at most one distinct event per sentence.\n"
    "4) Each event sentence must contain an explicit source entity or state, resulting entity or state, and process or mechanism.\n"
    "5) Omit uncertain relations rather than inferring missing roles.\n"
    "6) Do not repeat equivalent relations.\n"
    "7) End the output only after a complete sentence.\n"
)

VISUAL_ANALYSIS_PROMPT = (
    "Inspect the scientific figure carefully and produce a compact visual reading note for a second model pass. "
    "Transcribe visible technical labels, numbers, units, stage names, locations, and directional words as exactly "
    "as possible. Then state only arrow- or panel-supported mappings between source entities, processes, and result "
    "entities. Keep uncertain text explicitly marked uncertain. Do not add background knowledge, infer hidden causal "
    "links, or write the final descriptive paragraph. The paper caption and context are included below only to resolve "
    "visible abbreviations.\n\n"
)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str, data):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def tokenize_words(text: str) -> List[str]:
    return WORD_PATTERN.findall((text or "").lower())


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, start=1):
            tmp = dp[j]
            if x == y:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[-1]


def rouge_l_score(pred: str, ref: str) -> Tuple[float, float, float]:
    pred_tokens = tokenize_words(pred)
    ref_tokens = tokenize_words(ref)
    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def resolve_image_path(image_root: str, rel_path: str) -> str:
    path = Path(image_root) / rel_path
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return str(path)


def load_samples(data_path: str, image_root: str) -> List[Dict]:
    data = load_json(data_path)
    samples = []
    for i, item in enumerate(data):
        image_rel = item.get("images", [None])[0]
        if image_rel is None:
            raise ValueError(f"Sample {i} does not have an image path.")
        messages = item["messages"]
        samples.append(
            {
                "index": item.get("index", i),
                "image_path": resolve_image_path(image_root, image_rel),
                "system": messages[0]["content"],
                "user": messages[1]["content"],
                "gold": messages[2]["content"],
            }
        )
    return samples


def _detect_vl_model_type(model_path: str) -> str:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    return str(getattr(config, "model_type", "") or "").lower()


def load_model(
    model_path: str,
    adapter_path: Optional[str],
    gpu_max_memory_gib: Optional[int] = None,
    cpu_max_memory_gib: int = 200,
    offload_folder: Optional[str] = None,
    image_max_pixels: Optional[int] = None,
):
    model_type = _detect_vl_model_type(model_path)
    if model_type == "mllama":
        if MllamaForConditionalGeneration is None:
            raise RuntimeError("Current transformers version does not provide MllamaForConditionalGeneration.")
        model_cls = MllamaForConditionalGeneration
    elif model_type in {"qwen2_5_vl", "qwen2_vl"}:
        if Qwen2_5_VLForConditionalGeneration is None:
            raise RuntimeError("Current transformers version does not provide Qwen2_5_VLForConditionalGeneration.")
        model_cls = Qwen2_5_VLForConditionalGeneration
    elif model_type == "internvl":
        if InternVLForConditionalGeneration is None:
            raise RuntimeError("Current transformers version does not provide InternVLForConditionalGeneration.")
        model_cls = InternVLForConditionalGeneration
    else:
        if AutoModelForVision2Seq is None:
            raise RuntimeError(f"Unsupported VL model_type={model_type!r} and AutoModelForVision2Seq is unavailable.")
        model_cls = AutoModelForVision2Seq

    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if gpu_max_memory_gib is not None:
        load_kwargs["max_memory"] = {0: f"{gpu_max_memory_gib}GiB", "cpu": f"{cpu_max_memory_gib}GiB"}
        if offload_folder:
            load_kwargs["offload_folder"] = offload_folder
            load_kwargs["offload_state_dict"] = True

    model = model_cls.from_pretrained(model_path, **load_kwargs)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if image_max_pixels is not None and hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = image_max_pixels
    return model, processor, model_type


def build_user_text(user_text: str, prompt_mode: str) -> str:
    cleaned = user_text.replace("<image>\n", "", 1).replace("<image>", "", 1).strip()
    if prompt_mode == "trained_paragraph_events":
        # val12 already stores the training-time prompt verbatim.  Frozen
        # test24 stores the older one-paragraph trailer; replace only that
        # fixed instruction so both splits use the same inference task while
        # preserving all sample-specific title/caption/context text unchanged.
        if "Return exactly two sections with these headers:" in cleaned:
            return cleaned
        base = OLD_DESCRIPTION_TRAILER_PATTERN.sub("", cleaned).rstrip()
        return base + TRAINED_PARAGRAPH_EVENTS_SUFFIX
    if prompt_mode == "trained_paragraph_events_mechanism_gate":
        base = re.sub(
            r"\n\nReturn exactly two sections with these headers:.*\Z",
            "",
            cleaned,
            flags=re.S,
        ).rstrip()
        return base + TRAINED_PARAGRAPH_EVENTS_MECHANISM_GATE_SUFFIX
    if prompt_mode == "conservative":
        return cleaned + CONSERVATIVE_APPENDIX
    if prompt_mode == "conservative_balanced":
        return cleaned + CONSERVATIVE_BALANCED_APPENDIX
    if prompt_mode == "paragraph_events":
        return cleaned + PARAGRAPH_EVENTS_APPENDIX
    if prompt_mode == "relation_preserving":
        return cleaned + RELATION_PRESERVING_APPENDIX
    if prompt_mode == "quant_relation_preserving":
        return cleaned + QUANT_RELATION_PRESERVING_APPENDIX
    if prompt_mode == "precision_guard":
        return cleaned + PRECISION_GUARD_APPENDIX
    if prompt_mode == "relation_preserving_events":
        return cleaned + RELATION_PRESERVING_EVENTS_APPENDIX
    if prompt_mode == "event_chain_strict":
        return cleaned + EVENT_CHAIN_STRICT_APPENDIX
    if prompt_mode == "event_chain_precision_guard":
        return cleaned + EVENT_CHAIN_PRECISION_GUARD_APPENDIX
    if prompt_mode == "event_chain_stage2_guard":
        return cleaned + EVENT_CHAIN_STAGE2_GUARD_APPENDIX
    if prompt_mode == "event_chain_verbatim_guard":
        return cleaned + EVENT_CHAIN_VERBATIM_GUARD_APPENDIX
    if prompt_mode == "event_chain_verbatim_guard_v2":
        return cleaned + EVENT_CHAIN_VERBATIM_GUARD_V2_APPENDIX
    if prompt_mode == "event_chain_soft_guard":
        return cleaned + EVENT_CHAIN_SOFT_GUARD_APPENDIX
    return cleaned


def split_paragraph_events(pred_text: str) -> Tuple[str, str]:
    paragraph_match = PARAGRAPH_SECTION_PATTERN.search(pred_text or "")
    event_match = EVENT_SECTION_PATTERN.search(pred_text or "")
    paragraph = paragraph_match.group(1).strip() if paragraph_match else (pred_text or "").strip()
    if event_match:
        raw_event_text = event_match.group(1).strip()
        event_lines = [line.strip(" -\t") for line in raw_event_text.splitlines() if line.strip()]
        event_text = "\n".join(event_lines).strip()
    else:
        event_text = (pred_text or "").strip()
    return paragraph, event_text


def get_eval_text(pred_text: str, prompt_mode: str) -> str:
    if prompt_mode in {"trained_paragraph_events", "trained_paragraph_events_mechanism_gate", "paragraph_events", "relation_preserving_events", "event_chain_strict", "event_chain_precision_guard", "event_chain_stage2_guard", "event_chain_verbatim_guard", "event_chain_verbatim_guard_v2", "event_chain_soft_guard"}:
        paragraph, _ = split_paragraph_events(pred_text)
        return paragraph
    return pred_text


def get_stage2_input_text(pred_text: str, prompt_mode: str) -> str:
    if prompt_mode in {"trained_paragraph_events", "trained_paragraph_events_mechanism_gate"}:
        if not EVENT_SECTION_PATTERN.search(pred_text or ""):
            return ""
        _, event_text = split_paragraph_events(pred_text)
        return "" if event_text.casefold() in {"none", "none."} else event_text
    if prompt_mode in {"trained_paragraph_events", "paragraph_events", "relation_preserving_events", "event_chain_strict", "event_chain_precision_guard", "event_chain_stage2_guard", "event_chain_verbatim_guard", "event_chain_verbatim_guard_v2", "event_chain_soft_guard"}:
        _, event_text = split_paragraph_events(pred_text)
        return event_text
    return pred_text


def run_inference(
    samples: List[Dict],
    model_path: str,
    adapter_path: Optional[str],
    max_new_tokens: int,
    temperature: float,
    repetition_penalty: float,
    prompt_mode: str,
    gpu_max_memory_gib: Optional[int],
    cpu_max_memory_gib: int,
    offload_folder: Optional[str],
    image_max_pixels: Optional[int],
    visual_analysis_first_pass: bool,
    visual_analysis_max_new_tokens: int,
    structured_generation_guard: bool,
    paragraph_max_sentences: int,
    event_max_sentences: int,
    guard_retry_max_new_tokens: int,
) -> Tuple[List[str], List[Optional[str]], List[Optional[Dict]]]:
    model, processor, model_type = load_model(
        model_path,
        adapter_path,
        gpu_max_memory_gib=gpu_max_memory_gib,
        cpu_max_memory_gib=cpu_max_memory_gib,
        offload_folder=offload_folder,
        image_max_pixels=image_max_pixels,
    )
    first_param = next(model.parameters())
    target_device = first_param.device

    def move_inputs(batch):
        return {
            key: value.to(
                device=target_device,
                dtype=first_param.dtype if torch.is_floating_point(value) else value.dtype,
            )
            for key, value in batch.items()
        }

    effective_repetition_penalty = repetition_penalty
    if model_type == "mllama" and repetition_penalty not in (None, 1.0):
        print("[WARN] mllama image generation is incompatible with repetition_penalty; falling back to 1.0")
        effective_repetition_penalty = 1.0

    if temperature <= 0:
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
    else:
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": 0.9,
        }
    if effective_repetition_penalty not in (None, 1.0):
        gen_kwargs["repetition_penalty"] = effective_repetition_penalty

    outputs = []
    visual_analyses: List[Optional[str]] = []
    guard_diagnostics: List[Optional[Dict]] = []
    for i, sample in enumerate(samples):
        image = Image.open(sample["image_path"]).convert("RGB")
        visual_analysis = None
        if visual_analysis_first_pass:
            analysis_messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a precise scientific diagram reader."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": VISUAL_ANALYSIS_PROMPT + sample["user"]},
                    ],
                },
            ]
            analysis_text = processor.apply_chat_template(
                analysis_messages, tokenize=False, add_generation_prompt=True
            )
            analysis_inputs = processor(text=[analysis_text], images=[image], return_tensors="pt", padding=True)
            analysis_inputs = move_inputs(analysis_inputs)
            analysis_context = model.disable_adapter() if adapter_path and hasattr(model, "disable_adapter") else nullcontext()
            with analysis_context, torch.no_grad():
                analysis_generated = model.generate(
                    **analysis_inputs,
                    max_new_tokens=visual_analysis_max_new_tokens,
                    do_sample=False,
                    repetition_penalty=effective_repetition_penalty,
                )
            analysis_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(analysis_inputs["input_ids"], analysis_generated)
            ]
            visual_analysis = processor.batch_decode(
                analysis_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            del analysis_inputs, analysis_generated, analysis_trimmed

        user_text = build_user_text(sample["user"], prompt_mode)
        if visual_analysis:
            user_text += (
                "\n\nAutomatically generated visual reading note (may contain OCR errors; verify every item against the image "
                "and omit unsupported items):\n" + visual_analysis
            )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": sample["system"]}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inputs = move_inputs(inputs)

        primary_guard = None
        guarded_gen_kwargs = dict(gen_kwargs)
        if structured_generation_guard:
            primary_guard = ParagraphEventsStoppingCriteria(
                tokenizer=processor.tokenizer,
                prompt_length=inputs["input_ids"].shape[1],
                paragraph_max_sentences=paragraph_max_sentences,
                event_max_sentences=event_max_sentences,
            )
            guarded_gen_kwargs["stopping_criteria"] = StoppingCriteriaList([primary_guard])

        with torch.no_grad():
            generated = model.generate(**inputs, **guarded_gen_kwargs)

        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated)]
        pred_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        guard_diagnostic = None
        if structured_generation_guard:
            retry_text = None
            retry_stop_reason = None
            if needs_event_retry(pred_text):
                recovery_user_text = (
                    user_text
                    + "\n\nFormat-recovery pass: the previous response did not finish the required event section. "
                    "Output only the header Event Sentences: followed by at most "
                    f"{event_max_sentences} distinct, complete, source-grounded atomic event sentences, one per line. "
                    "Do not repeat sentences, add a Paragraph section, use bullets, or add commentary."
                )
                recovery_messages = [
                    {"role": "system", "content": [{"type": "text", "text": sample["system"]}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": recovery_user_text},
                        ],
                    },
                ]
                recovery_prompt = processor.apply_chat_template(
                    recovery_messages, tokenize=False, add_generation_prompt=True
                )
                recovery_inputs = processor(
                    text=[recovery_prompt], images=[image], return_tensors="pt", padding=True
                )
                recovery_inputs = move_inputs(recovery_inputs)
                retry_guard = ParagraphEventsStoppingCriteria(
                    tokenizer=processor.tokenizer,
                    prompt_length=recovery_inputs["input_ids"].shape[1],
                    paragraph_max_sentences=paragraph_max_sentences,
                    event_max_sentences=event_max_sentences,
                    event_only=True,
                )
                recovery_gen_kwargs = dict(gen_kwargs)
                recovery_gen_kwargs["max_new_tokens"] = guard_retry_max_new_tokens
                recovery_gen_kwargs["stopping_criteria"] = StoppingCriteriaList([retry_guard])
                with torch.no_grad():
                    recovery_generated = model.generate(**recovery_inputs, **recovery_gen_kwargs)
                recovery_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(recovery_inputs["input_ids"], recovery_generated)
                ]
                retry_text = processor.batch_decode(
                    recovery_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
                retry_stop_reason = retry_guard.stop_reason
                del recovery_inputs, recovery_generated, recovery_trimmed

            protected_text, guard_diagnostic = protect_output(
                primary_text=pred_text,
                retry_text=retry_text,
                paragraph_max_sentences=paragraph_max_sentences,
                event_max_sentences=event_max_sentences,
            )
            guard_diagnostic.update(
                {
                    "primary_stop_reason": primary_guard.stop_reason,
                    "retry_used": retry_text is not None,
                    "retry_stop_reason": retry_stop_reason,
                    "raw_primary_prediction": pred_text,
                    "raw_retry_prediction": retry_text,
                }
            )
            pred_text = protected_text

        outputs.append(pred_text)
        visual_analyses.append(visual_analysis)
        guard_diagnostics.append(guard_diagnostic)
        del inputs, generated, trimmed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (i + 1) % 1 == 0:
            print(f"[INFO] Inference progress: {i + 1}/{len(samples)}")
    return outputs, visual_analyses, guard_diagnostics


def evaluate_predictions(
    preds: List[str],
    refs: List[str],
    bert_model_path: str,
    batch_size: int,
    bert_num_layers: Optional[int],
) -> Dict:
    # BERTScore is needed only for metric evaluation, not for inference helpers.
    from bert_score import score as bert_score

    if len(preds) != len(refs):
        raise ValueError("preds and refs must have the same length.")

    smoothie = SmoothingFunction().method1
    bleu = corpus_bleu(
        [[tokenize_words(ref)] for ref in refs],
        [tokenize_words(pred) for pred in preds],
        smoothing_function=smoothie,
        weights=(0.25, 0.25, 0.25, 0.25),
    )

    rouge_p, rouge_r, rouge_f = [], [], []
    for pred, ref in zip(preds, refs):
        p, r, f1 = rouge_l_score(pred, ref)
        rouge_p.append(p)
        rouge_r.append(r)
        rouge_f.append(f1)

    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
    max_positions = getattr(bert_tokenizer, "model_max_length", 512)
    if not isinstance(max_positions, int) or max_positions <= 0 or max_positions > 100000:
        max_positions = 512
    max_positions = min(max_positions, 512)

    def truncate_for_bert(text: str) -> str:
        encoded = bert_tokenizer(
            text,
            truncation=True,
            max_length=max_positions,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return bert_tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)

    bert_preds = [truncate_for_bert(pred) for pred in preds]
    bert_refs = [truncate_for_bert(ref) for ref in refs]

    P, R, F1 = bert_score(
        bert_preds,
        bert_refs,
        model_type=bert_model_path,
        num_layers=bert_num_layers,
        lang="en",
        batch_size=batch_size,
        verbose=True,
    )

    return {
        "num_samples": len(preds),
        "bleu4": float(bleu),
        "rouge_l": {
            "precision": float(sum(rouge_p) / len(rouge_p)),
            "recall": float(sum(rouge_r) / len(rouge_r)),
            "f1": float(sum(rouge_f) / len(rouge_f)),
        },
        "bertscore": {
            "precision": float(P.mean().item()),
            "recall": float(R.mean().item()),
            "f1": float(F1.mean().item()),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate stage1 description generation on desc_fold0_test.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--pred-path", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--bert-model-path", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by sampled generation. Deterministic decoding is unaffected.",
    )
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--bert-batch-size", type=int, default=8)
    parser.add_argument("--bert-num-layers", type=int, default=12)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=[],
        help="Optional positional dataset rows for a diagnostic run.",
    )
    parser.add_argument("--gpu-max-memory-gib", type=int, default=None)
    parser.add_argument("--cpu-max-memory-gib", type=int, default=200)
    parser.add_argument("--offload-folder", type=str, default=None)
    parser.add_argument(
        "--image-max-pixels",
        type=int,
        default=None,
        help="Cap image preprocessing pixels; use the same value as the training YAML.",
    )
    parser.add_argument(
        "--visual-analysis-first-pass",
        action="store_true",
        help="Run a base-model visual label/arrow reading pass before adapter description generation.",
    )
    parser.add_argument("--visual-analysis-max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--structured-generation-guard",
        action="store_true",
        help="Stop repeated complete sentences, cap sections, and recover a missing Event Sentences section.",
    )
    parser.add_argument("--paragraph-max-sentences", type=int, default=4)
    parser.add_argument("--event-max-sentences", type=int, default=12)
    parser.add_argument("--guard-retry-max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--prompt-mode",
        choices=[
            "original",
            "trained_paragraph_events",
            "trained_paragraph_events_mechanism_gate",
            "conservative",
            "conservative_balanced",
            "paragraph_events",
            "relation_preserving",
            "quant_relation_preserving",
            "precision_guard",
            "relation_preserving_events",
            "event_chain_strict",
            "event_chain_precision_guard",
            "event_chain_stage2_guard",
            "event_chain_verbatim_guard",
            "event_chain_verbatim_guard_v2",
            "event_chain_soft_guard",
        ],
        default="original",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    samples = load_samples(args.data_path, args.image_root)
    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]
    if args.indices:
        wanted = set(args.indices)
        selected = [sample for ordinal, sample in enumerate(samples) if ordinal in wanted]
        missing = wanted - set(range(len(samples)))
        if missing:
            raise ValueError(f"requested indices are unavailable: {sorted(missing)}")
        samples = selected
    if not samples:
        raise ValueError("no samples selected")
    preds, visual_analyses, guard_diagnostics = run_inference(
        samples=samples,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        prompt_mode=args.prompt_mode,
        gpu_max_memory_gib=args.gpu_max_memory_gib,
        cpu_max_memory_gib=args.cpu_max_memory_gib,
        offload_folder=args.offload_folder,
        image_max_pixels=args.image_max_pixels,
        visual_analysis_first_pass=args.visual_analysis_first_pass,
        visual_analysis_max_new_tokens=args.visual_analysis_max_new_tokens,
        structured_generation_guard=args.structured_generation_guard,
        paragraph_max_sentences=args.paragraph_max_sentences,
        event_max_sentences=args.event_max_sentences,
        guard_retry_max_new_tokens=args.guard_retry_max_new_tokens,
    )
    eval_preds = [get_eval_text(pred, args.prompt_mode) for pred in preds]
    refs = [get_eval_text(sample["gold"], args.prompt_mode) for sample in samples]
    metrics = evaluate_predictions(eval_preds, refs, args.bert_model_path, args.bert_batch_size, args.bert_num_layers)

    pred_rows = []
    for sample, pred, visual_analysis, guard_diagnostic in zip(
        samples, preds, visual_analyses, guard_diagnostics
    ):
        pred_rows.append(
            {
                "index": sample["index"],
                "image_path": sample["image_path"],
                "prediction": pred,
                "visual_analysis": visual_analysis,
                "generation_guard": guard_diagnostic,
                "evaluation_text": get_eval_text(pred, args.prompt_mode),
                "stage2_input_text": get_stage2_input_text(pred, args.prompt_mode),
                "gold": sample["gold"],
            }
        )

    dump_json(args.pred_path, pred_rows)
    dump_json(
        args.result_path,
        {
            "model_path": args.model_path,
            "adapter_path": args.adapter_path,
            "bert_model_path": args.bert_model_path,
            "prompt_mode": args.prompt_mode,
            "image_max_pixels": args.image_max_pixels,
            "visual_analysis_first_pass": args.visual_analysis_first_pass,
            "visual_analysis_max_new_tokens": args.visual_analysis_max_new_tokens,
            "structured_generation_guard": args.structured_generation_guard,
            "paragraph_max_sentences": args.paragraph_max_sentences,
            "event_max_sentences": args.event_max_sentences,
            "guard_retry_max_new_tokens": args.guard_retry_max_new_tokens,
            "seed": args.seed,
            **metrics,
        },
    )

    print(f"[INFO] Predictions saved to: {args.pred_path}")
    print("\n===== Description Evaluation Summary =====")
    print(f"Samples: {metrics['num_samples']}")
    print(f"BLEU-4: {metrics['bleu4']:.4f}")
    print(f"ROUGE-L F1: {metrics['rouge_l']['f1']:.4f}")
    print(f"BERTScore F1: {metrics['bertscore']['f1']:.4f}")
    print(f"Results saved to: {args.result_path}")


if __name__ == "__main__":
    main()
