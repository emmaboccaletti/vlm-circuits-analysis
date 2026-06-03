import csv
import os
import sys
from typing import Dict, List, Optional

from PIL import Image

from vision_language_prompts import VLPrompt

sys.path.append("./third_party/TransformerLens")

import transformer_lens as lens

from general_utils import (
    get_image_size_for_model,
    get_single_token_tokens,
    load_image_for_model,
)


MOMENTS_CF_MODES = {"random_pair", "language_only", "vision_only", "both"}
MOMENTS_LABELS = ["yes", "no"]
MOMENTS_EVENT_TYPE_LABELS = ["goal", "corner", "shot"]


def _split_paths(field: str) -> List[str]:
    if field is None:
        return []
    field = field.strip()
    if not field:
        return []
    return [p for p in field.split("|") if p]


def _resolve_paths(image_paths: List[str], data_csv_path: str) -> List[str]:
    data_csv_dir = os.path.dirname(data_csv_path)
    resolved = []
    for path in image_paths:
        if os.path.isabs(path):
            resolved.append(path)
        else:
            resolved.append(os.path.join(data_csv_dir, path))
    return resolved


def _load_image_sequence(
    image_paths: List[str],
    model_name: str,
    target_size: Optional[tuple] = None,
) -> List[Image.Image]:
    return [load_image_for_model(path, model_name, target_size=target_size) for path in image_paths]


def load_moments_vl_prompts_list(
    data_csv_path: str,
    model: Optional[lens.HookedVLTransformer] = None,
    processor=None,
    language_only: bool = False,
    correct_preds_only: bool = True,
    image_size=None,
) -> List[VLPrompt]:
    """
    Load MOMENTS prompts from a CSV produced by create_moments_dataset.py.

    For MOMENTS we do not rely on model-generated predictions, so
    `correct_preds_only` is intentionally ignored. The argument is kept for API
    compatibility with the other task loaders.
    """
    if not os.path.exists(data_csv_path):
        raise FileNotFoundError(f"Missing MOMENTS CSV: {data_csv_path}")

    if model is None:
        raise ValueError("model must be provided to load MOMENTS data")

    if image_size is None:
        image_size = get_image_size_for_model(model.model_name.lower())

    vl_prompts: List[VLPrompt] = []
    with open(data_csv_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            prompt = row["prompt"]
            answer = row["answer"]
            cf_prompt = row.get("cf_prompt") or None
            cf_answer = row.get("cf_answer") or None
            image_paths = _resolve_paths(
                _split_paths(row.get("image_paths", "")), data_csv_path
            )
            cf_image_paths = _resolve_paths(
                _split_paths(row.get("cf_image_paths", "")), data_csv_path
            )

            images = []
            cf_images = None
            if not language_only:
                images = _load_image_sequence(
                    image_paths, model.model_name, target_size=image_size
                )
                if cf_image_paths:
                    cf_images = _load_image_sequence(
                        cf_image_paths, model.model_name, target_size=image_size
                    )
            else:
                images = []
                if cf_image_paths:
                    cf_images = []

            metadata = {
                "clip_id": row.get("clip_id"),
                "group_idx": row.get("group_idx"),
                "clip_name": row.get("clip_name"),
                "event_type": row.get("event_type"),
                "label": row.get("label"),
                "similarity": row.get("similarity"),
                "local_text": row.get("local_text"),
                "global_text": row.get("global_text"),
                "cf_mode": row.get("cf_mode"),
                "cf_prompt_state": row.get("cf_prompt_state"),
                "cf_prompt_changes": row.get("cf_prompt_changes"),
                "image_paths": image_paths,
                "cf_image_paths": cf_image_paths,
            }

            vl_prompts.append(
                VLPrompt(
                    prompt,
                    images,
                    answer,
                    cf_prompt=cf_prompt,
                    cf_images=cf_images,
                    cf_answer=cf_answer,
                    metadata=metadata,
                )
            )
    return vl_prompts


def load_moments_parallel_l_prompts(
    vl_prompts: List[VLPrompt], processor
) -> List[VLPrompt]:
    """
    Create textual-only counterparts for MOMENTS prompts by removing images.
    """
    parallel_l_prompts: List[VLPrompt] = []
    for vl_prompt in vl_prompts:
        parallel_l_prompts.append(
            VLPrompt(
                vl_prompt.prompt,
                [],
                str(vl_prompt.answer),
                cf_prompt=vl_prompt.cf_prompt,
                cf_images=[] if vl_prompt.cf_prompt is not None else None,
                cf_answer=vl_prompt.cf_answer,
                metadata=vl_prompt.metadata,
            )
        )
    return parallel_l_prompts


def get_moments_limited_labels(processor):
    """
    Return token strings for the yes/no labels used by MOMENTS.
    """
    labels = get_single_token_tokens(
        processor,
        ["yes", "no", " yes", " no", "Yes", "No"],
    )
    if len(labels) < 2:
        raise ValueError(
            "Could not find single-token yes/no labels for the MOMENTS task."
        )
    # Preserve the canonical yes/no order if possible.
    label_set = set(labels)
    ordered = [label for label in ["yes", "no", " yes", " no", "Yes", "No"] if label in label_set]
    return ordered[:2]


def get_moments_event_type_limited_labels(processor):
    """
    Return token strings for the MOMENTS event-type labels.
    """
    labels = get_single_token_tokens(
        processor,
        ["goal", "corner", "shot", " goal", " corner", " shot", "Goal", "Corner", "Shot"],
    )
    if len(labels) < 3:
        raise ValueError(
            "Could not find single-token goal/corner/shot labels for the MOMENTS event-type task."
        )
    label_set = set(labels)
    ordered = [
        label
        for label in ["goal", "corner", "shot", " goal", " corner", " shot", "Goal", "Corner", "Shot"]
        if label in label_set
    ]
    # Preserve canonical order if possible.
    canonical = []
    seen = set()
    for candidate in ordered:
        lowered = candidate.strip().lower()
        if lowered not in seen:
            seen.add(lowered)
            canonical.append(candidate)
    return canonical[:3]
