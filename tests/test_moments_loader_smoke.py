import csv
import os
import sys
import types
import tempfile
from pathlib import Path


def _install_minimal_shims():
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        torch.Tensor = type("Tensor", (), {})
        torch.dtype = type("dtype", (), {})
        torch.device = type("device", (), {})
        torch.manual_seed = lambda *args, **kwargs: None
        torch.cuda = types.SimpleNamespace(
            manual_seed=lambda *a, **k: None,
            manual_seed_all=lambda *a, **k: None,
        )
        torch.backends = types.SimpleNamespace(
            cudnn=types.SimpleNamespace(deterministic=False, benchmark=False)
        )
        sys.modules["torch"] = torch
    if "transformer_lens" not in sys.modules:
        tl = types.ModuleType("transformer_lens")
        tl.utils = types.SimpleNamespace(
            get_act_name=lambda name, layer=None: f"blocks.{layer}.{name}"
        )
        tl.HookedVLTransformer = type("HookedVLTransformer", (), {})
        tl.HookedTransformer = type("HookedTransformer", (), {})
        sys.modules["transformer_lens"] = tl


def test_moments_loader_smoke():
    _install_minimal_shims()

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.append(str(repo_root / "reproducing_code" / "vlm-circuits-analysis"))

    from moments_utils import load_moments_vl_prompts_list

    sample_frames = (
        repo_root
        / "thesis_project"
        / "data"
        / "MOMENTS_frames"
        / "frames"
        / "0jJj5Mme"
        / "im"
        / "1"
        / "IM_1"
    )
    frame_paths = sorted(str(p.resolve()) for p in sample_frames.glob("frame_*.png"))
    assert len(frame_paths) == 10

    transcript_path = (
        repo_root
        / "thesis_project"
        / "data"
        / "MOMENTS"
        / "0Glu8uEj"
        / "important-moments"
        / "1"
        / "IM_1_v1.json"
    )
    assert transcript_path.exists()

    fd, csv_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "clip_id",
                    "group_idx",
                    "clip_name",
                    "event_type",
                    "label",
                    "similarity",
                    "local_text",
                    "global_text",
                    "prompt",
                    "image_paths",
                    "answer",
                    "cf_mode",
                    "cf_prompt",
                    "cf_image_paths",
                    "cf_answer",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "clip_id": "0jJj5Mme",
                    "group_idx": "1",
                    "clip_name": "IM_1",
                    "event_type": "goal",
                    "label": "goal",
                    "similarity": "0.98",
                    "local_text": "Brilliant from Jesus, but what a finish that is.",
                    "global_text": "Brilliant from Jesus but what a finish that is",
                    "prompt": "You are watching a football match. Here are 10 frames from a short clip. Is this a goal? Answer yes or no.",
                    "image_paths": "|".join(frame_paths),
                    "answer": "yes",
                    "cf_mode": "vision_only",
                    "cf_prompt": "You are watching a football match. Here are 10 frames from a short clip. Is this a goal? Answer yes or no.",
                    "cf_image_paths": "",
                    "cf_answer": "no",
                }
            )

        class DummyModel:
            model_name = "llava-1.5"

        prompts = load_moments_vl_prompts_list(
            csv_path, model=DummyModel(), language_only=False
        )
        assert len(prompts) == 1
        prompt = prompts[0]
        assert len(prompt.images) == 10
        assert prompt.images[0].size == (252, 252)
        assert prompt.answer == "yes"
        assert prompt.cf_answer == "no"
        assert prompt.metadata["clip_id"] == "0jJj5Mme"
    finally:
        os.remove(csv_path)


def test_moments_random_pair_length_filter_drops_mismatch():
    _install_minimal_shims()

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.append(str(repo_root / "reproducing_code" / "vlm-circuits-analysis"))

    from moments_utils import load_moments_vl_prompts_list

    sample_frames = (
        repo_root
        / "thesis_project"
        / "data"
        / "MOMENTS_frames"
        / "frames"
        / "0jJj5Mme"
        / "im"
        / "1"
        / "IM_1"
    )
    frame_paths = sorted(str(p.resolve()) for p in sample_frames.glob("frame_*.png"))
    assert len(frame_paths) == 10

    fd, csv_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "clip_id",
                    "group_idx",
                    "clip_name",
                    "event_type",
                    "label",
                    "similarity",
                    "local_text",
                    "global_text",
                    "prompt",
                    "image_paths",
                    "answer",
                    "cf_mode",
                    "cf_prompt",
                    "cf_image_paths",
                    "cf_answer",
                    "cf_prompt_state",
                    "cf_prompt_changes",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "clip_id": "clip_a",
                    "group_idx": "1",
                    "clip_name": "IM_1",
                    "event_type": "goal",
                    "label": "goal",
                    "similarity": "0.98",
                    "local_text": "one two three",
                    "global_text": "one two three",
                    "prompt": "one two three",
                    "image_paths": "|".join(frame_paths[:1]),
                    "answer": "yes",
                    "cf_mode": "random_pair",
                    "cf_prompt": "one two three four five six",
                    "cf_image_paths": "|".join(frame_paths[:1]),
                    "cf_answer": "no",
                    "cf_prompt_state": "random_pair",
                    "cf_prompt_changes": "paired_with:clip_b__1__IM_2",
                }
            )
            writer.writerow(
                {
                    "clip_id": "clip_b",
                    "group_idx": "1",
                    "clip_name": "IM_2",
                    "event_type": "goal",
                    "label": "goal",
                    "similarity": "0.98",
                    "local_text": "one two three",
                    "global_text": "one two three",
                    "prompt": "one two three four",
                    "image_paths": "|".join(frame_paths[:1]),
                    "answer": "no",
                    "cf_mode": "random_pair",
                    "cf_prompt": "one two three four",
                    "cf_image_paths": "|".join(frame_paths[:1]),
                    "cf_answer": "yes",
                    "cf_prompt_state": "random_pair",
                    "cf_prompt_changes": "paired_with:clip_a__1__IM_1",
                }
            )

        class DummyModel:
            model_name = "llava-1.5"

            def to_tokens(self, prompt, images=None, prepend_bos=False, truncate=True):
                length = len(str(prompt).split()) + (len(images) if images else 0)
                return types.SimpleNamespace(shape=(1, length))

        prompts = load_moments_vl_prompts_list(
            csv_path, model=DummyModel(), language_only=False
        )
        assert len(prompts) == 1
        assert prompts[0].metadata["clip_id"] == "clip_b"
        assert prompts[0].metadata["prompt_token_length"] == prompts[0].metadata["cf_prompt_token_length"]
    finally:
        os.remove(csv_path)


def test_moments_random_pair_builder_same_length_only():
    _install_minimal_shims()

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.append(str(repo_root / "thesis_project" / "data_generation"))

    from create_moments_dataset import build_random_pair_rows

    clean_records = [
        {
            "clip_id": "clip_a",
            "group_idx": "1",
            "clip_name": "IM_1",
            "event_type": "goal",
            "label": "goal",
            "similarity": "1.0",
            "local_text": "local a",
            "global_text": "global a",
            "answer": "yes",
            "_sample_id": "clip_a__1__IM_1",
            "_pairing_token_length": 128,
            "_pairing_heuristic_length": 3,
            "prompt": "a",
            "image_paths": "a.png",
        },
        {
            "clip_id": "clip_b",
            "group_idx": "1",
            "clip_name": "IM_2",
            "event_type": "goal",
            "label": "goal",
            "similarity": "1.0",
            "local_text": "local b",
            "global_text": "global b",
            "answer": "no",
            "_sample_id": "clip_b__1__IM_2",
            "_pairing_token_length": 128,
            "_pairing_heuristic_length": 3,
            "prompt": "b",
            "image_paths": "b.png",
        },
        {
            "clip_id": "clip_c",
            "group_idx": "1",
            "clip_name": "IM_3",
            "event_type": "goal",
            "label": "goal",
            "similarity": "1.0",
            "local_text": "local c",
            "global_text": "global c",
            "answer": "no",
            "_sample_id": "clip_c__1__IM_3",
            "_pairing_token_length": 64,
            "_pairing_heuristic_length": 6,
            "prompt": "c",
            "image_paths": "c.png",
        },
    ]

    rows, stats = build_random_pair_rows(
        task="goal",
        clean_records=clean_records,
        length_mode="qwen",
    )
    assert len(rows) == 2
    assert {row["clip_id"] for row in rows} == {"clip_a", "clip_b"}
    assert {row["cf_prompt_changes"] for row in rows} == {
        "paired_with:clip_b__1__IM_2",
        "paired_with:clip_a__1__IM_1",
    }
    assert stats["kept"] == 2
    assert stats["dropped_no_candidate"] == 1


def test_moments_random_pair_builder_heuristic_length_mode():
    _install_minimal_shims()

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.append(str(repo_root / "thesis_project" / "data_generation"))

    from create_moments_dataset import build_random_pair_rows

    clean_records = [
        {
            "clip_id": "clip_a",
            "group_idx": "1",
            "clip_name": "IM_1",
            "event_type": "goal",
            "label": "goal",
            "similarity": "1.0",
            "local_text": "local a",
            "global_text": "global a",
            "answer": "yes",
            "_sample_id": "clip_a__1__IM_1",
            "_pairing_heuristic_length": 4,
            "prompt": "one two three four",
            "image_paths": "a.png",
        },
        {
            "clip_id": "clip_b",
            "group_idx": "1",
            "clip_name": "IM_2",
            "event_type": "goal",
            "label": "goal",
            "similarity": "1.0",
            "local_text": "local b",
            "global_text": "global b",
            "answer": "no",
            "_sample_id": "clip_b__1__IM_2",
            "_pairing_heuristic_length": 4,
            "prompt": "five six seven eight",
            "image_paths": "b.png",
        },
        {
            "clip_id": "clip_c",
            "group_idx": "1",
            "clip_name": "IM_3",
            "event_type": "goal",
            "label": "goal",
            "similarity": "1.0",
            "local_text": "local c",
            "global_text": "global c",
            "answer": "no",
            "_sample_id": "clip_c__1__IM_3",
            "_pairing_heuristic_length": 8,
            "prompt": "nine ten eleven twelve thirteen fourteen fifteen sixteen",
            "image_paths": "c.png",
        },
    ]

    rows, stats = build_random_pair_rows(
        task="goal",
        clean_records=clean_records,
        length_mode="heuristic",
    )
    assert len(rows) == 2
    assert {row["clip_id"] for row in rows} == {"clip_a", "clip_b"}
    assert stats["kept"] == 2
    assert stats["dropped_no_candidate"] == 1
