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
