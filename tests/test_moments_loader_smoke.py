import csv
import os
import sys
import types
import tempfile
from pathlib import Path
from unittest.mock import patch


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


def test_moments_language_length_filter_drops_mismatch():
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
                    "clip_id", "group_idx", "clip_name", "event_type", "label",
                    "similarity", "local_text", "global_text", "prompt",
                    "image_paths", "answer", "cf_mode", "cf_prompt",
                    "cf_image_paths", "cf_answer",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "clip_id": "clip_a", "group_idx": "1", "clip_name": "IM_1",
                "event_type": "goal", "label": "goal", "similarity": "1.0",
                "local_text": "one two", "global_text": "one two",
                "prompt": "one two", "image_paths": "|".join(frame_paths),
                "answer": "yes", "cf_mode": "language_only",
                "cf_prompt": "one two three", "cf_image_paths": "",
                "cf_answer": "yes",
            })

        class DummyModel:
            model_name = "llava-1.5"

            def to_tokens(self, prompt, images=None, prepend_bos=False, truncate=True):
                length = len(str(prompt).split()) + (len(images) if images else 0)
                return types.SimpleNamespace(shape=(1, length))

        prompts = load_moments_vl_prompts_list(
            csv_path, model=DummyModel(), language_only=True
        )
        assert prompts == []
    finally:
        os.remove(csv_path)


def test_moments_vision_length_filter_drops_mismatch():
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
                    "clip_id", "group_idx", "clip_name", "event_type", "label",
                    "similarity", "local_text", "global_text", "prompt",
                    "image_paths", "answer", "cf_mode", "cf_prompt",
                    "cf_image_paths", "cf_answer",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "clip_id": "clip_a", "group_idx": "1", "clip_name": "IM_1",
                "event_type": "goal", "label": "goal", "similarity": "1.0",
                "local_text": "one two", "global_text": "one two",
                "prompt": "one two", "image_paths": "|".join(frame_paths),
                "answer": "yes", "cf_mode": "vision_only",
                "cf_prompt": "one two", "cf_image_paths": "|".join(frame_paths),
                "cf_answer": "no",
            })

        class DummyModel:
            model_name = "llava-1.5"

            def __init__(self):
                self.calls = 0

            def to_tokens(self, prompt, images=None, prepend_bos=False, truncate=True):
                length = len(str(prompt).split()) + (len(images) if images else 0)
                self.calls += 1
                length += self.calls % 2
                return types.SimpleNamespace(shape=(1, length))

        prompts = load_moments_vl_prompts_list(
            csv_path, model=DummyModel(), language_only=False
        )
        assert prompts == []
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


def test_moments_language_vocab_buckets_by_qwen_length(monkeypatch):
    _install_minimal_shims()

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.append(str(repo_root / "thesis_project" / "data_generation"))

    from create_moments_dataset import build_language_replacement_vocab

    class FakeTokenizer:
        def __init__(self, length_map):
            self.length_map = length_map

        def __call__(self, text, add_special_tokens=False):
            tokens = []
            for raw in text.split():
                word = raw.strip(".,!?;:\"'")
                if not word:
                    continue
                tokens.extend([word] * self.length_map.get(word.lower(), 1))
            return {"input_ids": tokens}

    class FakeSent:
        def __init__(self, text):
            self.text = text
            self.start_char = 0
            self.end_char = len(text)

    class FakeToken:
        def __init__(self, text, idx, sent):
            self.text = text
            self.idx = idx
            self.sent = sent
            self.is_space = False
            self.is_punct = False

    class FakeDoc:
        def __init__(self, text):
            self.text = text
            self._sent = FakeSent(text)
            self._tokens = []
            offset = 0
            for part in text.split():
                idx = text.index(part, offset)
                self._tokens.append(FakeToken(part, idx, self._sent))
                offset = idx + len(part)
            self.sents = [self._sent]

        def __iter__(self):
            return iter(self._tokens)

    monkeypatch.setattr(
        "create_moments_dataset.load_spacy_nlp",
        lambda: (lambda text: FakeDoc(text)),
    )

    tokenizer = FakeTokenizer(
        {
            "alpha": 1,
            "beta": 1,
            "gamma": 2,
            "delta": 2,
        }
    )
    vocab = build_language_replacement_vocab(
        [
            {
                "local_text": "alpha gamma",
                "global_text": "beta delta",
            }
        ],
        tokenizer,
    )

    assert vocab == {
        "1": ["alpha", "beta"],
        "2": ["delta", "gamma"],
    }


def test_moments_dataset_sample_replacement_uses_same_length_words(monkeypatch):
    _install_minimal_shims()

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.append(str(repo_root / "thesis_project" / "data_generation"))

    import create_moments_dataset as cmd

    class FakeTokenizer:
        def __init__(self, length_map):
            self.length_map = length_map

        def __call__(self, text, add_special_tokens=False):
            tokens = []
            for raw in text.split():
                word = raw.strip(".,!?;:\"'")
                if not word:
                    continue
                tokens.extend([word] * self.length_map.get(word.lower(), 1))
            return {"input_ids": tokens}

    class FakeSent:
        def __init__(self, text):
            self.text = text
            self.start_char = 0
            self.end_char = len(text)

    class FakeToken:
        def __init__(self, text, idx, sent):
            self.text = text
            self.idx = idx
            self.sent = sent
            self.is_space = False
            self.is_punct = False

    class FakeDoc:
        def __init__(self, text):
            self.text = text
            self._sent = FakeSent(text)
            self._tokens = []
            offset = 0
            for part in text.split():
                idx = text.index(part, offset)
                self._tokens.append(FakeToken(part, idx, self._sent))
                offset = idx + len(part)
            self.sents = [self._sent]

        def __iter__(self):
            return iter(self._tokens)

    monkeypatch.setattr(cmd, "load_spacy_nlp", lambda: (lambda text: FakeDoc(text)))

    tokenizer = FakeTokenizer(
        {
            "alpha": 1,
            "beta": 1,
            "gamma": 2,
            "delta": 2,
            "kappa": 3,
            "you": 1,
            "are": 1,
            "watching": 1,
            "a": 1,
            "football": 1,
            "match": 1,
            "is": 1,
            "this": 1,
            "goal": 1,
            "answer": 1,
            "yes": 1,
            "or": 1,
            "no": 1,
        }
    )

    vocab = {
        "1": ["alpha", "beta", "kappa"],
        "2": ["gamma", "delta"],
    }
    text = "alpha kappa"

    result = cmd.replace_one_content_word_with_dataset_sample(
        text,
        task="goal",
        sample_id="clip_a__1__IM_1",
        perturbation_fraction=0.5,
        vocab=vocab,
        tokenizer=tokenizer,
    )

    assert result is not None
    perturbed_text, prompt_state, change_summary = result
    assert perturbed_text != text
    assert "dataset_sample" in prompt_state
    assert "->" in change_summary
    clean_prompt = cmd.build_prompt(text, "goal")
    perturbed_prompt = cmd.build_prompt(perturbed_text, "goal")
    assert cmd._token_length(tokenizer, clean_prompt) == cmd._token_length(
        tokenizer, perturbed_prompt
    )


def test_moments_runtime_alignment_helper():
    _install_minimal_shims()

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.append(str(repo_root / "reproducing_code" / "vlm-circuits-analysis"))

    from modality_alignment_utils import get_moments_alignment_info

    class FakeTokenId:
        def __init__(self, value):
            self._value = value

        def item(self):
            return self._value

    class FakeTokenTensor:
        def __init__(self, values):
            self._values = values

        def view(self, *_args, **_kwargs):
            return [FakeTokenId(v) for v in self._values]

    class DummyPrompt:
        def __init__(self, prompt, images):
            self.prompt = prompt
            self.images = images

    class DummyModel:
        def __init__(self):
            self._token_map = {
                "L_PROMPT": [100, 101, 200, 201, 300, 301, 302, 303],
                "VL_PROMPT": [100, 101, 500, 501, 200, 201, 300, 301, 302, 303],
                "Is this a goal? Answer yes or no.": [300, 301, 302, 303],
            }
            self._string_map = {
                100: "<|prefix_0|>",
                101: "<|prefix_1|>",
                200: "<|desc_0|>",
                201: "<|desc_1|>",
                300: "<|question_0|>",
                301: "<|question_1|>",
                302: "<|question_2|>",
                303: "<|question_3|>",
                500: "<|vision_0|>",
                501: "<|vision_1|>",
            }

        def to_tokens(self, prompt, images=None, prepend_bos=False):
            key = prompt
            if key not in self._token_map:
                raise KeyError(key)
            return FakeTokenTensor(self._token_map[key])

        def to_single_str_token(self, token_id):
            return self._string_map[token_id]

    model = DummyModel()
    l_prompt = DummyPrompt("L_PROMPT", [])
    vl_prompt = DummyPrompt("VL_PROMPT", ["frame_1.png"])

    info = get_moments_alignment_info(model, l_prompt, vl_prompt, "moments_goal")
    assert info["l_data_limits"] == [2, 4]
    assert info["vl_data_limits"] == [2, 4]
    assert info["prefix_len"] == 2
    assert info["l_seq_len"] == 8
    assert info["vl_seq_len"] == 10
    info["pos_mapping"].assert_full_mapping(l_prompt, vl_prompt, model)


def test_moments_runtime_alignment_helper_falls_back_to_suffix_length():
    _install_minimal_shims()

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.append(str(repo_root / "reproducing_code" / "vlm-circuits-analysis"))

    from modality_alignment_utils import get_moments_alignment_info

    class FakeTokenId:
        def __init__(self, value):
            self._value = value

        def item(self):
            return self._value

    class FakeTokenTensor:
        def __init__(self, values):
            self._values = values

        def view(self, *_args, **_kwargs):
            return [FakeTokenId(v) for v in self._values]

    class DummyPrompt:
        def __init__(self, prompt, images):
            self.prompt = prompt
            self.images = images

    class DummyModel:
        def __init__(self):
            self._token_map = {
                "L_PROMPT": [100, 101, 200, 201, 300, 301, 302, 303],
                "VL_PROMPT": [100, 101, 500, 501, 200, 201, 300, 301, 302, 303],
                # Deliberately different from the tail of L_PROMPT so the
                # helper exercises the suffix-length fallback.
                "Is this a goal? Answer yes or no.": [900, 901, 902, 903],
            }
            self._string_map = {
                100: "<|prefix_0|>",
                101: "<|prefix_1|>",
                200: "<|desc_0|>",
                201: "<|desc_1|>",
                300: "<|question_0|>",
                301: "<|question_1|>",
                302: "<|question_2|>",
                303: "<|question_3|>",
                500: "<|vision_0|>",
                501: "<|vision_1|>",
                900: "<|alt_question_0|>",
                901: "<|alt_question_1|>",
                902: "<|alt_question_2|>",
                903: "<|alt_question_3|>",
            }

        def to_tokens(self, prompt, images=None, prepend_bos=False):
            key = prompt
            if key not in self._token_map:
                raise KeyError(key)
            return FakeTokenTensor(self._token_map[key])

        def to_single_str_token(self, token_id):
            return self._string_map[token_id]

    model = DummyModel()
    l_prompt = DummyPrompt("L_PROMPT", [])
    vl_prompt = DummyPrompt("VL_PROMPT", ["frame_1.png"])

    info = get_moments_alignment_info(model, l_prompt, vl_prompt, "moments_goal")
    assert info["l_data_limits"] == [2, 4]
    assert info["vl_data_limits"] == [2, 4]
    assert info["prefix_len"] == 2


def test_moments_dataset_sample_language_rewrite_is_deterministic():
    _install_minimal_shims()

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.append(str(repo_root / "thesis_project" / "data_generation"))

    from create_moments_dataset import build_language_counterfactual_text

    class FakeSentence:
        def __init__(self, text):
            self.text = text
            self.start_char = 0

    class FakeToken:
        def __init__(self, text, pos_, tag_, idx, sent):
            self.text = text
            self.pos_ = pos_
            self.tag_ = tag_
            self.idx = idx
            self.is_stop = False
            self.is_punct = False
            self.is_space = False
            self.lemma_ = text.lower()
            self.sent = sent

    class FakeDoc:
        def __init__(self, text):
            self.text = text
            self._sent = FakeSentence(text)
            self._tokens = [
                FakeToken("Brilliant", "ADJ", "JJ", 0, self._sent),
                FakeToken("pass", "NOUN", "NN", 10, self._sent),
                FakeToken("and", "CCONJ", "CC", 15, self._sent),
                FakeToken("great", "ADJ", "JJ", 19, self._sent),
                FakeToken("shot", "NOUN", "NN", 25, self._sent),
            ]

        def __iter__(self):
            return iter(self._tokens)

        @property
        def sents(self):
            return [self._sent]

    class FakeNLP:
        def __call__(self, text):
            return FakeDoc(text)

    class FakeTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": text.split()}

    with patch("create_moments_dataset.load_spacy_nlp", return_value=FakeNLP()):
        text, state, changes = build_language_counterfactual_text(
            "Brilliant pass and great shot",
            "Global fallback text",
            perturbation_fraction=0.10,
            mode="dataset_sample",
            task="goal",
            sample_id="clip__1__IM_1",
            language_vocab={
                "1": ["awful", "block"],
            },
            language_tokenizer=FakeTokenizer(),
        )

    assert text.lower().startswith(("awful", "block"))
    assert state == "coverage_10pct+dataset_sample+qwen_same_token_length"
    assert "dataset_sample:length_1" in changes


def test_moments_language_rewrite_preserves_template_and_requested_coverage(monkeypatch):
    _install_minimal_shims()

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.append(str(repo_root / "thesis_project" / "data_generation"))

    import create_moments_dataset as cmd

    class FakeToken:
        def __init__(self, text, idx, sent):
            self.text = text
            self.idx = idx
            self.sent = sent
            self.is_stop = False
            self.is_punct = False
            self.is_space = False

    class FakeSentence:
        start_char = 0

    class FakeDoc:
        def __init__(self, text):
            sent = FakeSentence()
            self.tokens = []
            offset = 0
            for word in text.split():
                idx = text.index(word, offset)
                self.tokens.append(FakeToken(word, idx, sent))
                offset = idx + len(word)

        def __iter__(self):
            return iter(self.tokens)

        @property
        def sents(self):
            return [self.tokens[0].sent]

    class FakeNLP:
        def __call__(self, text):
            return FakeDoc(text)

    class FakeTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": text.split()}

    monkeypatch.setattr(cmd, "load_spacy_nlp", lambda: FakeNLP())
    text, state, changes = cmd.build_language_counterfactual_text(
        "alpha beta gamma delta",
        "unused fallback",
        perturbation_fraction=0.5,
        mode="dataset_sample",
        task="goal",
        sample_id="clip__1__IM_1",
        language_vocab={"1": ["alpha", "beta", "gamma", "delta", "replacement"]},
        language_tokenizer=FakeTokenizer(),
    )

    clean_prompt = cmd.build_prompt("alpha beta gamma delta", "goal")
    perturbed_prompt = cmd.build_prompt(text, "goal")
    assert cmd._token_length(FakeTokenizer(), clean_prompt) == cmd._token_length(
        FakeTokenizer(), perturbed_prompt
    )
    assert perturbed_prompt.endswith("Is this a goal? Answer yes or no.")
    assert perturbed_prompt.startswith("You are watching a football match.")
    assert len(changes.split("; ")) == 2
    assert state == "coverage_50pct+dataset_sample+qwen_same_token_length"
