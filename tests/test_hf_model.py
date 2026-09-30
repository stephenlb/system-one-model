"""Tests for the Hugging Face model in ``hf-model/`` (26-logit A-Z output layer).

Tiny tier: a small random Gemma 4 with the real tokenizer checks shapes,
save/reload, padding invariance, and the pipeline plumbing without 12B weights.
Live tier: runs the built ``build/truetype_system_one`` repo through
``transformers.pipeline`` and is skipped until ``hf-model/build.py`` has run
(override the location with ``TRUETYPE_HF_MODEL_DIR``).
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "hf-model"))
import build  # noqa: E402

BASE_ID = "google/gemma-4-12B"
BUILT_DIR = Path(os.environ.get("TRUETYPE_HF_MODEL_DIR", ROOT / "build" / "truetype_system_one"))

QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Does the text request a refund?"},
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "Charges", "returns": "Refunds", "tech": "Bugs"},
    },
    "anger": {
        "type": "score",
        "instructions": "How upset is the writer?",
        "criteria": ["calm", "annoyed", "furious"],
    },
}
STATE = "I was charged twice for order A-104 and I want my money back."


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained(BASE_ID, local_files_only=True)
    except Exception as exc:  # tokenizer not cached and no network in CI
        pytest.skip(f"{BASE_ID} tokenizer unavailable: {exc}")


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    """The staged code package, exactly as ``build.py`` writes it next to the weights."""
    out = tmp_path_factory.mktemp("stage") / "tt_stage"
    build.stage_code(out)
    sys.path.insert(0, str(out.parent))
    modules = {
        name: importlib.import_module(f"tt_stage.{name}")
        for name in ("configuration_truetype", "modeling_truetype", "pipeline_truetype")
    }
    yield out, modules
    sys.path.remove(str(out.parent))


@pytest.fixture(scope="module")
def tiny_model(staged, tokenizer):
    _, modules = staged
    base = modules["configuration_truetype"].Gemma4UnifiedConfig.from_pretrained(BASE_ID)
    config_dict = {k: v for k, v in base.to_dict().items() if k != "model_type"}
    text = config_dict["text_config"]
    text.pop("per_layer_config", None)  # derived from layer_types; keyed to the 48-layer model
    layers = 2
    text.update(
        hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
        num_attention_heads=2, num_key_value_heads=1, num_global_key_value_heads=1,
        head_dim=16, global_head_dim=32, layer_types=["sliding_attention", "full_attention"],
    )
    config = modules["configuration_truetype"].TrueTypeSystemOneConfig(
        **config_dict, letter_token_ids=build.letter_token_ids(tokenizer)
    )
    torch.manual_seed(0)
    model = modules["modeling_truetype"].TrueTypeSystemOneModel(config).eval()
    return model


# ------------------------------------------------------------------ no model


def test_letter_token_ids_are_26_distinct_single_tokens(tokenizer):
    ids = build.letter_token_ids(tokenizer)
    assert len(ids) == 26 and len(set(ids)) == 26
    # Same variant for every letter: the prompt ends "Answer:" so the next token is " Y".
    assert ids == [tokenizer.encode(f" {c}", add_special_tokens=False)[0] for c in build.LETTERS]


def test_stage_code_copies_shared_sources_from_src(staged):
    out, _ = staged
    for name in (*build.OWN, *build.SHARED, "__init__.py"):
        assert (out / name).exists(), name
    for name in build.SHARED:
        assert (out / name).read_text() == (ROOT / "src" / "truetype" / name).read_text()


def test_config_records_letters_and_unties_head(staged, tokenizer):
    config = staged[1]["configuration_truetype"].TrueTypeSystemOneConfig(
        letter_token_ids=build.letter_token_ids(tokenizer)
    )
    assert config.model_type == "truetype_system_one"
    assert config.tie_word_embeddings is False
    assert len(config.letter_token_ids) == config.num_letters == 26


# ---------------------------------------------------------------- tiny model


def test_head_has_26_rows_and_is_not_tied(tiny_model):
    assert tuple(tiny_model.lm_head.weight.shape) == (26, tiny_model.config.text_config.hidden_size)
    assert tiny_model.lm_head.weight.data_ptr() != tiny_model.get_input_embeddings().weight.data_ptr()


def test_letter_logits_shape_and_softcap(tiny_model, tokenizer):
    enc = tokenizer(["Answer:", "Question: is it? Answer:"], return_tensors="pt", padding=True)
    logits = tiny_model.letter_logits(enc["input_ids"], enc["attention_mask"])
    cap = tiny_model.config.get_text_config().final_logit_softcapping
    assert logits.shape == (2, 26) and logits.dtype == torch.float32
    assert logits.abs().max() <= cap


def test_left_padding_does_not_change_logits(tiny_model, tokenizer):
    tokenizer.padding_side = "left"
    short, long = "Answer:", "Text: a much longer prompt with many more tokens\nAnswer:"
    with torch.inference_mode():
        alone = tiny_model.letter_logits(**{k: v for k, v in tokenizer(short, return_tensors="pt").items()
                                            if k in ("input_ids", "attention_mask")})
        enc = tokenizer([short, long], return_tensors="pt", padding=True)
        batched = tiny_model.letter_logits(enc["input_ids"], enc["attention_mask"])
    assert torch.allclose(alone[0], batched[0], atol=1e-4)


def test_system_one_returns_normalized_typed_answers(tiny_model, tokenizer):
    result = tiny_model.system_one(tokenizer, STATE, QUESTIONS, include_letter_logits=True)
    assert set(result["answers"]) == set(QUESTIONS)
    assert 0.0 <= result["answers"]["refund"]["noul"] <= 1.0
    team = result["answers"]["team"]
    assert team["choice"] in QUESTIONS["team"]["criteria"]
    assert sum(team["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    anger = result["answers"]["anger"]
    assert 0.0 <= anger["score"] <= 2.0 and list(anger["legend"]) == ["0", "1", "2"]
    assert all(len(v) == 26 for v in result["letter_logits"].values())
    json.dumps(result)  # JSON-serializable


def test_system_one_batching_matches_single_batch(tiny_model, tokenizer):
    together = tiny_model.system_one(tokenizer, STATE, QUESTIONS, batch_size=16)
    chunked = tiny_model.system_one(tokenizer, STATE, QUESTIONS, batch_size=1)
    for qid in QUESTIONS:
        assert together["answers"][qid]["type"] == chunked["answers"][qid]["type"]
    assert together["answers"]["team"]["choice"] == chunked["answers"]["team"]["choice"]
    assert together["answers"]["refund"]["noul"] == pytest.approx(chunked["answers"]["refund"]["noul"], abs=1e-3)


def test_system_one_restores_tokenizer_padding_side(tiny_model, tokenizer):
    tokenizer.padding_side = "right"
    tiny_model.system_one(tokenizer, STATE, QUESTIONS)
    assert tokenizer.padding_side == "right"


def test_system_one_rejects_bad_requests(tiny_model, tokenizer):
    with pytest.raises(ValueError, match="at least one question"):
        tiny_model.system_one(tokenizer, STATE, {})
    with pytest.raises(ValueError, match="unsupported type"):
        tiny_model.system_one(tokenizer, STATE, {"q": {"type": "bogus", "instructions": "x"}})


def test_pipeline_matches_model(staged, tiny_model, tokenizer):
    pipe = staged[1]["pipeline_truetype"].SystemOnePipeline(model=tiny_model, tokenizer=tokenizer)
    piped = pipe({"state": STATE, "questions": QUESTIONS}, include_letter_logits=True)
    direct = tiny_model.system_one(tokenizer, STATE, QUESTIONS, include_letter_logits=True)
    assert piped["answers"].keys() == direct["answers"].keys()
    assert piped["answers"]["team"]["choice"] == direct["answers"]["team"]["choice"]
    for qid in QUESTIONS:
        for letter, value in direct["letter_logits"][qid].items():
            assert piped["letter_logits"][qid][letter] == pytest.approx(value, abs=1e-3)


def test_save_and_reload_round_trips_weights(staged, tiny_model, tokenizer, tmp_path):
    tiny_model.save_pretrained(tmp_path)
    reloaded = staged[1]["modeling_truetype"].TrueTypeSystemOneModel.from_pretrained(tmp_path, device_map="cpu").eval()
    assert torch.equal(reloaded.lm_head.weight.cpu(), tiny_model.lm_head.weight.cpu())
    a = tiny_model.system_one(tokenizer, STATE, QUESTIONS)
    b = reloaded.system_one(tokenizer, STATE, QUESTIONS)
    assert a["answers"]["team"]["choice"] == b["answers"]["team"]["choice"]
    assert a["answers"]["refund"]["noul"] == pytest.approx(b["answers"]["refund"]["noul"], abs=1e-4)


# ---------------------------------------------------------- built 12B repo

built = pytest.mark.skipif(not (BUILT_DIR / "model.safetensors").exists(), reason="run hf-model/build.py first")


@pytest.fixture(scope="module")
def built_pipeline():
    device = os.environ.get("TRUETYPE_TEST_DEVICE", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return transformers.pipeline(
        "system-one", model=str(BUILT_DIR), trust_remote_code=True, dtype=torch.bfloat16, device=device
    )


@built
def test_built_config_declares_auto_map_and_pipeline():
    config = json.loads((BUILT_DIR / "config.json").read_text())
    assert config["auto_map"]["AutoModelForMultimodalLM"] == "modeling_truetype.TrueTypeSystemOneModel"
    assert config["custom_pipelines"]["system-one"]["impl"] == "pipeline_truetype.SystemOnePipeline"
    assert config["tie_word_embeddings"] is False and len(config["letter_token_ids"]) == 26


@built
def test_built_pipeline_answers_are_correct(built_pipeline):
    answers = built_pipeline({"state": STATE, "questions": QUESTIONS})["answers"]
    assert answers["refund"]["noul"] > 0.9
    assert answers["team"]["choice"] == "billing"
    assert sum(answers["team"]["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    assert built_pipeline.model.lm_head.weight.shape[0] == 26
    negative = built_pipeline({"state": STATE, "questions": {"sports": {
        "type": "noul", "instructions": "Does the text talk about sports?"}}})
    assert negative["answers"]["sports"]["noul"] < 0.1


@built
def test_built_head_rows_equal_original_embedding_rows(built_pipeline):
    from safetensors import safe_open

    ids = built_pipeline.model.config.letter_token_ids
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(BASE_ID, "model.safetensors", local_files_only=True)
    with safe_open(path, "pt") as f:
        key = next(k for k in f.keys() if k.endswith("embed_tokens.weight"))
        original = f.get_slice(key)
        rows = torch.stack([original[i : i + 1][0] for i in ids])
    assert torch.equal(built_pipeline.model.lm_head.weight.cpu(), rows.to(torch.bfloat16))
