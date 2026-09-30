"""Build the Hugging Face model repo: Gemma 4 with a 26-logit A-Z output layer.

    python hf-model/build.py --out build/truetype_system_one

Downloads ``google/gemma-4-12B`` (or uses the local cache), swaps the tied
262k-row LM head for a 26-row head copied from the A-Z rows, and writes a
directory that loads with plain ``transformers`` + ``trust_remote_code=True``.
The directory name must be a valid Python identifier so it can be imported.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
SHARED = ("letters.py", "questions.py", "render.py")  # single source of truth: src/truetype
OWN = ("configuration_truetype.py", "modeling_truetype.py", "pipeline_truetype.py")
LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def letter_token_ids(tokenizer) -> list[int]:
    """One id per letter, all from the same variant (prompt ends ``Answer:``, so ``" Y"``)."""
    for template in (" {}", "{}"):
        ids = [tokenizer.encode(template.format(c), add_special_tokens=False) for c in LETTERS]
        if all(len(i) == 1 for i in ids):
            flat = [i[0] for i in ids]
            if len(set(flat)) == len(LETTERS):
                return flat
    raise RuntimeError("no single-token A-Z variant found")


def stage_code(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "__init__.py").write_text("")
    for name in OWN:
        shutil.copy(ROOT / name, out / name)
    for name in SHARED:
        shutil.copy(ROOT.parent / "src" / "truetype" / name, out / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="google/gemma-4-12B")
    parser.add_argument("--out", default=str(ROOT.parent / "build" / "truetype_system_one"))
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if not out.name.isidentifier():
        sys.exit(f"--out directory name {out.name!r} must be a valid Python identifier")

    stage_code(out)
    sys.path.insert(0, str(out.parent))
    config_mod = importlib.import_module(f"{out.name}.configuration_truetype")
    model_mod = importlib.import_module(f"{out.name}.modeling_truetype")

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    ids = letter_token_ids(tokenizer)

    base_config = config_mod.Gemma4UnifiedConfig.from_pretrained(args.base)
    config = config_mod.TrueTypeSystemOneConfig(
        **{k: v for k, v in base_config.to_dict().items() if k != "model_type"},
        letter_token_ids=ids,
    )
    # The 12B checkpoint has no lm_head tensor (it is tied to the embeddings), so
    # the new head loads as "missing" and is filled from the embedding rows below.
    model = model_mod.TrueTypeSystemOneModel.from_pretrained(
        args.base, config=config, dtype=getattr(torch, args.dtype), low_cpu_mem_usage=True
    )
    with torch.no_grad():
        embeddings = model.get_input_embeddings().weight
        model.lm_head.weight.copy_(embeddings.index_select(0, torch.tensor(ids, device=embeddings.device)))
    model.lm_head.requires_grad_(False)

    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)

    config_path = out / "config.json"
    saved = json.loads(config_path.read_text())
    saved["architectures"] = ["TrueTypeSystemOneModel"]
    saved["auto_map"] = {
        "AutoConfig": "configuration_truetype.TrueTypeSystemOneConfig",
        "AutoModel": "modeling_truetype.TrueTypeSystemOneModel",
        "AutoModelForMultimodalLM": "modeling_truetype.TrueTypeSystemOneModel",
        "AutoModelForImageTextToText": "modeling_truetype.TrueTypeSystemOneModel",
    }
    saved["custom_pipelines"] = {
        "system-one": {
            "impl": "pipeline_truetype.SystemOnePipeline",
            "pt": ["AutoModelForMultimodalLM"],
            "tf": [],
        }
    }
    config_path.write_text(json.dumps(saved, indent=2) + "\n")
    shutil.copy(ROOT / "README.md", out / "README.md")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
