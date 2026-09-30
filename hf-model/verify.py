"""Check the built model against the original full-vocabulary Gemma readout.

    python hf-model/verify.py build/truetype_system_one
"""

from __future__ import annotations

import gc
import sys

import torch
from transformers import AutoModelForMultimodalLM, AutoTokenizer, pipeline

sys.path.insert(0, "src")
from truetype.engine import EngineConfig, GemmaLetterEngine  # noqa: E402
from truetype.questions import build_question  # noqa: E402
from truetype.render import render_batch  # noqa: E402

STATE = "I was charged twice for order A-104 and I want my money back."
QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Does the text request a refund?"},
    "sports": {"type": "noul", "instructions": "Does the text talk about sports?"},
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "Charges and invoices", "returns": "Exchanges and damaged items", "tech": "Bugs"},
    },
    "anger": {
        "type": "score",
        "instructions": "How upset is the writer?",
        "criteria": ["calm", "mildly annoyed", "frustrated", "furious"],
    },
}


def main(path: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        path, trust_remote_code=True, dtype=torch.bfloat16, device_map=device if device != "mps" else None
    )
    if device == "mps":
        model = model.to("mps")
    model.eval()
    print("head:", tuple(model.lm_head.weight.shape))

    result = model.system_one(tokenizer, STATE, QUESTIONS, include_letter_logits=True)
    for qid, answer in result["answers"].items():
        print(qid, answer)

    del model
    gc.collect()

    pipe = pipeline("system-one", model=path, trust_remote_code=True, dtype=torch.bfloat16, device=device)
    piped = pipe({"state": STATE, "questions": QUESTIONS})
    print("pipeline matches model.system_one:", piped["answers"] == result["answers"])
    del pipe
    gc.collect()

    # Reference: original model, full vocabulary head, then gather the same 26 ids.
    engine = GemmaLetterEngine(EngineConfig(restrict_output_to_letters=False, device=device))
    prompts = [p.text for p in render_batch([build_question(k, v) for k, v in QUESTIONS.items()], STATE)]
    reference = engine.score_batch_device(prompts).cpu()
    worst = 0.0
    for row, qid in zip(reference, QUESTIONS):
        new = torch.tensor(list(result["letter_logits"][qid].values()))
        worst = max(worst, (row - new).abs().max().item())
        print(qid, "argmax new/ref:", int(new.argmax()), int(row.argmax()))
    print("max abs logit diff vs full-vocab head:", worst)


if __name__ == "__main__":
    main(sys.argv[1])
