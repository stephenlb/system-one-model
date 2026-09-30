---
library_name: transformers
license: gemma
base_model: google/gemma-4-12B
pipeline_tag: text-classification
tags:
  - custom_code
  - gemma4
  - typed-decisions
---

# System One Model (using Gemma 4 12B)

`google/gemma-4-12B` with its 262k-token language-model head replaced by a
low dimentional vector that outputs one logit per answer. The decoder is
unchanged. One forward pass gives the 26 logits; questions (`noul`, `choice`,
`score`) are answered by softmax over the logits each question.

Requires `transformers>=5.17`. The repo ships custom code, so pass
`trust_remote_code=True`.

## Model API

```python
from transformers import AutoModelForMultimodalLM, AutoTokenizer

repo = "stephenlb/system-one-model"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForMultimodalLM.from_pretrained(
    repo, trust_remote_code=True, dtype="bfloat16", device_map="auto"
)

result = model.system_one(
    tokenizer,
    state="I was charged twice for order A-104.",
    questions={
        "refund": {"type": "noul", "instructions": "Does the text request a refund?"},
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "Charges", "returns": "Refunds"},
        },
    },
)
print(result["answers"])
```

## Pipeline API

```python
from transformers import pipeline

pipe = pipeline("system-one", model=repo, trust_remote_code=True, dtype="bfloat16", device_map="auto")
print(pipe({"state": "I was charged twice.", "questions": {...}}))
```

## Output

```json
{"answers": {
  "refund": {"type": "noul", "noul": 0.93},
  "team": {"type": "choice", "choice": "billing",
           "probabilities": {"billing": 0.97, "returns": 0.03}, "confidence": 0.8},
  "urgency": {"type": "score", "score": 3.1, "legend": {"0": "..."},
              "probabilities": {"0": 0.0, "1": 0.1}, "confidence": 0.5}}}
```

`noul` is the probability of yes. `choice` returns the top label with
probabilities per option. `score` returns the expected level over an ordered
`criteria` list. Pass `temperature=` (default 0.7) to sharpen or soften.

## Raw logits

```python
enc = tokenizer(prompts, return_tensors="pt", padding=True)  # padding_side="left"
logits = model.letter_logits(enc.input_ids, enc.attention_mask)  # [batch, 26]
```

Source and prompt format: https://github.com/stephenlb/truetype.ai-open
