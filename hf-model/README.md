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

Using encoder and decoder from `google/gemma-4-12B` with its 262k-token
language-model head replaced by a low dimentional vector
that outputs one logit per answer. The forward pass gives the 26 logits.
Questions (`noul`, `choice`, `score`) are answered by softmax
over the logits for each question.

## Doom played by the model.

<video controls width="640" src="https://huggingface.co/stephenlb/system-one-model/resolve/main/typesafe-replica-doom-game-only.mp4"></video>

## Super Mario Bros. World 1-1 played by the model.

<video controls width="640" src="https://huggingface.co/stephenlb/system-one-model/resolve/main/mario-world-1.mp4"></video>

## Flappy Bird played by the model.

<video controls width="640" src="https://huggingface.co/stephenlb/system-one-model/resolve/main/jev-system-one-replica-flappy-bird.mp4"></video>

## Model API

Requires `transformers>=5.17`. The repo ships custom code, so pass
`trust_remote_code=True`.

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

## Examples

Outputs below are real results from this model (pipeline defaults, temperature 0.7).

### `noul`: yes/no probability

`noul` is the probability the answer is yes. Ask several yes/no questions about
one text in a single call.

```python
pipe({
    "state": "Please cancel my subscription and refund this month's charge.",
    "questions": {
        "refund": {"type": "noul", "instructions": "Does the text request a refund?"},
        "cancel": {"type": "noul", "instructions": "Does the text ask to cancel a subscription?"},
        "sports": {"type": "noul", "instructions": "Does the text talk about sports?"},
    },
})["answers"]
# {"refund": {"type": "noul", "noul": 0.9996},
#  "cancel": {"type": "noul", "noul": 0.9998},
#  "sports": {"type": "noul", "noul": 0.0023}}
```

Optional `criteria` overrides the yes/no wording: `{"true": "...", "false": "..."}`.

Another text, different questions: tone and content checks.

```python
pipe({
    "state": "Thanks so much for the quick help, the new dashboard looks great!",
    "questions": {
        "polite": {"type": "noul", "instructions": "Is the text polite?"},
        "complaint": {"type": "noul", "instructions": "Does the text contain a complaint?"},
        "question": {"type": "noul", "instructions": "Does the text ask a question?"},
    },
})["answers"]
# {"polite":    {"type": "noul", "noul": 0.9991},
#  "complaint": {"type": "noul", "noul": 0.0009},
#  "question":  {"type": "noul", "noul": 0.0080}}
```

### `choice`: pick one option

`criteria` maps each option label to a description. You get the top label,
a probability for every option, and a `confidence` (1.0 means all mass on one
option, 0.0 means uniform). Up to 26 options.

```python
pipe({
    "state": "My package says delivered but nothing arrived at my door.",
    "questions": {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "Charges, invoices, payment problems",
                "shipping": "Delivery status, delays, lost packages",
                "returns": "Exchanges, refunds, wrong or damaged items",
            },
        },
    },
})["answers"]
# {"team": {"type": "choice", "choice": "shipping",
#           "probabilities": {"shipping": 0.9977, "returns": 0.0016, "billing": 0.0007},
#           "confidence": 0.984}}
```

Intent classification with four options:

```python
pipe({
    "state": "Can you add a dark mode to the mobile app? My eyes hurt at night.",
    "questions": {
        "intent": {
            "type": "choice",
            "instructions": "What is the intent of the message?",
            "criteria": {
                "bug_report": "Something is broken or not working as expected",
                "feature_request": "Asks for new functionality or an improvement",
                "praise": "Compliments or thanks",
                "question": "Asks how to do something",
            },
        },
    },
})["answers"]
# {"intent": {"type": "choice", "choice": "feature_request",
#             "probabilities": {"feature_request": 0.9781, "bug_report": 0.0094,
#                               "question": 0.0079, "praise": 0.0046},
#             "confidence": 0.907}}
```

### `score`: ordered scale

`criteria` is an ordered list of levels (lowest first, 2 to 26 levels). `score`
is the expected level `sum(i * p_i)`, so 1.94 below means mostly level 2 with a
little level 1. `legend` maps each level to its description.

```python
pipe({
    "state": "The checkout page crashes every time I press Pay and I can't buy anything.",
    "questions": {
        "severity": {
            "type": "score",
            "instructions": "How severe is the reported issue?",
            "criteria": [
                "Cosmetic; no impact to functionality",
                "Broken or degraded feature, but workaround exists",
                "Blocking issue; no workaround exists",
            ],
        },
    },
})["answers"]["severity"]
# {"type": "score", "score": 1.942,
#  "legend": {"0": "Cosmetic; ...", "1": "Broken or degraded ...", "2": "Blocking issue; ..."},
#  "probabilities": {"0": 0.0018, "1": 0.0542, "2": 0.9440}, "confidence": 0.796}
```

A five-point scale on mixed feedback. The distribution is spread out, so
`score` lands between levels and `confidence` is low:

```python
pipe({
    "state": "The delivery was two days late, the box was dented, but the product itself works fine.",
    "questions": {
        "satisfaction": {
            "type": "score",
            "instructions": "How satisfied is the customer overall?",
            "criteria": ["Very dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very satisfied"],
        },
    },
})["answers"]["satisfaction"]
# {"type": "score", "score": 1.408,
#  "legend": {"0": "Very dissatisfied", "1": "Dissatisfied", "2": "Neutral",
#             "3": "Satisfied", "4": "Very satisfied"},
#  "probabilities": {"0": 0.1144, "1": 0.5705, "2": 0.1635, "3": 0.0957, "4": 0.0560},
#  "confidence": 0.223}
```

### Mixing types in one call

All questions in a call see the same `state` and run in one batched forward
pass. `state` may also be a dict or list; it is serialized to JSON.

```python
pipe({
    "state": {"customer": "A-104", "message": "The checkout page crashes when I press Pay."},
    "questions": {
        "urgent": {"type": "noul", "instructions": "Does the text express urgency?"},
        "sentiment": {
            "type": "choice",
            "instructions": "What is the sentiment of the text?",
            "criteria": {
                "positive": "Happy or satisfied",
                "neutral": "Factual, no strong emotion",
                "negative": "Unhappy or frustrated",
            },
        },
        "severity": {
            "type": "score",
            "instructions": "How severe is the reported issue?",
            "criteria": ["Cosmetic", "Degraded, workaround exists", "Blocking"],
        },
    },
}, temperature=0.5, include_letter_logits=True)
```

`temperature` below 0.7 sharpens the probabilities; above it, softens them. It
never changes which option wins. `include_letter_logits=True` adds the raw A-Z
logits per question under `letter_logits`.

## Raw logits

```python
enc = tokenizer(prompts, return_tensors="pt", padding=True)  # padding_side="left"
logits = model.letter_logits(enc.input_ids, enc.attention_mask)  # [batch, 26]
```

# Blocks.ai

We needed an open weight model that offered the capabilities of Jev System One Model.
Most common AI Agents require decisions making.
The System One model approach is a great new way to do this.
[Blocks.ai](https://blocks.ai) is the secure network for the Internet of Agents (IoA).
Whether connecting agents to users in your organization or making
your agents available for public use,
Blocks.ai makes your agents securely discoverable and callable
by everyone who needs them most.
[We Rebuilt Jev's API on an Open Model and Used It to Play Doom](https://blocks.ai/blog/jev-open-model-doom).
TypeSafe AI's Jev turns unstructured input into typed decisions and probabilities.
We rebuilt the API with an open model, reached 113ms median latency on an M4 Mac, and used it to play Doom.

# Github Source System One Model

https://github.com/stephenlb/system-one-model
