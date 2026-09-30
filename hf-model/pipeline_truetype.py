"""``pipeline("system-one", ...)``: typed questions in, probabilities and answers out.

    pipe = pipeline("system-one", model=repo, trust_remote_code=True)
    pipe({"state": "I was charged twice.",
          "questions": {"refund": {"type": "noul", "instructions": "Does the text request a refund?"}}})
"""

from __future__ import annotations

import json

import torch
from transformers import Pipeline

from .configuration_truetype import LETTERS
from .questions import build_question, score_question_tensor
from .render import render_batch


class SystemOnePipeline(Pipeline):
    def _sanitize_parameters(self, temperature=None, include_letter_logits=None, **kwargs):
        postprocess = {}
        if temperature is not None:
            postprocess["temperature"] = temperature
        if include_letter_logits is not None:
            postprocess["include_letter_logits"] = include_letter_logits
        return {}, {}, postprocess

    def preprocess(self, inputs):
        state, questions = inputs["state"], inputs["questions"]
        if not questions:
            raise ValueError("at least one question is required")
        state_text = state if isinstance(state, str) else json.dumps(state, indent=2)
        parsed = [build_question(qid, spec) for qid, spec in questions.items()]
        prompts = [item.text for item in render_batch(parsed, state_text)]
        self.tokenizer.padding_side = "left"
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
        return {"questions": parsed, "input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}

    def _forward(self, model_inputs):
        with torch.inference_mode():
            logits = self.model.letter_logits(
                model_inputs["input_ids"].to(self.model.device),
                model_inputs["attention_mask"].to(self.model.device),
            )
        return {"questions": model_inputs["questions"], "logits": logits}

    def postprocess(self, model_outputs, temperature=0.7, include_letter_logits=False):
        questions, logits = model_outputs["questions"], model_outputs["logits"]
        result = {
            "answers": {
                q.id: score_question_tensor(q, row, temperature=temperature).to_dict()
                for q, row in zip(questions, logits)
            }
        }
        if include_letter_logits:
            result["letter_logits"] = {q.id: dict(zip(LETTERS, row.tolist())) for q, row in zip(questions, logits)}
        return result
