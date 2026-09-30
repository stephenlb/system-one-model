"""Gemma 4 with a 26-logit A-Z output layer instead of the 262k-token LM head.

The decoder is unchanged. Only the final projection differs: ``lm_head`` has 26
rows (the A-Z rows of the original tied embedding), so one forward pass returns
one logit per letter and the vocabulary-sized matmul disappears. Gemma's final
logit soft-capping still runs on those 26 values, exactly as it did on the
original rows.

    model = AutoModelForMultimodalLM.from_pretrained(repo, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(repo)
    result = model.system_one(tokenizer, state="...", questions={...})
"""

from __future__ import annotations

import json

import torch
from torch import nn
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.gemma4_unified.modeling_gemma4_unified import (
    Gemma4UnifiedForConditionalGeneration,
)

from .configuration_truetype import LETTERS, TrueTypeSystemOneConfig
from .questions import build_question, score_question_tensor
from .render import render_batch

DEFAULT_TEMPERATURE = 0.7

# Gemma's checkpoints keep the vision embedder under legacy names that
# transformers renames per model_type. This model shares that layout.
register_checkpoint_conversion_mapping(
    TrueTypeSystemOneConfig.model_type, get_checkpoint_conversion_mapping("gemma4_unified"), overwrite=True
)


class TrueTypeSystemOneModel(Gemma4UnifiedForConditionalGeneration):
    config_class = TrueTypeSystemOneConfig
    _tied_weights_keys = {}

    def __init__(self, config: TrueTypeSystemOneConfig):
        super().__init__(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, len(LETTERS), bias=False)
        self.post_init()

    def letter_logits(self, input_ids, attention_mask=None, **kwargs) -> torch.Tensor:
        """``[batch, 26]`` float32 logits for A-Z at the next-token position.

        Inputs must be left-padded so the last position is the answer slot.
        """
        out = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=1,
            use_cache=False,
            **kwargs,
        )
        return out.logits[:, -1, :].float()

    @torch.inference_mode()
    def system_one(
        self,
        tokenizer,
        state,
        questions: dict[str, dict],
        temperature: float = DEFAULT_TEMPERATURE,
        batch_size: int = 16,
        include_letter_logits: bool = False,
    ) -> dict:
        """Answer typed questions (``noul``, ``choice``, ``score``) about ``state``.

        ``questions`` maps an id to ``{"type", "instructions", "criteria"}``.
        Returns ``{"answers": {id: {...probabilities and answer...}}}``.
        """
        if not questions:
            raise ValueError("at least one question is required")
        state_text = state if isinstance(state, str) else json.dumps(state, indent=2)
        parsed = [build_question(qid, spec) for qid, spec in questions.items()]
        prompts = [item.text for item in render_batch(parsed, state_text)]

        rows = []
        padding_side, tokenizer.padding_side = tokenizer.padding_side, "left"
        try:
            for start in range(0, len(prompts), max(1, batch_size)):
                encoded = tokenizer(
                    prompts[start : start + batch_size],
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=True,
                ).to(self.device)
                rows.extend(self.letter_logits(encoded["input_ids"], encoded["attention_mask"]))
        finally:
            tokenizer.padding_side = padding_side

        result = {"answers": {}}
        for question, row in zip(parsed, rows):
            result["answers"][question.id] = score_question_tensor(
                question, row, temperature=temperature
            ).to_dict()
        if include_letter_logits:
            result["letter_logits"] = {
                question.id: dict(zip(LETTERS, row.tolist())) for question, row in zip(parsed, rows)
            }
        return result
