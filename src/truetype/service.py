"""Service layer: TypeSafe-shaped requests in, TypeSafe-shaped answers out."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from .engine import DEFAULT_TEMPERATURE, EngineConfig, GemmaLetterEngine
from .letters import batch_letter_logits, distribution_from_letter_logits
from .questions import Question, build_question, score_question_tensor
from .render import render_batch, render_example_prefix, render_target_block


@dataclass
class BatchResult:
    model: str
    answers: dict[str, dict] = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    debug: dict = field(default_factory=dict)


class TypeSafeReplica:
    """Evaluates a state against typed questions using a single Gemma forward pass."""

    def __init__(self, engine: GemmaLetterEngine | None = None, model_name: str = "gemma-4-12b") -> None:
        self.engine = engine or GemmaLetterEngine(EngineConfig())
        self.model_name = model_name

    def warm(self, questions: dict[str, dict]) -> float:
        """Pre-fill the prefix KV cache for questions we know are coming.

        Warming does not make a first-seen question faster than one forward pass;
        the miss path already costs that. It moves the prefill before the caller's
        timed loop, which keeps per-decision latency uniform. Reused prefixes
        benefit: a Doom loop uses one prefix about 40 times. One-shot prefixes do
        not: warming four rooms costs about 4.8 seconds to save 1.6 seconds.
        Returns the elapsed milliseconds.
        """
        started = time.perf_counter()
        parsed = [build_question(qid, spec) for qid, spec in questions.items()]
        if not parsed:
            return 0.0
        if self.engine.config.prefix_cache:
            self.engine.ensure_prefix_capacity(len(parsed))
        for question in parsed:
            self.engine.score_with_prefix_device(
                render_example_prefix(question),
                [render_target_block(question, "warmup")],
            )
        return (time.perf_counter() - started) * 1000

    def _score_question_values(
        self,
        questions: list[Question],
        state: str,
    ) -> tuple[list[object], dict[str, str]]:
        """Return device A-Z rows for the optimized engine path."""
        rendered = render_batch(questions, state)
        prompts = {item.question_id: item.text for item in rendered}
        if not self.engine.config.prefix_cache:
            values = self.engine.score_batch_device(list(prompts.values()))
            return list(values), prompts

        self.engine.ensure_prefix_capacity(len(questions))
        values = []
        for question in questions:
            values.append(
                self.engine.score_with_prefix_device(
                    render_example_prefix(question), [render_target_block(question, state)]
                )[0]
            )
        return values, prompts

    def system_one(
        self,
        *,
        state: str | dict | list,
        questions: dict[str, dict],
        model: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        include_debug: bool = False,
    ) -> BatchResult:
        if not questions:
            raise ValueError("at least one question is required")

        state_text = state if isinstance(state, str) else json.dumps(state, indent=2)
        parsed: list[Question] = [
            build_question(qid, spec) for qid, spec in questions.items()
        ]

        rows, prompts = self._score_question_values(parsed, state_text)

        answers: dict[str, dict] = {}
        debug: dict[str, dict] = {}
        for question, row in zip(parsed, rows):
            answers[question.id] = score_question_tensor(
                question, row, temperature=temperature
            ).to_dict()
            if include_debug:
                logits = batch_letter_logits(row.unsqueeze(0))[0]
                top = distribution_from_letter_logits(
                    logits, top_k=self.engine.config.top_k, temperature=temperature
                )
                debug[question.id] = {
                    "prompt": prompts[question.id],
                    "letter_logits": logits,
                    "top_k": top.top_k,
                    "top_k_letters": [letter for letter, _ in top.ranked],
                }

        input_tokens = sum(map(len, prompts.values())) // 4
        return BatchResult(
            model=model or self.model_name,
            answers=answers,
            usage={"input_tokens": input_tokens, "output_tokens": len(parsed)},
            debug=debug,
        )
