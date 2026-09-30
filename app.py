"""Managed Hugging Face Space entry point: a small UI beside the REST API."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr

# A Space runs this file from the repository root; retain the normal src layout.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from truetype.api import SystemOneRequest, app as api_app, system_one


EXAMPLE_QUESTIONS = json.dumps(
    {
        "refund": {
            "type": "noul",
            "instructions": "Does the text request a refund?",
        },
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "Charges, invoices, and payment problems",
                "returns": "Exchanges, refunds, and damaged items",
            },
        },
    },
    indent=2,
)


def evaluate(state: str, questions_json: str) -> str:
    """Evaluate a state against System One questions entered in the Space UI."""
    try:
        questions = json.loads(questions_json)
        result = system_one(SystemOneRequest(state=state, questions=questions))
        return json.dumps(result.model_dump(), indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, indent=2)


with gr.Blocks(title="Truetype Jev Replica") as demo:
    gr.Markdown("# Truetype Jev Replica\nOne-token typed decisions from Gemma 4.")
    state = gr.Textbox(label="State", lines=5, value="I was charged twice for order A-104.")
    questions = gr.Code(label="Questions JSON", language="json", value=EXAMPLE_QUESTIONS)
    submit = gr.Button("Decide", variant="primary")
    output = gr.Code(label="Response", language="json")
    submit.click(evaluate, inputs=[state, questions], outputs=output)
    gr.Markdown("REST API: `/v1/systemone` · Docs: `/docs` · Health: `/health`")


app = gr.mount_gradio_app(api_app, demo, path="/ui")


if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
