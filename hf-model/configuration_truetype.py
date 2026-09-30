"""Config for the Gemma 4 System One model with a 26-logit A-Z output layer."""

from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedConfig

LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class TrueTypeSystemOneConfig(Gemma4UnifiedConfig):
    """Gemma 4 config plus the tokenizer ids the 26 output rows were copied from.

    ``letter_token_ids[i]`` is the vocabulary id whose embedding row became output
    row ``i`` (``A`` to ``Z``). It is kept for provenance and for verification
    against the original vocabulary head.
    """

    model_type = "truetype_system_one"

    def __init__(self, letter_token_ids=None, num_letters=len(LETTERS), **kwargs):
        # The 26-row head is its own parameter, not the input embedding.
        kwargs["tie_word_embeddings"] = False
        super().__init__(**kwargs)
        self.letter_token_ids = list(letter_token_ids or [])
        self.num_letters = num_letters
