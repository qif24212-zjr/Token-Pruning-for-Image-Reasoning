from semantic.question_encoder import \
QuestionEncoder

import semantic.global_semantic \
as gs

enc = QuestionEncoder()

gs.SEMANTIC_FEATURE = \
enc.encode(
    "What color is the cat?"
)

print(
    gs.SEMANTIC_FEATURE.shape
)