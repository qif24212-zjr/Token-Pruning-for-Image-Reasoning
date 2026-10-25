import torch

from transformers import (
    AutoTokenizer,
    AutoModel
)


class SemanticEncoder:

    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.model = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.model.eval()

    @torch.no_grad()
    def encode(self, text):

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        outputs = self.model(**inputs)

        # mean pooling
        embedding = outputs.last_hidden_state.mean(
            dim=1
        )

        embedding = embedding.squeeze(0)

        return embedding


if __name__ == "__main__":

    encoder = SemanticEncoder()

    question = "What color is the cat?"

    feat = encoder.encode(question)

    print("shape =", feat.shape)

    print(feat[:10])