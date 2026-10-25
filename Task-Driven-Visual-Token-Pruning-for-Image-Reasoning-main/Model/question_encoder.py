import torch

from transformers import (
    AutoTokenizer,
    AutoModel
)

class QuestionEncoder:

    def __init__(self):

        model_path = \
        "/root/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots"

        import os

        snapshot = os.listdir(
            model_path
        )[0]

        model_path = \
        f"{model_path}/{snapshot}"

        self.tokenizer = \
        AutoTokenizer.from_pretrained(
            model_path
        )

        self.model = \
        AutoModel.from_pretrained(
            model_path
        )

        self.model.eval()

    @torch.no_grad()
    def encode(self,text):

        x = self.tokenizer(
            text,
            return_tensors="pt"
        )

        y = self.model(**x)

        feat = y.last_hidden_state.mean(
            dim=1
        )

        return feat


if __name__=="__main__":

    enc = QuestionEncoder()

    feat = enc.encode(
        "What color is the cat?"
    )

    print(feat.shape)