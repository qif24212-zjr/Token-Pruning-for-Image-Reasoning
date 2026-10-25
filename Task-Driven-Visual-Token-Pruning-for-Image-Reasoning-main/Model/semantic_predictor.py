import torch
import torch.nn as nn


class SemanticPredictor(nn.Module):

    def __init__(self):

        super().__init__()

        self.semantic_proj = nn.Linear(
            384,
            256
        )

        self.score_head = nn.Sequential(

            nn.Linear(
                4096 + 256 + 1,
                512
            ),

            nn.ReLU(),

            nn.Linear(
                512,
                1
            )
        )

    def forward(
        self,
        visual,
        semantic,
        mask
    ):

        semantic = self.semantic_proj(
            semantic
        )
        print(
                "SEMANTIC NORM:",
                semantic.norm(dim=-1)
        )
        semantic = semantic.unsqueeze(1)

        semantic = semantic.expand(
            visual.shape[0],
            visual.shape[1],
            256
        )

        mask = mask.float()

        mask = mask.unsqueeze(-1)

        fusion = torch.cat(
            [
                visual,
                semantic,
                mask
            ],
            dim=-1
        )

        score = self.score_head(
            fusion
        )

        return score.squeeze(-1)


if __name__ == "__main__":

    model = SemanticPredictor()

    visual = torch.randn(
        1,
        576,
        4096
    )

    semantic = torch.randn(
        1,
        384
    )

    mask = torch.randint(
        0,
        2,
        (1,576)
    ).float()

    y = model(
        visual,
        semantic,
        mask
    )

    print(y.shape)