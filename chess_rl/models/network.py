import torch as T
import torch.nn as nn

from chess_rl.models.residual import ResidualBlock


class ChessAgent(nn.Module):
    def __init__(self, input_shape, channels, out):
        super().__init__()
        feature_size = (channels * 2) * 8 * 8
        self.shared = nn.Sequential(
            nn.Conv2d(input_shape, channels * 2, 3, 1, 1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(),
            nn.Conv2d(channels * 2, channels * 2, 3, 1, 1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(),
            nn.Conv2d(channels * 2, channels * 2, 3, 1, 1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(),
            *[ResidualBlock(channels * 2) for _ in range(4)],
            nn.Flatten(),
            nn.Linear(feature_size, 256),
            nn.ReLU(),
        )

        self.actor = nn.Linear(256, out)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = self.shared(x)

        policy_logits = self.actor(x)
        value = self.critic(x)

        return policy_logits, value


if __name__ == "__main__":
    ang = ChessAgent(18, 64, 64 * 64)
    test = T.randn((32, 18, 8, 8))
    shared = ang.shared(test)

    policy, value = ang(test)

    print(shared.shape)

    print(policy.shape)
    print(value.shape)
