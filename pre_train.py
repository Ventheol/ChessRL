import torch as T
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from chess_rl.config import DEVICE
from chess_rl.ppo.ppo import Agent
from chess_rl.utility import apply_mask, seed


def pretrain_policy(epochs=20, batch_size=128, lr=1e-3):
    print("Initializing Seed...")
    seed()
    data = T.load("data/pgn_dataset.pt")
    dataset = TensorDataset(data["states"], data["actions"], data["values"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("Creating Agent...")
    agent = Agent()
    optimizer = T.optim.Adam(agent.network.parameters(), lr=lr)
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    print("Behaviour Cloning Pre-Training...")
    agent.network.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for states, target_actions, target_values in loader:
            states = states.to(DEVICE)
            target_actions = target_actions.to(DEVICE)
            target_values = target_values.to(DEVICE)
            masks = agent.env.legal_move(states)
            policy_logits, value_preds = agent.network(states)
            masked_logits = apply_mask(policy_logits, masks)
            p_loss = policy_criterion(masked_logits, target_actions)
            v_loss = value_criterion(value_preds, target_values)
            loss = p_loss + 0.5 * v_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs} -> Loss: {total_loss / len(loader):.3f}")
    agent.save(update=0)
    print("Pretraining complete!")


if __name__ == "__main__":
    pretrain_policy()
