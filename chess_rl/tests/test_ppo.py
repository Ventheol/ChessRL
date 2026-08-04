import torch as T

from chess_rl.ppo.gae import get_advantages
from chess_rl.ppo.ppo import Agent


def main():
    print("Creating agent...")
    agent = Agent()

    print("Collecting rollout...")
    agent.rollout_step()

    buffer = agent.buffer.get()

    print("\n===== BUFFER =====")
    for key, value in buffer.items():
        print(f"{key:10} : {tuple(value.shape)}")

    print("\nComputing GAE...")
    print(buffer["rewards"].shape)
    print(buffer["values"].shape)
    print(len(buffer["values"]))
    returns, advantages = get_advantages(
        values=buffer["values"],
        dones=buffer["dones"],
        rewards=buffer["rewards"],
    )

    print("returns:", returns.shape)
    print("advantages:", advantages.shape)

    print("\nTesting evaluate()...")

    log_probs, values, entropy = agent.evaluate(
        buffer["states"],
        buffer["actions"],
        buffer["masks"],
    )

    print("log_probs:", log_probs.shape)
    print("values:", values.shape)
    print("entropy:", entropy.shape)

    print("\nTesting PPO loss...")

    loss = agent.loss(
        buffer["log_probs"],
        log_probs,
        advantages,
        values,
        returns,
        entropy,
    )

    print("loss =", loss.item())

    print("\nTesting backward...")

    agent.optimizer.zero_grad()
    loss.backward()

    grad_norm = T.nn.utils.clip_grad_norm_(
        agent.network.parameters(),
        0.5,
    )

    print("Gradient norm:", grad_norm.item())

    agent.optimizer.step()

    agent.save()
    agent.load()
    print("\n✅ PPO smoke test passed!")


if __name__ == "__main__":
    main()
