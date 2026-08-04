from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir="logs"):

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(self.log_dir)

    def log_training(
        self,
        update,
        loss,
        actor_loss,
        critic_loss,
        entropy,
        reward,
    ):

        self.writer.add_scalar("Loss/Total", loss, update)
        self.writer.add_scalar("Loss/Actor", actor_loss, update)
        self.writer.add_scalar("Loss/Critic", critic_loss, update)

        self.writer.add_scalar("Policy/Entropy", entropy, update)

        self.writer.add_scalar("Reward/Average", reward, update)

    def log_evaluation(
        self,
        update,
        wins,
        losses,
        draws,
    ):

        self.writer.add_scalar("Evaluation/Wins", wins, update)
        self.writer.add_scalar("Evaluation/Losses", losses, update)
        self.writer.add_scalar("Evaluation/Draws", draws, update)

    def close(self):
        self.writer.close()
