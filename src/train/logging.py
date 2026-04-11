"""Training logger abstraction for TensorBoard and Weights & Biases."""

from __future__ import annotations

from typing import Dict, Optional


class TrainLogger:
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        raise NotImplementedError

    def log_text(self, key: str, text: str, step: int) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None


class NoOpLogger(TrainLogger):
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        del metrics, step

    def log_text(self, key: str, text: str, step: int) -> None:
        del key, text, step


class TensorBoardLogger(TrainLogger):
    def __init__(self, log_dir: str) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=log_dir)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)

    def log_text(self, key: str, text: str, step: int) -> None:
        self.writer.add_text(key, text, step)

    def close(self) -> None:
        self.writer.close()


class WandBLogger(TrainLogger):
    def __init__(self, project: str, run_name: Optional[str] = None, config: Optional[dict] = None) -> None:
        import wandb

        self.wandb = wandb
        self.wandb.init(project=project, name=run_name, config=config or {})

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        self.wandb.log(metrics, step=step)

    def log_text(self, key: str, text: str, step: int) -> None:
        self.wandb.log({key: text}, step=step)

    def close(self) -> None:
        self.wandb.finish()


def build_logger(kind: str, **kwargs) -> TrainLogger:
    kind = kind.lower()
    if kind == "none":
        return NoOpLogger()
    if kind == "tensorboard":
        return TensorBoardLogger(log_dir=kwargs.get("log_dir", "runs/default"))
    if kind == "wandb":
        return WandBLogger(
            project=kwargs.get("project", "latent-diffusion"),
            run_name=kwargs.get("run_name"),
            config=kwargs.get("config"),
        )
    raise ValueError(f"Unknown logger kind: {kind}")
