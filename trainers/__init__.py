from trainers.trainer import BaseTrainer, DeeponevecTrainer
from trainers.logger import BaseLogger


def get_trainer(optimizer, cfg, scheduler=None):
    trainer_type = cfg.get("trainer", "base")
    device = cfg["device"]
    if trainer_type == "base":
        trainer = BaseTrainer(optimizer, cfg["training"], device=device, scheduler=scheduler)
    elif trainer_type == "deepovec":
        trainer = DeeponevecTrainer(optimizer, cfg["training"], device=device, scheduler=scheduler)
    return trainer

def get_logger(cfg, writer):
    logger_type = cfg["logger"].get("type", "base")
    endwith = cfg["logger"].get("endwith", [])
    if logger_type in ["base"]:
        logger = BaseLogger(writer, endwith=endwith)
    return logger
