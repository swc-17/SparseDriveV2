
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)

class CheckpointCallback(Callback):
    def __init__(self):
        super().__init__()
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        
        trainer.strategy.barrier()
        epoch = trainer.current_epoch
        ckpt_dir = Path(trainer.default_root_dir) / "periodic_pdm_ckpts"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"ep{epoch+1:04d}.ckpt"

        trainer.save_checkpoint(str(ckpt_path))
        trainer.print(f"[PDM] saved ckpt: {ckpt_path}")

        trainer.strategy.barrier()