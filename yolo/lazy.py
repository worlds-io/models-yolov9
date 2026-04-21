# disable OpenCV multithreading, as it can misbehave when used with PyTorch worker forking
import cv2
cv2.setNumThreads(0)

import glob
import os
import sys
from pathlib import Path

import hydra
import torch
from lightning import Trainer
from omegaconf import OmegaConf

# run a quick autotuning pass at the start of training so that we use the optimal GPU kernels for the model
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.model.yolo import create_model
from yolo.tools.solver import BaseModel, InferenceModel, TrainModel, ValidateModel
from yolo.utils.logging_utils import setup

@hydra.main(config_path='config', config_name='config', version_base=None)
def main(cfg: Config):
    if 'DATASET_OVERRIDE_CONFIG' in os.environ:
        cfg.dataset = OmegaConf.load(os.environ['DATASET_OVERRIDE_CONFIG'])

    early_stopping_patience = getattr(cfg.task, 'early_stopping_patience', None)
    epochs = getattr(cfg.task, 'epoch', None)
    val_batch_size = getattr(getattr(cfg.task, 'validation', cfg.task).data, 'batch_size', 32)

    callbacks, loggers, save_path = setup(cfg, early_stopping_patience=early_stopping_patience)

    # Gradient accumulation: keep the effective (nominal) batch size constant
    # regardless of available GPU memory. The per-step physical batch may be
    # smaller than the paper's batch size; we accumulate gradients across
    # grad_accum_steps forward/backward passes to recover the same update.
    accumulate_grad_batches = 1
    if cfg.task.task == 'train':
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        batch_size = cfg.task.data.batch_size
        nominal_batch_size = getattr(cfg.task.data, 'nominal_batch_size', None) or batch_size
        total_step_batch = batch_size * world_size
        if nominal_batch_size % total_step_batch != 0:
            print(
                f"Warning: nominal_batch_size ({nominal_batch_size}) is not divisible by "
                f"batch_size * world_size ({total_step_batch}); effective batch size will be "
                f"{(nominal_batch_size // total_step_batch) * total_step_batch}"
            )
        accumulate_grad_batches = max(1, nominal_batch_size // total_step_batch)
        print(
            f"Gradient accumulation: physical batch={batch_size}, world_size={world_size}, "
            f"nominal batch={nominal_batch_size}, accumulate_grad_batches={accumulate_grad_batches}"
        )

    trainer = Trainer(
        accelerator='auto',
        max_epochs=epochs,
        precision='16-mixed',
        logger=loggers,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        log_every_n_steps=50,
        deterministic=False,
        enable_progress_bar=False,
        default_root_dir=save_path,
        limit_val_batches=5000 // val_batch_size,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    if cfg.task.task == 'train':
        model = TrainModel(cfg)
        trainer.fit(model)

        export_onnx(cfg)
    if cfg.task.task == 'validation':
        model = ValidateModel(cfg)
        trainer.validate(model)
    if cfg.task.task == 'inference':
        model = InferenceModel(cfg)
        trainer.predict(model)


def _get_latest_checkpoint(cfg: Config) -> str:
    checkpoints_directory = os.path.join(cfg.out_path, cfg.task.task, cfg.name, 'checkpoints')

    best_checkpoint = os.path.join(checkpoints_directory, 'best.ckpt')
    if os.path.exists(best_checkpoint):
        return best_checkpoint

    checkpoint_files = glob.glob('%s/*.ckpt' % checkpoints_directory)
    if len(checkpoint_files) == 0:
        raise RuntimeError('No checkpoint found after training completed')

    return max(checkpoint_files, key=os.path.getctime)


def export_onnx(cfg: Config):
    print('Exporting model to ONNX...')
    checkpoint_file = _get_latest_checkpoint(cfg)

    checkpoint_weights = torch.load(checkpoint_file, map_location=torch.device('cpu'), weights_only=False)
    if 'state_dict' in checkpoint_weights:
        checkpoint_weights = checkpoint_weights['state_dict']
    else:
        raise RuntimeError('Could not find state_dict in checkpoint after training completed')

    cfg.model.is_exporting    = True
    cfg.model.model.auxiliary = {}
    export_model = create_model(cfg.model, cfg=cfg, class_num=cfg.dataset.class_num, weight_path=False).to('cpu')

    export_model_state_dict = {}
    for model_key, model_weight in checkpoint_weights.items():
        model_key = model_key.replace('model.model.', 'model.')
        if model_key in export_model.state_dict():
            export_model_state_dict[model_key] = model_weight

    export_model.load_state_dict(export_model_state_dict, strict=True)
    export_model.eval()

    dummy_input = torch.zeros((1, cfg.image_size[1], cfg.image_size[0], 3))
    torch.onnx.export(
        export_model,
        (dummy_input,),
        cfg.export_path,
        export_params=True,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    )

    print('Successfully exported model to %s' % cfg.export_path)


if __name__ == '__main__':
    main()
