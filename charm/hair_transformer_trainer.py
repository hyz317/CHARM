from __future__ import annotations

from contextlib import nullcontext
from functools import partial
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from beartype.typing import Tuple, List, Type

from sklearn.model_selection import train_test_split
import torch
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from pytorch_custom_utils import (
    get_adam_optimizer,
    OptimizerWithWarmupSchedule,
    add_wandb_tracker_contextmanager
)

from .hair_transformer import HairTransformerDiscrete
from .utils import (
    count_parameters,
    exists,
    divisible_by,
    cycle,
    torch_to
)
from .utils.collate import custom_collate2
from .utils.logger import print_log
from .utils.typing import typecheck, beartype_isinstance

DEFAULT_DDP_KWARGS = DistributedDataParallelKwargs(
    find_unused_parameters = True
)

def create_model(cfg_model):
    kwargs = cfg_model
    name = kwargs.pop('name')
    model = get_model(name)(**kwargs)
    print_log("Model '{}' init: nb_params={:,}, kwargs={}".format(name, count_parameters(model), kwargs))
    return model


def get_model(name):
    return {
        'charm': HairTransformerDiscrete,
    }[name]

def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")


@add_wandb_tracker_contextmanager()
class HairTransformerTrainer(Module):
    @typecheck
    def __init__(
        self,
        model,
        dataset: Dataset,
        batch_size: int,
        grad_accum_every: int,
        num_train_steps: int = 100,
        num_train_epochs: int | None = None,
        val_dataset: Dataset | None = None,
        val_ratio: float = 0.,
        val_every: int = 100,
        val_num_batches: int = 5,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.,
        max_grad_norm: float | None = 0.5,
        scheduler: Type[_LRScheduler] | None = None,
        scheduler_kwargs: dict = dict(),
        accelerator_kwargs: dict = dict(),
        optimizer_kwargs: dict = dict(),
        checkpoint_every = 1000,
        checkpoint_every_epoch: int | None = None,
        checkpoint_folder = './checkpoints',
        data_kwargs: Tuple[str, ...] = ['translation', 'width', 'thickness', 'pc'],
        warmup_steps = 1000,
        use_wandb_tracking = False
    ):
        super().__init__()

        self.use_wandb_tracking = use_wandb_tracking

        if use_wandb_tracking:
            accelerator_kwargs['log_with'] = 'wandb'

        if 'kwargs_handlers' not in accelerator_kwargs:
            accelerator_kwargs['kwargs_handlers'] = [DEFAULT_DDP_KWARGS]

        self.accelerator = Accelerator(**accelerator_kwargs)
        self.accelerator.init_trackers('Hair_transformer')

        self.model = model

        optimizer = get_adam_optimizer(
            model.parameters(),
            lr=learning_rate,
            wd=weight_decay,
            filter_by_requires_grad=True,
            **optimizer_kwargs
        )

        self.optimizer = OptimizerWithWarmupSchedule(
            accelerator=self.accelerator,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            warmup_steps=warmup_steps,
            max_grad_norm=max_grad_norm
        )

        self.should_validate = exists(val_dataset) or val_ratio > 0
        if self.should_validate:
            self.val_every = val_every
            self.val_num_batches = val_num_batches

        if exists(val_dataset):
            self.dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=partial(custom_collate2, pad_id=model.pad_id),
                num_workers=8,
                pin_memory=True
            )
            self.train_len = len(dataset)

            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=partial(custom_collate2, pad_id=model.pad_id),
                num_workers=8,
                pin_memory=True
            )

            print(f'train_len: {self.train_len}, val_len: {len(val_dataset)}')

        elif val_ratio > 0:
            val_len = int(len(dataset) * val_ratio)
            self.train_len = len(dataset) - val_len
            
            train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=val_len, shuffle=False)
            train_sampler = SubsetRandomSampler(train_indices)
            val_sampler = SubsetRandomSampler(val_indices)

            self.dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                collate_fn=partial(custom_collate2, pad_id=model.pad_id),
                num_workers=8,
                pin_memory=True
            )

            self.val_dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                collate_fn=partial(custom_collate2, pad_id=model.pad_id),
                num_workers=8,
                pin_memory=True
            )

            print(f'train_len: {self.train_len}, val_len: {int(len(dataset) * val_ratio)}')

        else:
            self.dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=partial(custom_collate2, pad_id=model.pad_id),
                num_workers=8,
                pin_memory=True
            )
            self.train_len = len(dataset)

            print(f'train_len: {self.train_len}')

        if hasattr(dataset, 'data_kwargs') and exists(dataset.data_kwargs):
            assert beartype_isinstance(dataset.data_kwargs, List[str])
            self.data_kwargs = dataset.data_kwargs
        else:
            self.data_kwargs = data_kwargs

        self.register_buffer('step', torch.tensor(0))

        if exists(self.model.conditioner):
            self.model.conditioner.model.to(self.accelerator.device)
        (
            self.model,
            self.dataloader
        ) = self.accelerator.prepare(
            self.model,
            self.dataloader
        )

        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every
        if exists(num_train_epochs):
            self.num_train_steps = num_train_epochs * self.train_len // (batch_size * grad_accum_every)
        else:
            self.num_train_steps = num_train_steps

        if exists(checkpoint_every_epoch):
            self.checkpoint_every = checkpoint_every_epoch * self.train_len // (batch_size * grad_accum_every)
            self.checkpoint_every_epoch = checkpoint_every_epoch
        else:
            self.checkpoint_every = checkpoint_every
            self.checkpoint_every_epoch = None
        self.checkpoint_folder = Path(checkpoint_folder)
        self.checkpoint_folder.mkdir(exist_ok=True, parents=True)
        print_model_parameters(self.model)

    def log(self, **data_kwargs):
        self.accelerator.log(data_kwargs, step=self.step.item())

    @property
    def device(self):
        return self.unwrapped_model.device

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def wait(self):
        return self.accelerator.wait_for_everyone()

    def print(self, msg):
        return self.accelerator.print(msg)

    def save(self, path, overwrite = True):
        path = Path(path)
        assert overwrite or not path.exists()

        pkg = dict(
            model = self.unwrapped_model.state_dict(),
            optimizer = self.optimizer.state_dict(),
            step = self.step.item(),
        )

        torch.save(pkg, str(path))

    def load(self, path):
        path = Path(path)
        assert path.exists()

        pkg = torch.load(str(path), map_location = 'cpu')

        self.model.load_state_dict(pkg['model'], strict=False)
        self.optimizer.load_state_dict(pkg['optimizer'])
        self.step.copy_(pkg['step'])

    def next_data_to_forward_kwargs(self, dl_iter) -> dict:
        data = next(dl_iter)

        if isinstance(data, tuple):
            forward_kwargs = dict(zip(self.data_kwargs, data))

        elif isinstance(data, dict):
            forward_kwargs = data

        return forward_kwargs

    def forward(self):
        step = self.step.item()
        dl_iter = cycle(self.dataloader)

        if self.should_validate:
            val_dl_iter = cycle(self.val_dataloader)

        while step < self.num_train_steps:

            for i in range(self.grad_accum_every):
                is_last = i == (self.grad_accum_every - 1)
                maybe_no_sync = partial(self.accelerator.no_sync, self.model) if not is_last else nullcontext

                forward_kwargs = self.next_data_to_forward_kwargs(dl_iter)

                with self.accelerator.autocast(), maybe_no_sync():

                    loss = self.model(**forward_kwargs)

                    loss, (
                        eos_loss,
                        mos_loss,
                        width_loss,
                        thickness_loss,
                        translation_loss,
                    ) = loss

                    self.accelerator.backward(loss / self.grad_accum_every)

            if divisible_by(step, 10):
                self.print(f'step: {step} | total loss: {loss.item():.3f} | eos loss: {eos_loss.item():.3f} | mos loss: {mos_loss.item():.3f} | \
width loss: {width_loss.item():.3f} | thickness loss: {thickness_loss.item():.3f} | translation loss: {translation_loss.item():.3f}')

            self.log(
                total_loss=loss.item(),
                eos_loss=eos_loss.item(),
                mos_loss=mos_loss.item(),
                width_loss=width_loss.item(),
                thickness_loss=thickness_loss.item(),
                translation_loss=translation_loss.item(),
                learning_rate=self.optimizer.scheduler.get_last_lr()[0]
            )

            self.optimizer.step()
            self.optimizer.zero_grad()

            step += 1
            self.step.add_(1)

            self.wait()

            if self.is_main and self.should_validate and divisible_by(step, self.val_every):
                total_val_loss = 0.
                total_val_eos_loss = 0.
                total_val_mos_loss = 0.
                total_val_width_loss = 0.
                total_val_thickness_loss = 0.
                total_val_translation_loss = 0.

                self.unwrapped_model.eval()

                num_val_batches = self.val_num_batches * self.grad_accum_every

                for _ in range(num_val_batches):
                    with self.accelerator.autocast(), torch.no_grad():

                        forward_kwargs = self.next_data_to_forward_kwargs(val_dl_iter)
                        forward_kwargs = torch_to(forward_kwargs, self.device)

                        val_loss = self.unwrapped_model(**forward_kwargs)

                        val_loss, (
                            val_eos_loss,
                            val_mos_loss,
                            val_width_loss,
                            val_thickness_loss,
                            val_translation_loss,
                        ) = val_loss
                        total_val_eos_loss += (val_eos_loss / num_val_batches)
                        total_val_mos_loss += (val_mos_loss / num_val_batches)
                        total_val_width_loss += (val_width_loss / num_val_batches)
                        total_val_thickness_loss += (val_thickness_loss / num_val_batches)
                        total_val_translation_loss += (val_translation_loss / num_val_batches)

                self.print(f'valid eos loss: {total_val_eos_loss.item():.3f} | valid mos loss: {total_val_mos_loss.item():.3f} | \
valid width loss: {total_val_width_loss.item():.3f} | valid thickness loss: {total_val_thickness_loss.item():.3f} | \
valid translation loss: {total_val_translation_loss.item():.3f}')
                self.log(
                    val_eos_loss=total_val_eos_loss.item(),
                    val_mos_loss=total_val_mos_loss.item(),
                    val_width_loss=total_val_width_loss.item(),
                    val_thickness_loss=total_val_thickness_loss.item(),
                    val_translation_loss=total_val_translation_loss.item(),
                )

                self.unwrapped_model.train()

            self.wait()

            if self.is_main and divisible_by(step, self.checkpoint_every):
                if exists(self.checkpoint_every_epoch):
                    checkpoint_num = step * self.batch_size * self.grad_accum_every // self.train_len
                else:
                    checkpoint_num = step // self.checkpoint_every
                self.save(self.checkpoint_folder / f'mesh-transformer.ckpt.{checkpoint_num}.pt')

            self.wait()

        self.print('training complete')
