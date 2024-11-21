from transformers.integrations import (
    get_reporting_integration_callbacks,
    hp_params,
    is_fairscale_available,
)
import pandas as pd
import numpy as np
import functools
import os
from packaging import version
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, RandomSampler, SequentialSampler
from transformers import RobertaModel
from transformers.pytorch_utils import is_torch_less_than_1_11
from cus_trainer import PartialLabelTrainer
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.trainer_pt_utils import (nested_detach, 
                                           nested_numpify,
                                           nested_concat,
                                           get_model_param_count, 
                                           get_module_class_from_name,
                                           SequentialDistributedSampler,
                                           IterableDatasetShard,
                                           get_dataloader_sampler,
                                           find_batch_size)
from transformers.utils import (is_sagemaker_mp_enabled, 
                                is_apex_available, 
                                is_sagemaker_mp_enabled, 
                                is_torch_tpu_available,
                                is_accelerate_available, 
                                is_datasets_available,
                                is_torch_tpu_available,
                                is_sagemaker_dp_enabled,
                                is_torch_neuroncore_available,
                                logging)
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.training_args import TrainingArguments, ParallelMode
from transformers.data.data_collator import DataCollator
import datasets
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import (EvalPrediction, 
                                        ShardedDDPOption, 
                                        EvalLoopOutput, 
                                        HPSearchBackend,
                                        denumpify_detensorize,
                                        speed_metrics,
                                        TrainOutput,
                                        has_length)
from transformers.trainer_callback import TrainerCallback, TrainerState
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.dependency_versions_check import dep_version_check
from transformers.optimization import Adafactor, get_scheduler
import inspect 
import math
import sys
import time

if is_apex_available():
    from apex import amp

if is_fairscale_available():
    dep_version_check("fairscale")
    import fairscale
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.nn.wrap import auto_wrap
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import DistributedDataParallelKwargs, GradientAccumulationPlugin

if is_datasets_available():
    import datasets

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


logger = logging.get_logger(__name__)

class CVARTrainer(PartialLabelTrainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        weight_estimator: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        valid_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
        self.num_workers = args.num_workers
        self.weight_estimator = weight_estimator
        self.weight_estimator_wrapped = weight_estimator
        self.weight_optimizer = None
        self.weight_lr_scheduler = None
        self.loss = "nll"
        self.valid_dataset = valid_dataset
        self.group_frac = torch.ones(15, device=weight_estimator.device)
        self.group_frac /= self.group_frac.sum()
        self.step_size = 0.01



    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label_1", "label_2", "label_3", "label_4", "label_5", "label_6", "label", "label_ids"] + self.label_names))

    def training_step(self, model: nn.Module, weight_estimator: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        # if is_sagemaker_mp_enabled():
        #     loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
        #     return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, weight_estimator, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps


    def compute_loss(self, model, weight_estimator, inputs, return_outputs=False):
        labels_list = []
        for i in range(self.num_workers):
            labels = inputs.pop("label_" + str(i+1))
            labels_list.append(labels)

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        weight_outputs = weight_estimator(**inputs)
        worker_logits = weight_outputs.get("logits")
        worker_quality_vec = nn.functional.softmax(worker_logits, dim=-1)
        
        labels_vec = torch.zeros((len(labels), 2)).cuda()
        for i in range(self.num_workers):
            labels_onehot = torch.nn.functional.one_hot(labels_list[i], num_classes=2).float() * worker_quality_vec[:,i].view(1, -1, 1)
            labels_vec += labels_onehot.squeeze(0)
        weight = labels_vec/torch.sum(labels_vec)

        
        if self.loss == "nll":
            log_probs = -nn.functional.log_softmax(logits, dim=-1)
            weighted_log_probs = torch.mul(log_probs, weight)
            loss = torch.mean(weighted_log_probs)
        elif self.loss == "mse":
            probs = nn.functional.softmax(logits, dim=-1)
            loss = torch.mean((weight - probs)**2)

        return (loss, outputs) if return_outputs else loss
    
    def compute_pred_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # for i in range(self.num_workers):
        #     label = inputs.pop("label_" + str(i+1))

        weight = torch.nn.functional.one_hot(labels, num_classes=2).float()
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        weighted_log_probs = torch.mul(log_probs, weight)
        loss = torch.mean(weighted_log_probs)
        return (loss, outputs) if return_outputs else loss
    
    def compute_valid_loss(self, model, weight_estimator, inputs, return_outputs=False):

        labels = inputs.pop("labels")

        labels_list = []
        for i in range(self.num_workers):
            labels_list.append(inputs.pop("label_" + str(i+1)))

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        worker_outputs = weight_estimator(**inputs)
        worker_logits = worker_outputs.get("logits")
    
        worker_quality_vec = nn.functional.softmax(worker_logits, dim=-1)

        labels_onehot = torch.nn.functional.one_hot(labels, num_classes=2).float()

        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        weighted_log_probs = torch.mul(log_probs.clone(), labels_onehot)
        loss = torch.sum(weighted_log_probs, dim=1)
        
        entropy = None
        worker_log_probs = None
        worker_probs = torch.zeros(len(labels), device=labels.device)
        for j in range(len(labels)):
            count = 0
            for i in range(self.num_workers):
                # Check if worker's label matches the ground truth label for sample j
                if labels_list[i][j] == labels[j]:
                    # Accumulate the quality if the worker's label matches the ground truth label for sample j
                   worker_probs[j] = worker_probs[j] + worker_quality_vec[j, i]
                   count += 1
            if count == 0:
                worker_quality = worker_quality_vec[j,:].clone()
                current_entropy = torch.sum(- worker_quality * torch.log(worker_quality))
                current_worker_log_prob = torch.log(torch.min(worker_quality))
            else:
                worker_prob = worker_probs[j].clone()
                current_entropy = - worker_prob * torch.log(worker_prob)
                current_worker_log_prob = torch.log(worker_prob)
            current_entropy = current_entropy.reshape(1)
            entropy = current_entropy if entropy is None else torch.cat((entropy, current_entropy))
            
            current_worker_log_prob = current_worker_log_prob.reshape(1)
            worker_log_probs = current_worker_log_prob if worker_log_probs is None else torch.cat((worker_log_probs, current_worker_log_prob))
        return (loss, outputs) if return_outputs else loss, worker_log_probs, entropy

    def validation_step(
        self,
        model: nn.Module,
        weight_estimator: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False
        

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        if is_sagemaker_mp_enabled():
            raw_outputs = smp_forward_only(model, inputs)
            if has_labels or loss_without_labels:
                if isinstance(raw_outputs, dict):
                    loss_mb = raw_outputs["loss"]
                    logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    loss_mb = raw_outputs[0]
                    logits_mb = raw_outputs[1:]

                loss = loss_mb.reduce_mean().detach().cpu()
                logits = smp_nested_concat(logits_mb)
            else:
                loss = None
                if isinstance(raw_outputs, dict):
                    logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                else:
                    logits_mb = raw_outputs
                logits = smp_nested_concat(logits_mb)
        else:
            if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    result = self.compute_valid_loss(model, weight_estimator, inputs, return_outputs=True)
                    loss = result[0][0]
                    outputs = result[0][1]
                    worker_log_probs = result[1].clone()
                    entropy = result[2].clone()
                loss = loss.mean().detach()

                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
                    
                    
            else:
                loss = None
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, worker_log_probs, entropy, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, worker_log_probs, entropy, logits, labels)  

    def _wrap_weight_estimator(self, weight_estimator, training=True, dataloader=None):
        if self.args.use_ipex:
            dtype = torch.bfloat16 if self.use_cpu_amp else torch.float32
            weight_estimator = self.ipex_optimize_model(weight_estimator, training, dtype=dtype)

        if is_sagemaker_mp_enabled():
            # Wrapping the base model twice in a DistributedModel will raise an error.
            if isinstance(self.weight_estimator_wrapped, smp.model.DistributedModel):
                return self.weight_estimator_wrapped
            return smp.DistributedModel(weight_estimator, backward_passes_per_step=self.args.gradient_accumulation_steps)

        # train/eval could be run multiple-times - if already wrapped, don't re-wrap it again
        if unwrap_model(weight_estimator) is not weight_estimator:
            return weight_estimator

        # Mixed precision training with apex (torch < 1.6)
        if self.use_apex and training:
            weight_estimator, self.weight_optimizer = amp.initialize(weight_estimator, self.weight_optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization) / 8bit models does not support DDP
        if self.args.n_gpu > 1 and not getattr(weight_estimator, "is_loaded_in_8bit", False):
            weight_estimator = nn.DataParallel(weight_estimator)

        if self.args.jit_mode_eval:
            start_time = time.time()
            weight_estimator = self.torch_jit_model_eval(weight_estimator, dataloader, training)
            self.jit_compilation_time = round(time.time() - start_time, 4)

        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
        if not training:
            return weight_estimator

        # Distributed training (should be after apex fp16 initialization)
        if self.sharded_ddp is not None:
            # Sharded DDP!
            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                weight_estimator = ShardedDDP(weight_estimator, self.weight_optimizer)
            else:
                mixed_precision = self.args.fp16 or self.args.bf16
                cpu_offload = ShardedDDPOption.OFFLOAD in self.args.sharded_ddp
                zero_3 = self.sharded_ddp == ShardedDDPOption.ZERO_DP_3
                # XXX: Breaking the self.model convention but I see no way around it for now.
                if ShardedDDPOption.AUTO_WRAP in self.args.sharded_ddp:
                    weight_estimator = auto_wrap(weight_estimator)
                self.weight_estimator = weight_estimator = FullyShardedDDP(
                    weight_estimator,
                    mixed_precision=mixed_precision,
                    reshard_after_forward=zero_3,
                    cpu_offload=cpu_offload,
                ).to(self.args.device)
        # Distributed training using PyTorch FSDP
        elif self.fsdp is not None and self.args.fsdp_config["xla"]:
            try:
                from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
                from torch_xla.distributed.fsdp import checkpoint_module
                from torch_xla.distributed.fsdp.wrap import (
                    size_based_auto_wrap_policy,
                    transformer_auto_wrap_policy,
                )
            except ImportError:
                raise ImportError("Missing XLA FSDP related module; please make sure to use torch-xla >= 2.0.")
            auto_wrap_policy = None
            auto_wrapper_callable = None
            default_transformer_cls_names_to_wrap = getattr(weight_estimator, "_no_split_modules", None)
            fsdp_transformer_layer_cls_to_wrap = self.args.fsdp_config.get(
                "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
            )

            if self.args.fsdp_config["min_num_params"] > 0:
                auto_wrap_policy = functools.partial(
                    size_based_auto_wrap_policy, min_num_params=self.args.fsdp_config["min_num_params"]
                )
            elif fsdp_transformer_layer_cls_to_wrap is not None:
                transformer_cls_to_wrap = set()
                for layer_class in fsdp_transformer_layer_cls_to_wrap:
                    transformer_cls = get_module_class_from_name(weight_estimator, layer_class)
                    if transformer_cls is None:
                        raise Exception("Could not find the transformer layer class to wrap in the model.")
                    else:
                        transformer_cls_to_wrap.add(transformer_cls)

                auto_wrap_policy = functools.partial(
                    transformer_auto_wrap_policy,
                    # Transformer layer class to wrap
                    transformer_layer_cls=transformer_cls_to_wrap,
                )
            fsdp_kwargs = self.args.xla_fsdp_config
            if self.args.fsdp_config["xla_fsdp_grad_ckpt"]:
                # Apply gradient checkpointing to auto-wrapped sub-modules if specified
                def auto_wrapper_callable(m, *args, **kwargs):
                    return FSDP(checkpoint_module(m), *args, **kwargs)

            # Wrap the base model with an outer FSDP wrapper
            self.weight_estimator = weight_estimator = FSDP(
                weight_estimator,
                auto_wrap_policy=auto_wrap_policy,
                auto_wrapper_callable=auto_wrapper_callable,
                **fsdp_kwargs,
            )

            # Patch `xm.optimizer_step` should not reduce gradients in this case,
            # as FSDP does not need gradient reduction over sharded parameters.
            def patched_optimizer_step(weight_optimizer, barrier=False, optimizer_args={}):
                loss = weight_optimizer.step(**optimizer_args)
                if barrier:
                    xm.mark_step()
                return loss

            xm.optimizer_step = patched_optimizer_step
        elif is_sagemaker_dp_enabled():
            weight_estimator = nn.parallel.DistributedDataParallel(
                weight_estimator, device_ids=[int(os.getenv("SMDATAPARALLEL_LOCAL_RANK"))]
            )
        elif self.args.parallel_mode == ParallelMode.DISTRIBUTED:
            if is_torch_neuroncore_available():
                return weight_estimator
            kwargs = {}
            if self.args.ddp_find_unused_parameters is not None:
                kwargs["find_unused_parameters"] = self.args.ddp_find_unused_parameters
            elif isinstance(weight_estimator, PreTrainedModel):
                # find_unused_parameters breaks checkpointing as per
                # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
                kwargs["find_unused_parameters"] = not weight_estimator.is_gradient_checkpointing
            else:
                kwargs["find_unused_parameters"] = True

            if self.args.ddp_bucket_cap_mb is not None:
                kwargs["bucket_cap_mb"] = self.args.ddp_bucket_cap_mb

            if self.args.ddp_broadcast_buffers is not None:
                kwargs["broadcast_buffers"] = self.args.ddp_broadcast_buffers

            self.accelerator.ddp_handler = DistributedDataParallelKwargs(**kwargs)

        return weight_estimator

    def create_weight_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.weight_estimator_wrapped if is_sagemaker_mp_enabled() else self.weight_estimator

        if self.weight_optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.weight_optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.weight_optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            logger.info(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.weight_optimizer = smp.DistributedOptimizer(self.weight_optimizer)

        return self.weight_optimizer

    def create_weight_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if self.weight_lr_scheduler is None:
            self.weight_lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.weight_optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
            self._created_weight_lr_scheduler = True
        return self.weight_lr_scheduler

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        self.create_optimizer()
        self.create_weight_optimizer()
        if IS_SAGEMAKER_MP_POST_1_10 and smp.state.cfg.fp16:
            # If smp >= 1.10 and fp16 is enabled, we unwrap the optimizer
            optimizer = self.optimizer.optimizer
            weight_optimizer = self.weight_optimizer.optimizer
        else:
            optimizer = self.optimizer
            weight_optimizer = self.weight_optimizer
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)
        self.create_weight_scheduler(num_training_steps=num_training_steps, optimizer=weight_optimizer)

    def _get_valid_sampler(self, valid_dataset: Dataset) -> Optional[torch.utils.data.Sampler]:
        # Deprecated code
        if self.args.use_legacy_prediction_loop:
            if is_torch_tpu_available():
                return SequentialDistributedSampler(
                    valid_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
                )
            elif is_sagemaker_mp_enabled():
                return SequentialDistributedSampler(
                    valid_dataset,
                    num_replicas=smp.dp_size(),
                    rank=smp.dp_rank(),
                    batch_size=self.args.per_device_valid_batch_size,
                )
            else:
                return SequentialSampler(valid_dataset)

        if self.args.world_size <= 1:
            return SequentialSampler(valid_dataset)
        else:
            return None

    def get_valid_dataloader(self, valid_dataset: Optional[Dataset] = None) -> DataLoader:
        if valid_dataset is None and self.valid_dataset is None:
            raise ValueError("Trainer: validation requires a valid_dataset.")
        valid_dataset = valid_dataset if valid_dataset is not None else self.valid_dataset
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(valid_dataset, datasets.Dataset):
            valid_dataset = self._remove_unused_columns(valid_dataset, description="validation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="validation")

        dataloader_params = {
            "batch_size": self.args.valid_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(valid_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_valid_sampler(valid_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        return self.accelerator.prepare(DataLoader(valid_dataset, **dataloader_params))


    def validation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:

        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        weight_estimator = self._wrap_weight_estimator(self.weight_estimator, training=True, dataloader=dataloader)
        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped
        

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_valid:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_valid:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.valid_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")



        self.callback_handler.valid_dataloader = dataloader
        # Do this before wrapping.
        valid_dataset = getattr(dataloader, "dataset", None)
        

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None


        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        all_worker_log_probs = None
        all_entropy = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # # Update the observed num examples
            # observed_batch_size = find_batch_size(inputs)
            # if observed_batch_size is not None:
            #     observed_num_examples += observed_batch_size
            #     # For batch samplers, batch_size is not known by the dataloader in advance.
            #     if batch_size is None:
            #         batch_size = observed_batch_size

            # Prediction step
            loss, worker_log_probs, entropy, logits, labels = self.validation_step(model, weight_estimator, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            all_worker_log_probs = worker_log_probs if all_worker_log_probs is None else torch.cat((all_worker_log_probs, worker_log_probs))
            all_entropy = entropy if all_entropy is None else torch.cat((all_entropy, entropy))
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self.accelerator.gather_for_metrics((loss.repeat(batch_size)))
                losses_host = losses if losses_host is None else nested_concat(losses_host, losses, padding_index=-100)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.accelerator.gather_for_metrics((inputs_decode))
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.accelerator.gather_for_metrics((logits))
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)

            if labels is not None:
                labels = self.accelerator.gather_for_metrics((labels))
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if (
                args.eval_accumulation_steps is not None
                and (step + 1) % args.eval_accumulation_steps == 0
                and (self.accelerator.sync_gradients or version.parse(accelerate_version) > version.parse("0.20.3"))
            ):
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )


                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(valid_dataset):
            num_samples = len(valid_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(valid_dataset, IterableDatasetShard) and getattr(valid_dataset, "num_examples", 0) > 0:
            num_samples = valid_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples


        metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples),all_worker_log_probs, all_entropy


    def validate(
        self,
        valid_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "valid",
    ) -> Dict[str, float]:
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        
        valid_dataloader = self.get_valid_dataloader(valid_dataset)
        start_time = time.time()
        valid_loop = self.validation_loop

        output, worker_log_probs, entropy = valid_loop(
            valid_dataloader,
            description="Validation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.valid_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics, worker_log_probs, entropy

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
            or self.is_fsdp_enabled
        )

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

            self.weight_lr_scheduler = None
            self._created_weight_lr_scheduler = False

        
        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)
            self.weighter_optimizer, self.weight_lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.weight_estimator.gradient_checkpointing_enable()
        

        model = self._wrap_model(self.model_wrapped)
        weight_estimator = self._wrap_model(self.weight_estimator_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # Fairscale Sharded DDP, FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self.model = self.accelerator.prepare(self.model)
                self.weight_estimator = self.accelerator.prepare(self.weight_estimator)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            self.weight_estimator.eval()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
            if hasattr(self.weight_lr_scheduler, "step"):
                if self.use_apex:
                    weight_estimator = self.accelerator.prepare(self.weight_estimator)
                else:
                    weight_estimator, self.weight_optimizer = self.accelerator.prepare(self.weight_estimator, self.weight_optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                weight_estimator, self.weight_optimizer, self.weight_lr_scheduler = self.accelerator.prepare(
                    self.weight_estimator, self.weight_optimizer, self.weight_lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model
            self.weight_estimator = self.weight_estimator_wrapped = weight_estimator

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model
        if weight_estimator is not self.weight_estimator:
            self.weight_estimator_wrapped = weight_estimator

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped
            self.weight_deepspeed = self.weight_estimator_wrapped
        
        

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)
            self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # Check if saved optimizer or scheduler states exist
        
        
        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters of classifier = {get_model_param_count(model, trainable_only=True):,}")
        logger.info(f"  Number of trainable parameters of RL agent = {get_model_param_count(weight_estimator, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.callback_handler.weight_estimator = self.weight_estimator
        self.callback_handler.weight_optimizer = self.weight_optimizer
        self.callback_handler.weight_lr_scheduler = self.weight_lr_scheduler        
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                is_random_sampler = isinstance(sampler, RandomSampler)
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)

        df = pd.read_csv(args.valid_file)
        group_vec = df['dominant_topic'].tolist()

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, weight_estimator, inputs)


                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step


                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc or (
                        version.parse(accelerate_version) <= version.parse("0.20.3")
                    ):
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            if is_torch_tpu_available():
                                gradients = xm._fetch_gradients(self.optimizer)
                                xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.weight_optimizer.clip_master_grads(args.max_grad_norm)
                        elif hasattr(self.weight_optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.weight_optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(weight_estimator, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            weight_estimator.clip_grad_norm_(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.weight_optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            self.accelerator.clip_grad_norm_(
                                weight_estimator.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            # tpu-comment: accelerate wrapped optimizers call xm.optimizer_step
                            self.optimizer.step()
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()
                        optimizer_was_run = not self.accelerator.optimizer_step_was_skipped

                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()
                    
                    model.zero_grad()                    
                    
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                    
                    
                    if step % 1 == 0:
                        self.weight_optimizer.zero_grad()
                        model.eval()
                        self.weight_estimator.train()

                        indices = np.random.choice(1000,20)
                        valid_dataset = datasets.Dataset.from_dict(self.valid_dataset[indices])

                        metrics, worker_log_probs, entropy = self.validate(valid_dataset=valid_dataset,
                                                        ignore_keys=ignore_keys_for_eval) 

                        pg_loss = - worker_log_probs

                        with torch.no_grad():
                            p = torch.ones_like(pg_loss) / pg_loss.shape[0]
                        
                        m = p.shape[0]

                        with torch.no_grad():
                            idx = torch.nonzero(p)  # where is annoyingly backwards incompatible
                            kl = np.log(m) + (p[idx] * torch.log(p[idx])).sum()

                        pg_loss = torch.dot(p, pg_loss)

                        pg_loss.backward()
                        self.weight_optimizer.step()
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        return TrainOutput(self.state.global_step, train_loss, metrics)
