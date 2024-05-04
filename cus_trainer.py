from torch import nn
from transformers import Trainer
import inspect
import torch
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import is_sagemaker_mp_enabled
from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollator
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_callback import TrainerCallback
# from transformers.trainer_pt_utils import smp_forward_only, smp_nested_concat

class PartialLabelTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
        self.loss = 'nll'
        self.weight = None
        self.method = 'proden'
        self.num_workers = args.num_workers

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label_1", "label_2", "label_3", "label_4", "label_5", "label_6", "label", "label_ids"] + self.label_names))


    def compute_loss(self, model, inputs, return_outputs=False):
        
        labels = inputs.pop("label_1")
        labels_vec = torch.zeros((len(labels), 2)).cuda()
        for i in range(self.num_workers):
            if i != 0:
                labels = inputs.pop("label_" + str(i+1))
            labels_onehot = torch.nn.functional.one_hot(labels, num_classes=2).float()
            labels_vec += labels_onehot
        labels_vec[labels_vec > 0] = 1
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if self.weight == None:
            self.weight = torch.nn.functional.normalize(labels_vec, p=1,dim=1)
        else:
            if self.method == 'proden':
                self.weight = torch.nn.functional.normalize(logits, p=1,dim=1)

        # forward pass
        if self.loss == "nll":
            log_probs = -nn.functional.log_softmax(logits, dim=-1)
            weighted_log_probs = torch.mul(log_probs, self.weight)
            loss = torch.mean(weighted_log_probs)
        elif self.loss == "mse":
            probs = nn.functional.softmax(logits, dim=-1)
            loss = torch.mean((self.weight - probs)**2)

        return (loss, outputs) if return_outputs else loss
    
    def compute_pred_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # weight_1 = torch.mean(labels, axis=0, dtype=float)
        # weight_0 = torch.sub(torch.ones_like(weight_1), weight_1)
        weight = torch.nn.functional.one_hot(labels, num_classes=2).float()
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        weighted_log_probs = torch.mul(log_probs, weight)
        loss = torch.mean(weighted_log_probs)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
            self,
            model: nn.Module,
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

            with torch.no_grad():
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
                            loss, outputs = self.compute_pred_loss(model, inputs, return_outputs=True)
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
                return (loss, None, None)

            logits = nested_detach(logits)
            if len(logits) == 1:
                logits = logits[0]

            return (loss, logits, labels)


class SoftLabelTrainer(PartialLabelTrainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
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
        self.worker_quality_vec = torch.ones(self.num_workers)
        self.loss = "mse"

        exclude_claude = True

        if exclude_claude:
            self.worker_quality_vec[2] = 0


    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("label_1")
        labels_vec = torch.zeros((len(labels), 2)).cuda()
        for i in range(self.num_workers):
            if i != 0:
                labels = inputs.pop("label_" + str(i+1))
            labels_onehot = torch.nn.functional.one_hot(labels, num_classes=2).float() * self.worker_quality_vec[i]
            labels_vec += labels_onehot
        weight = labels_vec/torch.sum(labels_vec)
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if self.loss == "nll":
            log_probs = -nn.functional.log_softmax(logits, dim=-1)
            weighted_log_probs = torch.mul(log_probs, weight)
            loss = torch.mean(weighted_log_probs)
        elif self.loss == "mse":
            probs = nn.functional.softmax(logits, dim=-1)
            loss = torch.mean((weight - probs)**2)

        return (loss, outputs) if return_outputs else loss







