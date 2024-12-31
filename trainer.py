from typing import Dict, List, Tuple, Optional, Any, Union
from transformers.trainer import Trainer
from torch import nn
from torch.utils.data import Dataset
import torch


class DSITrainer(Trainer):
    def __init__(self, restrict_decode_vocab, id_max_length, **kwds):
        super().__init__(**kwds)
        self.restrict_decode_vocab = restrict_decode_vocab
        self.id_max_length = id_max_length

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels']).loss
        if return_outputs:
            return loss, [None, None]  # fake outputs
        return loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()
        # eval_loss = super().prediction_step(model, inputs, True, ignore_keys)[0]
        inputs['labels'] = inputs['labels'].to(self.args.device)
        with torch.no_grad():
            # Greedy search
            # doc_ids = model.generate(
            #     inputs['input_ids'].to(self.args.device),
            #     max_length=20,
            #     prefix_allowed_tokens_fn=self.restrict_decode_vocab,
            #     early_stopping=True,)

            # Beam search
            num_beam_search = 50 # for train faster
            batch_beams = model.generate( # shape: batch_size x num_beam_search, a (a <= id_max_length)
                inputs['input_ids'].to(self.args.device),
                max_length=self.id_max_length, # max output length
                num_beams=num_beam_search,
                prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                num_return_sequences=num_beam_search,
                early_stopping=True, )

            if batch_beams.shape[-1] < self.id_max_length:
                batch_beams = self._pad_tensors_to_max_len(batch_beams, self.id_max_length)

            inputs['labels'] = self._pad_tensors_to_max_len(inputs['labels'], self.id_max_length)

            batch_beams = batch_beams.reshape(inputs['input_ids'].shape[0], self.id_max_length, -1) # shape: batch_size, id_max_length, num_beam_search

        return (None, batch_beams, inputs['labels'])

    # def _pad_tensors_to_max_len(self, tensor, max_length):
    #     if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
    #         # If PAD token is not defined at least EOS token has to be defined
    #         pad_token_id = (
    #             self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
    #         )
    #     else:
    #         if self.model.config.pad_token_id is not None:
    #             pad_token_id = self.model.config.pad_token_id
    #         else:
    #             raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")
    #     tensor[tensor == -100] = self.tokenizer.pad_token_id
    #     padded_tensor = pad_token_id * torch.ones(
    #         (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
    #     )
    #     padded_tensor[:, : tensor.shape[-1]] = tensor
    #     return padded_tensor
    
    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined, use EOS token
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model to pad tensors")

        # Replace -100 with pad_token_id for compatibility
        tensor[tensor == -100] = pad_token_id

        # Check if tensor is 3D or higher
        if len(tensor.shape) > 2:
            # Create a new padded tensor for each slice along the batch dimension
            padded_tensors = []
            for i in range(tensor.shape[0]):
                padded_tensor = pad_token_id * torch.ones(
                    (tensor.shape[1], max_length), dtype=tensor.dtype, device=tensor.device
                )
                padded_tensor[:, :tensor.shape[-1]] = tensor[i]
                padded_tensors.append(padded_tensor)

            # Stack all padded tensors to form the final padded tensor
            padded_tensor = torch.stack(padded_tensors, dim=0)
        else:
            # Handle 2D tensor case
            padded_tensor = pad_token_id * torch.ones(
                (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
            )
            padded_tensor[:, :tensor.shape[-1]] = tensor

        return padded_tensor


class DocTqueryTrainer(Trainer):
    def __init__(self, do_generation: bool, **kwds):
        super().__init__(**kwds)
        self.do_generation = do_generation

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels']).loss
        if return_outputs:
            return loss, [None, None]  # fake outputs
        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        if not self.do_generation:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        outputs = self.model.generate(
            input_ids=inputs[0]['input_ids'].to(self.args.device),
            attention_mask=inputs[0]['attention_mask'].to(self.args.device),
            max_length=self.max_length,
            do_sample=True,
            top_k=self.top_k,
            num_return_sequences=self.num_return_sequences)
        labels = torch.tensor(inputs[1], device=self.args.device).repeat_interleave(self.num_return_sequences)

        if outputs.shape[-1] < self.max_length:
            outputs = self._pad_tensors_to_max_len(outputs, self.max_length)
        return (None, outputs.reshape(inputs[0]['input_ids'].shape[0], self.num_return_sequences, -1),
                labels.reshape(inputs[0]['input_ids'].shape[0], self.num_return_sequences, -1))

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    def predict(
            self,
            test_dataset: Dataset,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "test",
            max_length: Optional[int] = None,
            num_return_sequences: Optional[int] = None,
            top_k: Optional[int] = None,
    ):

        self.max_length = max_length
        self.num_return_sequences = num_return_sequences
        self.top_k = top_k
        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
