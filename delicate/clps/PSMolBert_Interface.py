# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 19:35:38 2022

@author: yuyi6
"""
from torch import nn
import logging
from abc import abstractmethod
import pytorch_lightning as pl
from typing import Dict, Tuple
from torch.utils.data.dataloader import DataLoader
from argparse import Namespace
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, StepLR
from transformers.modeling_albert import AlbertModel, AlbertTransformer, AlbertPreTrainedModel
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup)
from molbert.datasets.dataloading import MolbertDataLoader
from transformers.modeling_transfo_xl import PositionalEmbedding

PSMolbertBatchType = Tuple[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], torch.Tensor]

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class SuperPositionalEmbedding(PositionalEmbedding):
    """
    Same as PositionalEmbedding in XLTransformer, BUT has different batch dimension
    """

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        pos_emb = pos_emb.unsqueeze(0)
        if bsz is not None:
            pos_emb = pos_emb.expand(bsz, -1, -1)
        return pos_emb

class SuperPositionalPSBertEmbeddings(nn.Module):
    """
    Same as BertEmbeddings, using non-learnt (computed) positional embeddings
    """

    def __init__(self, config):
        super(SuperPositionalPSBertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = SuperPositionalEmbedding(config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, inputs_embeds=None):
        # do word embedding first to determine its type (float or half)
        words_embeddings = self.word_embeddings(input_ids)

        # if position_ids or token_type_ids were not provided, used defaults
        if position_ids is None:
            seq_length = input_ids.size(1)
            position_ids = torch.arange(seq_length, dtype=words_embeddings.dtype, device=words_embeddings.device)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if inputs_embeds is None:
            inputs_embeds = words_embeddings
        position_embeddings = self.position_embeddings(position_ids, input_ids.size(0))
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class PSBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output 
    
    
class PSBert_Model(AlbertModel):
    
    def __init__(self, config):
        super().__init__(config)
        
        # self.embedding_size=768
        self.embeddings = SuperPositionalPSBertEmbeddings(config)
        self.encoder = AlbertTransformer(config)
      
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()
        # self.pooler_activation = nn.Tanh()
        self.init_weights()
    
class PSBertModel_task(AlbertPreTrainedModel):
    """
    add task
    """

    def __init__(self, config, tasks: nn.ModuleList):
        super().__init__(config)
        self.psbert = PSBert_Model(config)
        self.init_weights()
        self.tasks = tasks

    def forward(self, input_ids, token_type_ids, attention_mask):
        sequence_output, pooled_output = self.psbert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        
        return {task.name: task(sequence_output, pooled_output) for task in self.tasks}
    
class PSMolbertModel(pl.LightningModule):
    def __init__(self, args: Namespace): 
        super().__init__()
        self.hparams = args

        self._datasets = None

        self.config = self.get_config()
        self.tasks = self.get_tasks(self.config)
        if len(self.tasks) == 0:
            raise ValueError('You did not specify any tasks... exiting.')

        self.model = PSBertModel_task(self.config, nn.ModuleList(self.tasks))

    def forward(self, batch_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward method.
        Args:
            batch_inputs: including 'input_ids', 'token_type_ids' and 'attention_mask'
        Returns:
            Returns dictionary of outputs, different depending on the model type and tasks
        """
        return self.model(**batch_inputs)

    def step(self, batch: PSMolbertBatchType, mode: str):
        """
        For a certain batch, performs a forward step and evaluates the losses
        Args:
            batch: Contains three components:
                - input dictionary for the batch with keys 'input_ids', 'token_type_ids' and 'attention_mask';
                - label dictionary of the expected outputs such as 'lm_label_ids', 'unmasked_lm_label_ids' and
                additional ones, depending on the tasks;
                - and an array of masks (should be all true) with the length of the true batch size
            mode: 'train', 'valid' or 'test'
        Returns:
            Returns dictionary of logs
        """
        (batch_inputs, batch_labels), _ = batch

        y_hat = self.forward(batch_inputs)

        losses = self.evaluate_losses(batch_labels, y_hat)
        loss = torch.sum(torch.stack(list(losses.values())))
        tensorboard_logs = {f'{mode}_loss': loss, **losses}
        return {'loss': loss, f'{mode}_loss': loss, 'log': tensorboard_logs}

    def training_step(self, batch: PSMolbertBatchType, batch_idx: int) -> Dict[str, torch.Tensor]:
        return self.step(batch, 'train')

    def training_epoch_end(self, outputs) -> Dict[str, Dict[str, torch.Tensor]]:
        # OPTIONAL
        avg_loss = torch.stack([x['train_loss'] for x in outputs]).mean()
        tensorboard_logs = {'train_loss': avg_loss}
        return {'log': tensorboard_logs}

    def validation_step(self, batch: PSMolbertBatchType, batch_idx: int) -> Dict[str, torch.Tensor]:
        return self.step(batch, 'valid')

    def validation_epoch_end(self, outputs) -> Dict[str, Dict[str, torch.Tensor]]:
        # OPTIONAL
        avg_loss = torch.stack([x['valid_loss'] for x in outputs]).mean()
        tensorboard_logs = {'valid_loss': avg_loss}
        return {'log': tensorboard_logs}

    def test_step(self, batch: PSMolbertBatchType, batch_idx: int) -> Dict[str, torch.Tensor]:
        return self.step(batch, 'test')

    def test_epoch_end(self, outputs) -> Dict[str, Dict[str, torch.Tensor]]:
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'log': tensorboard_logs}

    def evaluate_losses(self, batch_labels, batch_predictions):
        loss_dict = {task.name: task.compute_loss(batch_labels, batch_predictions) for task in self.tasks}
        return loss_dict

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = self._initialise_lr_scheduler(optimizer)

        return [optimizer], [scheduler]

    def _initialise_lr_scheduler(self, optimizer):

        num_batches = len(self.datasets['train']) // self.hparams.batch_size
        num_training_steps = num_batches // self.hparams.accumulate_grad_batches * self.hparams.max_epochs
        warmup_steps = int(num_training_steps * self.hparams.warmup_proportion)

        if self.hparams.learning_rate_scheduler == 'linear_with_warmup':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
            )
        elif self.hparams.learning_rate_scheduler == 'cosine_with_hard_restarts_warmup':
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps, num_cycles=1
            )
        elif self.hparams.learning_rate_scheduler == 'cosine_schedule_with_warmup':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
            )
        elif self.hparams.learning_rate_scheduler == 'constant_schedule_with_warmup':
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)

        elif self.hparams.learning_rate_scheduler == 'cosine_annealing_warm_restarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, warmup_steps)
        elif self.hparams.learning_rate_scheduler == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(optimizer)
        elif self.hparams.learning_rate_scheduler == 'constant':
            scheduler = StepLR(optimizer, 10, gamma=1.0)
        else:
            raise ValueError(
                f'learning_rate_scheduler needs to be one of '
                f'linear_with_warmup, cosine_with_hard_restarts_warmup, cosine_schedule_with_warmup, '
                f'constant_schedule_with_warmup, cosine_annealing_warm_restarts, reduce_on_plateau, '
                f'step_lr. '
                f'Given: {self.hparams.learning_rate_scheduler}'
            )

        logger.info(
            f'SCHEDULER: {self.hparams.learning_rate_scheduler} '
            f'num_batches={num_batches} '
            f'num_training_steps={num_training_steps} '
            f'warmup_steps={warmup_steps}'
        )

        return {'scheduler': scheduler, 'monitor': 'valid_loss', 'interval': 'step', 'frequency': 1}

    def get_config(self):
        raise NotImplementedError

    def get_tasks(self, config):
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        dataset = self.datasets['train']
        return self._get_dataloader(dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        dataset = self.datasets['valid']
        return self._get_dataloader(dataset)

    def test_dataloader(self) -> DataLoader:
        dataset = self.datasets['test']
        return self._get_dataloader(dataset)

    def _get_dataloader(self, dataset, **kwargs):
        return MolbertDataLoader(
            dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, **kwargs
        )

    @property
    def datasets(self):
        if self._datasets is None:
            self._datasets = self.load_datasets()

        return self._datasets

    @abstractmethod
    def load_datasets(self):
        raise NotImplementedError