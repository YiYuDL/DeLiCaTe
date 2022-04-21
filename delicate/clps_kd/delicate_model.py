# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 08:50:39 2022

@author: yuyi6
"""

from transformers.modeling_bert import BertLMPredictionHead
from torch import nn
import torch
from delicate.clps.PSMolBert_Interface import PSBert_Model
from transformers.configuration_albert import AlbertConfig
from transformers.modeling_albert import AlbertPreTrainedModel
BertLayerNorm = torch.nn.LayerNorm
from torch.nn.modules.loss import CrossEntropyLoss


class PSMolBertconfig(AlbertConfig):
    """
    Same as AlBertConfig, BUT
    adds any kwarg as a member field
    """

    def __init__(
        self,
        vocab_size_or_config_json_file = 42,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        embedding_size=768,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        num_physchem_properties=0,
        type_vocab_size=2,
        initializer_range=0.02,
        **kwargs,
    ):
        super(PSMolBertconfig, self).__init__(
            vocab_size_or_config_json_file,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            embedding_size=768,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            num_physchem_properties=0,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
        )

        for k, v in kwargs.items():
            setattr(self, k, v)

def initializer_builder(std):
    _std = std
    def init_bert_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=_std)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    return init_bert_weights

class PSMolbertdistill(AlbertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.psbert = PSBert_Model(config)
     
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        initializer = initializer_builder(0.02)
        self.apply(initializer)
        self.masked_lm_head = BertLMPredictionHead(config)

        self.loss_lm = CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = config.vocab_size
       

    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        sequence_output, pooled_output, hidden_states, attentions  = self.albert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
            output_hidden_states=True, output_attentions=True)
       
        loss = None
        pre_lm = self.masked_lm_head(sequence_output)
        logits = pre_lm
        if labels is not None:
            
            loss_lm = self.loss_lm(
                pre_lm.view(-1, self.vocab_size), labels.view(-1))
        
            loss = loss_lm
        
        return logits, pooled_output, hidden_states, attentions, loss