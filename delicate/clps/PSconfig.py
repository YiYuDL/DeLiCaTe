# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 22:26:47 2022

@author: yuyi6
"""
from transformers.configuration_albert import AlbertConfig

class PSBertConfig(AlbertConfig):
    """
    Same as BertConfig, BUT
    adds any kwarg as a member field
    """

    def __init__(
        self,
        vocab_size = 42,
        hidden_size=768,
        num_hidden_layers=12,
        embedding_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        **kwargs,
    ):
        super(PSBertConfig, self).__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            embedding_size=embedding_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
        )

        for k, v in kwargs.items():
            setattr(self, k, v)