# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 22:59:12 2022

@author: yuyi6
"""

from argparse import ArgumentParser

from delicate.clps.PSMolBert_Interface import PSMolbertModel
from delicate.clps.PSMolBert_pretrain import PSMolbertpretrainModel
from delicate.clps.appInterface import BaseMolAlbertApp


class SmilesMolAlbertApp(BaseMolAlbertApp):
    @staticmethod
    def get_model(args) -> PSMolbertModel:
        
        model = PSMolbertpretrainModel(args)
        return model

    @staticmethod
    def add_parser_arguments(parser: ArgumentParser) -> ArgumentParser:
        """
        Adds model specific options to the default parser
        """
        
        parser.add_argument(
            '--num_physchem_properties', type=int, default=49, help='Adds physchem property task (how many to predict)'
        )
        parser.add_argument('--is_same_smiles', type=int, default=0, help='Adds is_same_smiles task')
        parser.add_argument('--permute', type=int, default=0, help='Permute smiles')
        parser.add_argument(
            '--named_descriptor_set', type=str, default='surface', help='What set of descriptors to use ("all" or "simple")'
        )
        parser.add_argument('--vocab_size', default=42, type=int, help='Vocabulary size for smiles index featurizer')
        parser.add_argument('--num_hidden_layers', default=12, type=int, help='number of hidden layers for PSMolBERT')
        parser.add_argument('--num_workers', default=0, type=int)
        return parser



if __name__ == '__main__':
    SmilesMolAlbertApp().run()
