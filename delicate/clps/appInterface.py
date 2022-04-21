# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 23:54:33 2022

@author: yuyi6
"""

import logging
import pprint
from abc import ABC
from argparse import ArgumentParser, Namespace

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from molbert.apps.args import get_default_parser
from delicate.clps.PSMolBert_Interface import PSMolbertModel

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class BaseMolAlbertApp(ABC):
    @staticmethod
    def load_model_weights(model: PSMolbertModel, checkpoint_file: str) -> PSMolbertModel:
        """
        PL `load_from_checkpoint` seems to fail to reload model weights. This function loads them manually.
        See: https://github.com/PyTorchLightning/pytorch-lightning/issues/525
        """
        logger.info(f'Loading model weights from {checkpoint_file}')
        checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)

        model.load_state_dict(checkpoint['state_dict'], strict=False)
        return model

    def run(self, args=None):
        args = self.parse_args(args)
        seed_everything(args.seed)

        pprint.pprint('args')
        pprint.pprint(args.__dict__)
        pprint.pprint('*********************')

        checkpoint_callback = ModelCheckpoint(monitor='valid_loss', verbose=True,
                                              save_weights_only = True ,save_last=True)

        logger.info(args)

        lr_logger = LearningRateLogger()

        trainer = Trainer(
            default_root_dir=args.default_root_dir,
            progress_bar_refresh_rate=args.progress_bar_refresh_rate,
            min_epochs=args.min_epochs,
            max_epochs=args.max_epochs,
            val_check_interval=args.val_check_interval,
            limit_val_batches=args.limit_val_batches,
            gpus=args.gpus,
            distributed_backend=args.distributed_backend,
            row_log_interval=1,
            amp_level=args.amp_level,
            precision=args.precision,
            num_nodes=args.num_nodes,
            tpu_cores=args.tpu_cores,
            accumulate_grad_batches=args.accumulate_grad_batches,
            checkpoint_callback=checkpoint_callback,
            resume_from_checkpoint=args.resume_from_checkpoint,
            fast_dev_run=args.fast_dev_run,
            callbacks=[lr_logger],
        )

        model = self.get_model(args)
        logger.info(f'Start Training model {model}')

        logger.info('')
        trainer.fit(model)
        logger.info('Training loop finished.')

        return trainer

    def parse_args(self, args) -> Namespace:
        """
        Parse command line arguments
        """
        parser = get_default_parser()
        parser = self.add_parser_arguments(parser)
        return parser.parse_args(args=args)

    @staticmethod
    def get_model(args):
        raise NotImplementedError

    @staticmethod
    def add_parser_arguments(parser: ArgumentParser) -> ArgumentParser:
        """
        Adds model specific options to the default parser
        """
        raise NotImplementedError