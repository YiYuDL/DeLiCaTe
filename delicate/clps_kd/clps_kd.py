# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 14:36:19 2022

@author: yuyi6
"""

import logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    )
logger = logging.getLogger("Main")

import os,random
import numpy as np
import torch
from torch.utils.data import RandomSampler
from delicate.kd.kd_dataloading import kdDataLoader
from molbert.datasets.smiles import BertSmilesDataset
from molbert.utils.lm_utils import get_seq_lengths
from molbert.utils.featurizer.molfeaturizer import SmilesIndexFeaturizer
from delicate.kd.kd_utils import divide_parameters
import delicate.kd.kd_config
from transformers import AdamW, get_linear_schedule_with_warmup
from delicate.clps_kd.delicate_model import PSMolbertdistill,PSMolBertconfig
from textbrewer import DistillationConfig, TrainingConfig, GeneralDistiller
from delicate.kd.predict_function import predict
from functools import partial

def args_check(args):
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        logger.warning("Output directory () already exists and is not empty.")
    

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count() if not args.no_cuda else 0
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))
    args.n_gpu = n_gpu
    args.device = device
    return device, n_gpu

# def evaluation()

def main():
    #parse arguments
    delicate.kd.kd_config.parse()
    args = delicate.kd.kd_config.args
    for k,v in vars(args).items():
        logger.info(f"{k}:{v}")
    #set seeds
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    #arguments check
    device, n_gpu = args_check(args)
    os.makedirs(args.output_dir, exist_ok=True)
    forward_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    args.forward_batch_size = forward_batch_size
    
    #load bert config
    Molbert_config = PSMolBertconfig.from_json_file(args.bert_config_file_T)
    KDMolbert_config = PSMolBertconfig.from_json_file(args.bert_config_file_S)
    
    model_T = PSMolbertdistill(Molbert_config)
    model_S = PSMolbertdistill(KDMolbert_config)
    
    #Load teacher
    # state_dict_T = torch.load(args.tuned_checkpoint_T, map_location=lambda storage, loc: storage)
    weight_T = torch.load(args.tuned_checkpoint_T)
    model_T.load_state_dict(weight_T, strict=False)
    # model_T.eval()
   
    #Load student
    logger.info("Student Model loaded")
    logger.info("Model is randomly initialized.")
    model_T.to(device)
    model_S.to(device)  
    
    #dataloader
    single_seq_len, total_seq_len = get_seq_lengths(128, 0)
    featurizer = SmilesIndexFeaturizer.bert_smiles_index_featurizer(128)
    train_dataset = None
    eval_dataset  = None
    
    train_dataset = BertSmilesDataset(
        input_path= args.train_file,
        featurizer=featurizer,
        single_seq_len=single_seq_len,
        total_seq_len=total_seq_len,
        num_physchem=0,
        is_same= False,
        permute = True)
       
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = kdDataLoader(train_dataset,
        sampler=train_sampler, batch_size=args.train_batch_size, num_workers=args.num_workers) 
    
    eval_dataset = BertSmilesDataset(
        input_path= args.predict_file,
        featurizer=featurizer,
        single_seq_len=single_seq_len,
        num_physchem=0,
        total_seq_len=total_seq_len,
        is_same= False,
        permute = True)
    
    #optimizer and schedule
    params = list(model_S.named_parameters())
    all_trainable_params = divide_parameters(params, lr=args.learning_rate)
    logger.info("Length of all_trainable_params: %d", len(all_trainable_params))
    optimizer = AdamW(all_trainable_params,lr=args.learning_rate)
    
    scheduler_class = get_linear_schedule_with_warmup
    num_train_steps = int(len(train_dataloader)//args.gradient_accumulation_steps * args.num_train_epochs)
    scheduler_args = {'num_warmup_steps': int(args.warmup_proportion*num_train_steps),
                      'num_training_steps': num_train_steps}
    
    ########## DISTILLATION ###########
    train_config = TrainingConfig(
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        ckpt_frequency = args.ckpt_frequency,
        log_dir = args.output_dir,
        output_dir = args.output_dir,
        fp16 = args.fp16,
        device = args.device)
    
    #match
    from delicate.kd.matches import matches
    intermediate_matches = None
    if isinstance(args.matches,(list,tuple)):
        intermediate_matches = []
        for match in args.matches:
            intermediate_matches += matches[match]
    logger.info(f"{intermediate_matches}")
    distill_config = DistillationConfig(
        temperature=args.temperature,
        intermediate_matches=intermediate_matches)
    
    #adaptor
    def simple_adaptor(batch, model_outputs):
        # The second element of model_outputs is the logits before softmax
        # The third element of model_outputs is hidden states
        return {'logits': (model_outputs[0],),
                'hidden': model_outputs[2]}

    distiller = GeneralDistiller(train_config = train_config,
                                 distill_config = distill_config, 
                                 model_T = model_T, model_S = model_S,
                                 adaptor_T = simple_adaptor,
                                 adaptor_S = simple_adaptor)
    callback_func = partial(predict, eval_datasets=eval_dataset, args=args)
    with distiller:
        distiller.train(optimizer, scheduler_args=scheduler_args, dataloader=train_dataloader,
                        scheduler_class=scheduler_class ,num_epochs=args.num_train_epochs,callback=callback_func,max_grad_norm=1)
            

if __name__ == "__main__":
    main()    