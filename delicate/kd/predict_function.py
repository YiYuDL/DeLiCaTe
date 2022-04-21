
import torch
from torch.utils.data import SequentialSampler,DistributedSampler
from tqdm import tqdm
import logging
from delicate.kd.kd_dataloading import kdDataLoader
logger = logging.getLogger(__name__)

def predict(model,eval_datasets,step,args):
    
    logger.info("Predicting...")
    logger.info("***** Running predictions *****")
    logger.info("  Num  examples = %d", len(eval_datasets))
    logger.info("  Batch size = %d", args.predict_batch_size)
    eval_sampler = SequentialSampler(eval_datasets) if args.local_rank == -1 else DistributedSampler(eval_datasets)
    eval_dataloader = kdDataLoader(eval_datasets, sampler=eval_sampler, batch_size=args.predict_batch_size)
    model.eval()
  
    eval_loss = 0.0
    nb_eval_steps = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=None):
        input_ids, token_type_ids, attention_mask, labels = batch
        
        input_ids = input_ids.to(args.device)
        token_type_ids = token_type_ids.to(args.device)
        attention_mask = attention_mask.to(args.device)
        labels = labels.to(args.device)
        with torch.no_grad():
            logits, pooled_output, hidden_states, attentions, loss = model(
                input_ids, token_type_ids, attention_mask,labels)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel evaluating

            eval_loss += loss
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    results = {'loss':eval_loss}
    logger.info("***** Eval results %s *****")
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))
    return results