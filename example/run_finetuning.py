"""
@author: YiYuDL

Fine-tuning the pretrained or distilled model.
Adapted in part from MolBERT (https://github.com/BenevolentAI/MolBERT).
"""
import os
import json
import numpy as np
import pandas as pd
import click
import tempfile
from delicate.data.data_preparation import data_load
from delicate.finetune.finetune_PSapp import finetunePSApp
from delicate.finetune.finetune_KDapp import finetuneKDApp

def finetune(
    dataset,
    train_path,
    valid_path,
    test_path,
    mode,
    model,
    label_column,
    pretrained_model_path,
    max_epochs,
    freeze_level,
    learning_rate,
    num_hidden_layers,
    num_workers,
    batch_size,
):
    """
    This function runs finetuning for given arguments.

    Args:
        dataset: Name of the MoleculeNet dataset, e.g. BBBP
        train_path: file to the csv file containing the training data
        valid_path: file to the csv file containing the validation data
        test_path: file to the csv file containing the test data
        mode: either regression or classification
        label_column: name of the column in the csv files containing the labels
        pretrained_model_path: path to a pretrained molbert model
        max_epochs: how many epochs to run at most
        freeze_level: determines what parts of the model will be frozen. More details are given in molbert/apps/finetune.py
        learning_rate: what learning rate to use
        num_workers: how many workers to use
        batch_size: what batch size to use for training
    """

    # default_path = os.path.join('./logs/', datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'))
    output_dir = os.path.join('./lightning_logs_molbert/', dataset)
    raw_args_str = (
        f"--max_seq_length 512 "
        f"--batch_size {batch_size} "
        f"--max_epochs {max_epochs} "
        f"--num_workers {num_workers} "
        f"--fast_dev_run 0 "
        f"--num_hidden_layers {num_hidden_layers} "
        f"--train_file {train_path} "
        f"--valid_file {valid_path} "
        f"--test_file {test_path} "
        f"--mode {mode} "
        f"--output_size {1 if mode == 'regression' else 2} "
        f"--pretrained_model_path {pretrained_model_path} "
        f"--label_column {label_column} "
        f"--freeze_level {freeze_level} "
        f"--gpus 1 "
        f"--learning_rate {learning_rate} "
        f"--learning_rate_scheduler linear_with_warmup "
        f"--default_root_dir {output_dir}"
    )

    raw_args = raw_args_str.split(" ")

    if model == 'PSMolBERT' or model == 'DeLiCaTe':
        lightning_trainer = finetunePSApp().run(raw_args)
    else:
        lightning_trainer = finetuneKDApp().run(raw_args)
    
    return lightning_trainer


def cv(dataset, model, pretrained_model_path, freeze_level, learning_rate, num_workers, batch_size,num_hidden_layers):
    """
    This function runs cross-validation for finetuning MolBERT. The splits are obtained from ChemBench.

    Args:
        dataset: Name of the MoleculeNet dataset
        summary_df: summary dataframe loaded from chembench
        pretrained_model_path: path to a pretrained MolBERT model
        freeze_level: determines which parts of the model will be frozen.
        learning_rate: what learning rate to use
        num_workers: how many processes to use for data loading
        batch_size: what batch size to use
    """
    df, indices = data_load(dataset)
    df = df.rename(columns={'smiles': 'SMILES'})
    df.columns = [col.replace(' ', '_') for col in df.columns]
    print('dataset loaded', df.shape)
    R2 = []
    for i, (train_idx, valid_idx, test_idx) in enumerate(indices):
        train_df = df.iloc[train_idx]
        valid_df = df.iloc[valid_idx]
        test_df = df.iloc[test_idx]

        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, f"{dataset}_train.csv")
            valid_path = os.path.join(tmpdir, f"{dataset}_valid.csv")
            test_path = os.path.join(tmpdir, f"{dataset}_test.csv")

            train_df.to_csv(train_path)
            valid_df.to_csv(valid_path)
            test_df.to_csv(test_path)
            
            if df['label'].isin([0,1]).all() == True:
                mode = 'classification'
                
            else:
                mode = 'regression'
                
            # summary_df = get_summary_df()
            # mode = summary_df[summary_df['task_name'] == dataset].iloc[0]['task_type'].strip()
            print('mode =', mode)

            trainer = finetune(
                dataset=dataset,
                train_path=train_path,
                valid_path=valid_path,
                test_path=test_path,
                mode=mode,
                model = model,
                label_column=df.columns[-1],
                num_hidden_layers = num_hidden_layers,
                pretrained_model_path=pretrained_model_path,
                max_epochs=20,
                freeze_level=freeze_level,
                learning_rate=learning_rate,
                num_workers=num_workers,
                batch_size=batch_size,
            )
            print(f'fold {i}: saving model to: ', trainer.ckpt_path)
            
            trainer.test()
            metrics_path = os.path.join(os.path.dirname(trainer.ckpt_path), 'metrics.json')
            f = open(metrics_path)
            me = json.load(f)
            r2 = me['R2']
            R2.append(r2)
    result_m = round(np.mean(R2),3)
    result_s = round(np.std(R2),3)
    return result_m,result_s
        

@click.command()
@click.option('--pretrained_model_path', type=str, required=True)
@click.option('--num_hidden_layers', type=int, required=True)
@click.option('--freeze_level', type=int, required=True)
@click.option('--model', type=str, required=True, help="PSMolBERT, DeLiCaTe or MolBERT")
@click.option('--learning_rate', type=float, required=True)
@click.option('--num_workers', type=int, default=0)
@click.option('--batch_size', type=int, default=16)

def main(pretrained_model_path, model, freeze_level, learning_rate, num_workers, batch_size, num_hidden_layers):
    dataset_list = ['lipop','esol','freesolv','egfr','fgfr1']
    col_list = ['dataset','r2','str']
    df1 = pd.DataFrame(columns=col_list)
    
    for dataset in dataset_list:
        print(f'Running experiment for {dataset}')
        result_m,result_s = cv(dataset,model, pretrained_model_path, freeze_level,
                               learning_rate, num_workers, batch_size,num_hidden_layers,)
        df1.loc[len(df1)] = [dataset,result_m,result_s]
    
    R2_m = df1['r2'].mean()
    df1.loc[len(df1)] = ['Avg',R2_m,np.nan]
    results = 'results' + '_reg' + '_' + pretrained_model_path
    df1.to_csv(results,index=False)
    
if __name__ == "__main__":
    main()
