"""
    A very simple training script!
"""

import torch
import os
from mtrfg.utils import (save_json, 
                        build_dataloader,  
                        setup_config, 
                        build_model, 
                        build_optimizer, 
                        make_dir,
                        get_args, 
                        set_seeds,
                        write_text,
                        ner_collate_fuction,
                        train_epoch,
                        run_evaluation,
                        load_json,
                        validate_epoch,
                        get_label_index_mapping, 
                        save_python_command,
                        save_reproduce_training_cmd
                        )

from mtrfg.model import MTRfg
from mtrfg.config import default_cfg
from mtrfg.evaluation import evaluate_model, evaluate_model_with_all_labels
from tqdm import tqdm
import numpy as np
from pprint import pprint
import sys

## get the arguments to modify the config
args = get_args()  

## modify config based on environment
config = setup_config(default_cfg, args)

## if label index map is provided, we must use that to override number of labels
label_index_map = load_json(args['labels_json_path']) if 'labels_json_path' in args else get_label_index_mapping(config['train_file'])
config['n_tags'] = len(label_index_map['tag2class']) 
config['n_edge_labels'] = len(label_index_map['edgelabel2class'])

## make directory and save config! 
make_dir(config['save_dir'])
save_json(config, os.path.join(config['save_dir'], 'config.json'))

## set seeds
set_seeds(config['seed'])

## build a dataloader
train_loader = build_dataloader(config, loader_type = 'train', collate_function=ner_collate_fuction, label_index_map=label_index_map)
save_json(train_loader.dataset.label_index_map, os.path.join(config['save_dir'], 'labels.json'))

## build a validation dataloader
val_loader = build_dataloader(config, loader_type = 'val', collate_function=ner_collate_fuction,label_index_map=label_index_map)

## build a model
model_start_path = args['model_start_path'] if 'model_start_path' in args else None ## this is an extra argument introduced that'd perform model initialization for finetuning!
"""
This is to load the model from a certain checkpoint, could be used to resume the training or finetuning a new model. 
Note that when you are performing finetuning, pass --labels_json_path argument with appropriate labels file
so that model is finetuned/continued being trained on the same labels it saw earlier! 
"""
model = build_model(config, model_start_path = model_start_path) 

## optimizer 
optimizer = build_optimizer(config, model)

## command run should be saved so we know what did we run exactly to get that training
cmd_file = os.path.join(config['save_dir'], 'train_command.txt')
save_python_command(cmd_file, sys.argv)

## build a command to reproduce exact same training! 
reproduce_training_cmd_file = os.path.join(config['save_dir'], 'full_train_reproduce_cmd.txt')
save_reproduce_training_cmd(sys.argv[0], config, args, reproduce_training_cmd_file)

## train the model
curr_best_val_value = -np.inf
latest_save = 0
with tqdm(range(config['epochs'])) as pbar:
    for epoch in pbar:

        if epoch >= config['freeze_until_epoch']:
            model.encoder.unfreeze_encoder()

        model = train_epoch(model,train_loader, optimizer, epoch)

        ## validation
        ## let's evaluate model on validation dataset
        eval_function = evaluate_model
        eval_function_name = eval_function.__name__
        val_results, _ = run_evaluation(model, val_loader, eval_function = eval_function, config = config, label_index_map = label_index_map)
        parser_prec = val_results['parser_labeled_results']['P']
        tagger_prec = val_results['tagger_results']['P']
        save_json(val_results, os.path.join(config['save_dir'], f'val_results_with_{eval_function_name}_epoch_{epoch}.json'))

        ## let's get validation loss for early stopping
        val_loss = validate_epoch(model, val_loader)

        ## if parser is frozen, then tagger result should determine early stopping
        labeled_prec = tagger_prec if config['freeze_parser'] else parser_prec

        ## early stopping if best model is already found!
        if latest_save < config['patience'] and config['early_stopping']:
            if labeled_prec > curr_best_val_value:
                torch.save(model.state_dict(), config['model_path'])    
                curr_best_val_value = round(labeled_prec, 3)
                
                latest_save = 0
            else:
                latest_save += 1

            print(f'Epoch: {epoch}, current validation loss: {val_loss} and Labeled precision: {labeled_prec}')
        else: 
            break
    

## training is done, let's run the final evaluation
model.load_state_dict(torch.load(config['model_path']))
val_results, benchmark_metrics = run_evaluation(model, val_loader, eval_function = evaluate_model, config = config, label_index_map = label_index_map)
save_json(val_results, os.path.join(config['save_dir'], 'val_results_best.json'))
save_json(benchmark_metrics, os.path.join(config['save_dir'], 'val_results_benchmark.json'))
pprint(val_results)

## let's run the trained model on test dataset
## let's run and get results on test dataset too
test_loader = build_dataloader(config, loader_type = 'test', collate_function=ner_collate_fuction, label_index_map=label_index_map)
test_results, benchmark_metrics = run_evaluation(model, test_loader, eval_function = evaluate_model, config = config, label_index_map = label_index_map)
save_json(test_results, os.path.join(config['save_dir'], 'test_results.json'))
save_json(benchmark_metrics, os.path.join(config['save_dir'], 'test_results_benchmark.json'))