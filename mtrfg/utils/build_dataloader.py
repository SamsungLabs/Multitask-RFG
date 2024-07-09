"""
    Get dataloader
"""

import os
from torch.utils.data import DataLoader
from mtrfg.utils import NERdataset, get_label_index_mapping, merge_conllu_files
from transformers import AutoTokenizer
from torch.utils.data._utils.collate import default_collate
import warnings
from typing import List

def build_dataloader(config, loader_type = 'train', collate_function = default_collate, label_index_map = None):
    """
        Build dataloader based on config file
    """

    if label_index_map:
        label_index_map = label_index_map
    else:
        warnings.warn(f'Label to index map is not provided to build the dataloader, using train file in current config to build the dataloader. If you are testing or evaluating, provide a json file with labels using --labels_json_path flag.')
        label_index_map = get_label_index_mapping(config['train_file'])
    
    kwargs = {'add_prefix_space' : True}
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], **kwargs)
    
    if loader_type == 'train':
        if isinstance(config['train_file'], List):
            print("Received list of files for train, merging them.")
            config['train_file'] = merge_conllu_files(config['train_file'], '/tmp/train.conllu')
        train_data = NERdataset(config['train_file'], tokenizer = tokenizer, label_index_map=label_index_map,
            config = config)
        train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle = config['shuffle'], num_workers = 0, collate_fn = collate_function)
        return train_loader

    if loader_type == 'val':
        if isinstance(config['val_file'], List):
            print("Received list of files for validation, merging them.")
            config['val_file'] = merge_conllu_files(config['val_file'], '/tmp/dev.conllu')
        val_data = NERdataset(config['val_file'], tokenizer = tokenizer, label_index_map=label_index_map, 
            config = config)
        val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle = config['shuffle'], num_workers = 0, collate_fn = collate_function)
        return val_loader

    if loader_type == 'test':
        if isinstance(config['test_file'], List):
            print("Received list of files for test, merging them.")
            config['test_file'] = merge_conllu_files(config['test_file'], '/tmp/test.conllu')
        test_data = NERdataset(config['test_file'], tokenizer = tokenizer, label_index_map=label_index_map,
            config = config)
        test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle = config['shuffle'], num_workers = 0, collate_fn = collate_function)
        return test_loader

    return None