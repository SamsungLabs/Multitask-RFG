"""
    This file contains different encoders to be used for 
    extracting encoded representations from a transformer. 
"""


import torch 
import torch.nn as nn
import os
import sys
from transformers import AutoModel 
from allennlp.modules import Seq2SeqEncoder
import numpy as np
from transformers import BatchEncoding
from typing import List, Dict, Optional
import inspect

class Encoder(nn.Module):
    """
        This is an encoder class, used to encoder inputs into
        contextualized vector representations. This is a minimalistic 
        design, and support must be extended for all classes. 
        This is inherited from AutoModel, to load 
        '''
            model = Encoder.from_pretrained('bert-base-cased', num_labels = 10)
        '''
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.encoder = AutoModel.from_pretrained(self.config['model_name'])
        self.encoder_input_keys = [key for key in inspect.signature(self.encoder.forward).parameters.keys()]

        if self.config['use_multihead_attention']:
            self.mha = nn.MultiheadAttention(self.config['encoder_output_dim'], self.config['self_attention_heads'], batch_first = True, dropout = 0.2)

    def freeze_encoder(self):
        """
            freeze model parameters
        """
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """
            unfreeze model parameters
        """
        for param in self.encoder.parameters():
            param.requires_grad = True

    def merge_subword_representation(self, outputs, encoded_input: BatchEncoding):
        """
            Merges subword representations
            Parameters:
            ----------
            outputs: Of type torch.Tensor
            encoded_input: Of type BatchEncoding
        """

        ## keep it same dimensional as original output to keep 
        ## it batch friendly!
        
        outputs_new = outputs.last_hidden_state.clone()

        for i, batch in enumerate(outputs.last_hidden_state):
            word_idxs = encoded_input['word_ids_custom'][i]

            tot_words = torch.max(word_idxs).item() + 1
            
            for word_idx in range(tot_words):
                """
                    merging subword representations
                """
                word_representation = batch[torch.where(word_idxs == word_idx)[0]].mean(dim = 0)
                outputs_new[i][word_idx] = word_representation
        
        return outputs_new

    def forward(
        self, 
        encoded_input: Dict) -> torch.Tensor:

        """
            This is borrowed from BertForTokenClassification forward pass
            https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L1709
        """
        
        input_to_encoder = {key : encoded_input[key] if key in encoded_input else None for key in self.encoder_input_keys}
        input_to_encoder = {key : value for key, value in input_to_encoder.items() if not key.endswith('_custom')}

        outputs = self.encoder(**input_to_encoder)
        encoded_output = self.merge_subword_representation(outputs, encoded_input)
        
        # """
        #     let's pass through learnable multihead attention. 
        #     ## For us, Key, query and value are the same, so it's self attention 
        #     mechanism. 
        #     Important to know that in pytorch's multihead attention, padding values in the 
        #     mask are denoted by 1 and actual values are denoted by 0, while we have them
        #     other way around in the huggingface. So the module ignores any item with 
        #     True in the mask, and False values are considered correct. 
        #     Read documentation for more https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        # """

        if self.config['use_multihead_attention']:
            encoded_output, _ = self.mha(encoded_output, encoded_output, encoded_output, key_padding_mask = encoded_input['attention_mask'] < 0.5 )
            
        return encoded_output