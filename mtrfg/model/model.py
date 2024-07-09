"""
    This is the model, used for tagging and parsing
"""

import torch
from mtrfg.encoder import Encoder
from mtrfg.parser import BiaffineDependencyParser
from mtrfg.tagger import Tagger
import numpy as np
import warnings

class MTRfg(torch.nn.Module):
    def __init__(self, config): 
        super(MTRfg, self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.tagger = Tagger(config)
        self.parser = BiaffineDependencyParser.get_model(config)
        self.mode = 'train'

    def freeze_tagger(self):
        """ Freeze tagger if asked for!"""
        for param in self.tagger.parameters():
            param.requires_grad = False

    def freeze_parser(self):
        """ Freeze parser if asked for!"""
        for param in self.parser.parameters():
            param.requires_grad = False

    def get_output_as_list_of_dicts(self, tagger_ouptut, parser_output, model_input):
        """
            Returns list of dictionaries, each element in the dictionary is 
            1 item in the batch, list has same length as batchsize. The dictionary
            will contain 7 fields, 'words', 'head_tags_gt', 'head_tags_pred', 'pos_tags_gt',
            'pos_tags_pred', 'head_indices_gt', 'head_indices_pred'. During evalution, all fields
            should have exactly identical length, during testing, '*_gt' keys() will have empty 
            tensors.
        """
        outputs = []
        batch_size = len(tagger_ouptut)

        for i in range(batch_size):
            elem_dict = {}
            
            ## find non-masked indices
            valid_input_indices = torch.where(model_input['encoded_input']['words_mask_custom'][i] == 1)[0].cpu().detach().numpy().tolist()

            input_length = len(valid_input_indices)

            elem_dict['words'] = np.array(model_input['words'][i])[valid_input_indices].tolist()
            
            elem_dict['head_tags_gt'] = model_input['head_tags'][i].cpu().detach().numpy()[valid_input_indices].tolist()
            elem_dict['head_tags_pred'] = parser_output['predicted_dependencies'][i]

            elem_dict['head_indices_gt'] = model_input['head_indices'][i].cpu().detach().numpy()[valid_input_indices].tolist()
            elem_dict['head_indices_pred'] = parser_output['predicted_heads'][i]
            
            elem_dict['pos_tags_gt'] = model_input['pos_tag_labels'][i].cpu().detach().numpy()[valid_input_indices].tolist()
            elem_dict['pos_tags_pred'] = tagger_ouptut[i]
            
            assert np.all([len(elem_dict[key]) == input_length for key in elem_dict]), "Predictions are not same length as input!"
            ## append
            outputs.append(elem_dict)

        return outputs

    def set_mode(self, mode = 'train'):
        """
            This function will determine if loss should be computed or evaluation metrics
        """
        assert mode in ['train', 'test', 'validation'], f"Mode {mode} is not valid. Mode should be among ['train', 'test', 'validation'] "
        self.tagger.set_mode(mode)
        self.mode = mode


    def forward(self, model_input):
        """
            A dictionary containing inputs and labels
        """        

        ## Building representations
        encoder_input = model_input['encoded_input']
        encoder_output = self.encoder(encoder_input) ## we get new attention mask because we have merged representations

        ## tagging the input
        tagger_output = self.tagger(encoder_output, mask = encoder_input['words_mask_custom'], labels = model_input['pos_tag_labels'])

        ## predicted tags
        pos_tags_pred = self.tagger.get_predicted_classes_as_one_hot(tagger_output.logits)

        ## tags
        try:
            pos_tags_gt = torch.nn.functional.one_hot(model_input['pos_tag_labels'], num_classes = self.config['n_tags'])
        except:
            warnings.warn("Ground truth tags are unavailable, using predicted tags for all purposes.")
            pos_tags_gt = pos_tags_pred

        ## during training, we use gt labels, otherwise, we use predicted labels
        if self.mode in ['train', 'validation']:  
            head_tags, head_indices = model_input['head_tags'], model_input['head_indices']
        
        elif self.mode == 'test':
            # pos_tags = model_input['pos_tag_labels']
            head_tags, head_indices = None, None
        
        pos_tags_parser = pos_tags_pred if self.config['use_pred_tags'] else pos_tags_gt
        parser_output = self.parser(encoder_output, pos_tags_parser.float(), encoder_input['words_mask_custom'], head_tags = head_tags, head_indices = head_indices)

        ## calculate loss, when training or validation
        if self.mode in ['train', 'validation']:
            loss = 25 * (parser_output['loss'] + tagger_output.loss)
            return loss
        elif self.mode == 'test':
            tagger_human_readable = self.tagger.make_output_human_readable(tagger_output, encoder_input['words_mask_custom'])
            parser_human_readable = self.parser.make_output_human_readable(parser_output)            
            output_as_list_of_dicts = self.get_output_as_list_of_dicts(tagger_human_readable, parser_human_readable, model_input)
            return output_as_list_of_dicts
            ## when not training
            ## get human readable outputs
