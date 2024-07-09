"""
	All taggers for tagging encoded sequence
"""

import torch
import torch.nn as nn
from transformers.modeling_outputs import TokenClassifierOutput
from allennlp.modules import Seq2SeqEncoder

class Tagger(nn.Module):
	"""
		This is a tagger head. Input is representation coming from the encoder 
		and output is classes for each token
	"""
	def __init__(self, config): 
		"""
		Args:
			hidden_dropout_prob: Dropout probability
			hidden_size: size of encoder representations
			num_labels: Number of class labels. 
		"""
		super().__init__()
		hidden_dropout_prob = 0.2
		self.dropout = nn.Dropout(hidden_dropout_prob)
		
		## this is a simplest classifier, but this can be better
		## TODO: Make this a big FCN, or have a decoder module
		## which can perform sequetial decoding. 
		hidden_size_tagger = 128
		encoder_output_size = config['encoder_output_dim']
		self.num_tags = config['n_tags']
		self.seq_encoder =  Seq2SeqEncoder.by_name('stacked_bidirectional_lstm')(input_size=encoder_output_size, hidden_size=hidden_size_tagger,
			num_layers=1, recurrent_dropout_probability=0.2, use_highway=True)
		self.classifier = nn.Linear(2 * hidden_size_tagger, self.num_tags)
		self.tagger_loss = nn.CrossEntropyLoss(ignore_index=0)
		self.gumbel_softmax = config['gumbel_softmax']
		self.apply(self._init_weights)
		self.mode = 'train'

	def set_mode(self, mode = 'train'):
		"""
			set mode for training v/s validation v/s test
		"""
		self.mode = mode

	def _init_weights(self, module):
		if hasattr(module, 'weight'):
			if module.weight is not None:
				torch.nn.init.xavier_uniform_(module.weight)
		if hasattr(module, 'bias'):
			if module.bias is not None:
				module.bias.data.zero_()


	def forward(self,encoder_reps: torch.Tensor, mask: torch.Tensor, labels = None):
		"""
			Using encoder representations to predict tag for each 
			token.
			Args:
				encoder_reps: Encoder representations. batchsize x seq_len x hidden_size
				labels: batchsize x seq_len
		"""

		self.tag_representations = self.seq_encoder(self.dropout(encoder_reps), mask)
		logits = self.classifier(self.tag_representations)
		
		## calculate loss if training
		## this must be made better in future
		if self.mode in ['train', 'validation']:
			loss = self.tagger_loss(logits.reshape(-1, self.num_tags), labels.reshape(-1))
		elif self.mode in ['test']:
			loss = None

		## output return
		return TokenClassifierOutput(loss = loss, logits = logits)

	def softargmax(self, input, beta=100):
		"""
			Soft argmax would be differentiable 
			version of argmax to ensure that we can retain
			gradients after argmax
		"""
		
		*_, n = input.shape

		## for numerical stability
		# input = input - (torch.max(input, dim =-1))[0].unsqueeze(-1) ## log-exp trick.

		## softmax with scaled input to ensure peaky distribution
		input = nn.functional.softmax(beta * input, dim=-1)
		indices = torch.linspace(0, n-1, n).to(input.device)
		
		## alright, let's get the output class
		result = torch.sum(input * indices, dim=-1).to(input.device)
		pred_tags = torch.round(torch.clamp(result, min = 0, max = n-1)).long() ## we claim the values between number of classes, and we round it to nearest integer, and convert it into a long data structure to be used later.
		
		return pred_tags
	
	def get_predicted_classes_as_one_hot(self, tagger_output, temperature = 1e-3):
		"""
			get predicted classes as one hot vector using softmax 
			with lower temprature. This ensures that we get one
			hot classes in a differentiable way, so that we can 
			pass down the tags for other tasks/downstream applications
			and ensure sustained gradient chain. 
		"""
		tagger_output = tagger_output - (torch.max(tagger_output, dim =-1))[0].unsqueeze(-1) ## log-exp trick.
		if self.gumbel_softmax:
			tagger_output = nn.functional.gumbel_softmax(tagger_output, tau = temperature, hard = False, dim = -1)
		else:
			tagger_output = nn.functional.softmax(tagger_output / temperature , dim=-1)
		return tagger_output

	def get_predicted_classes(self, tagger_output):
		"""
			Get predicted classes from 
			tagger's output logits
		"""
		
		pred_classes = self.softargmax(tagger_output)

		# pred_classes = torch.argmax(tagger_output, dim = -1)
		return pred_classes

	def make_output_human_readable(self, tagger_output, attention_mask):
		"""
			This is to make tagger output more easily parsable.
			Parameters:
			----------
				tagger_output: Output of the tagger, of type TokenClassifierOutput
				attention_mask: attention mask for the tagger, to extract correct logits for class calculation
		"""
		batchsize = tagger_output.logits.shape[0]

		tagger_out_classes = []
		
		for i in range(batchsize):

			## logit of a single element in the batch
			logits = tagger_output.logits[i]

			## corresponding attention mask
			mask = attention_mask[i]

			## mask on logits
			logits_masked = logits[torch.where(mask == 1)[0]]
			
			## predicted classes
			class_labels = torch.argmax(logits_masked, dim = 1)
			
			## appending
			tagger_out_classes.append(class_labels.cpu().detach().numpy().tolist())


		return tagger_out_classes