
# End-to-end Parsing of Procedural Text into Flow Graphs


This reository contains code to reproduce main experiments in paper titled [End-to-end Parsing of Procedural Text into Flow Graphs
](https://aclanthology.org/2024.lrec-main.517.pdf), accepted to LREC-COLING 2024. 

--------------------------------------------------------------------------------

- [End-to-end Parsing of Procedural Text into Flow Graphs](#end-to-end-parsing-of-procedural-text-into-flow-graphs)
- [Abstract](#abstract)
- [Overview](#overview)
- [Installation (Only tested with python)](#installation-only-tested-with-python)
  - [Verifying the installation](#verifying-the-installation)
- [Data format](#data-format)
- [Using this repository](#using-this-repository)
  - [Training](#training)
  - [Finetuning](#finetuning)
  - [Multiple splits training](#multiple-splits-training)
  - [Custom training](#custom-training)
  - [Evaluation](#evaluation)
  - [Inference:](#inference)
    - [Method 1: Input as list of strings.](#method-1-input-as-list-of-strings)
    - [Method 2: Input as text file, each input in a new line.](#method-2-input-as-text-file-each-input-in-a-new-line)
    - [Method 3: Infer using CoNLL-U file](#method-3-infer-using-conll-u-file)
  - [Notebooks](#notebooks)
- [Contributors](#contributors)

# Abstract
We focus on the problem of parsing procedural text into fine-grained flow graphs that encode actions and entities, as well as their interactions. Specifically, we focus on parsing cooking recipes, and address a few limitations of existing parsers. Unlike SOTA approaches to flow graph parsing that work in two separate stages — identifying actions and entities (tagging) and encoding their interactions via connecting edges (graph generation) — we propose an end-to-end multi-task framework that simultaneously performs tagging and graph generation. In addition, due to the end-to-end nature of our proposed model, we can unify the input representation, and moreover can use compact encoders, resulting in small models with significantly fewer parameters than SOTA models. Another key challenge in training flow graph parsers is the lack of sufficient annotated data, due to the costly nature of the fine-grained annotations. We address this problem by taking advantage of the abundant unlabelled recipes, and show that pre-training on automatically-generated noisy silver annotations (from unlabelled recipes) results in a large improvement in flow graph parsing.

# Overview
MTrfg (short for Multitask flow graph parsing), is an effort to build better flow graph parsing system. In this work, we build a unified, end-to-end and consice framework to perform flow graph parsing. Flow graph parsing consists of 2 systems. Tagging and graph parsing. In this work, we use a single model to do solve both these tasks. We show in the paper that our model is smaller than current benchmarks, while beating prior art by 6-8 points on F1-score.

# Installation (Only tested with python)
```
git clone https://github.com/SamsungLabs/Multitask-RFG
cd Multitask-RFG
python3 -m venv MTrfg_env
source MTrfg_env/bin/activate
python3 -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu111
python3 -m pip install -e . 
python3 -m spacy download en_core_web_md
```
## Verifying the installation

To verify if `mtrfg` has successfully been built, fire up the python interpreter, and import!

```py
import mtrfg
print(mtrfg.__version__)
```

You should see the version number displayed.
# Data format
We expect data to be in [conllu format](https://universaldependencies.org/), where each line represents a token and columns are tab-separated. Below is the snippet of how the data would look like.

The relevant columns are FORM, LABEL, HEAD, DEPREL (only FORM and LABEL as input to prediction). The column DEPRELS contains additional dependency relations if a token has more than one head.

A small recipe in CoNLL-U format:
```
ID	FORM 	(LEMMA)	POS	LABEL	(FEATS)	HEAD	DEPREL	DEPRELS	(MISC)

1	Pour	_	_	B-Ac	_	11	t	_	_
2	ingredients	_	_	B-F	_	1	t	_	_
3	over	_	_	O	_	0	root	_	_
4	ice	_	_	B-F	_	1	d	_	_
5	in	_	_	O	_	0	root	_	_
6	a	_	_	O	_	0	root	_	_
7	high	_	_	B-St	_	8	o	_	_
8	ball	_	_	B-T	_	1	d	_	_
9	glass	_	_	I-T	_	0	root	_	_
10	;	_	_	O	_	0	root	_	_
11	stir	_	_	B-Ac	_	13	d	_	_
12	.	_	_	O	_	0	root	_	_
13	Garnish	_	_	B-Ac	_	0	root	_	_
14	with	_	_	O	_	0	root	_	_
15	a	_	_	O	_	0	root	_	_
16	lemon	_	_	B-F	_	13	f-comp	_	_
17	or	_	_	O	_	0	root	_	_
18	orange	_	_	B-F	_	16	o	_	_
19	twist	_	_	I-F	_	0	root	_	_
20	.	_	_	O	_	0	root	_	_
```

In CoNLL-U data, each input is separated by an empty line. You can put data files inside `data` directory, and provide appropriate paths. Gold data used for training the model can be found [here](https://github.com/interactive-cookbook/ara/tree/main/Alignment_Model/preprocessing/data/Parser). These same 3 splits are used for gold data training. 

# Using this repository
## Training
To train the model, all you need to do is pass the train, test and validation filepaths to the script. Note that if you do not have train, test and validation splits. This will not work. The code will expect all 3 files to be present. Below is an example of how to launch the training. 

```
python3 tools/train.py --opts --train_file "/path/to/train/file" --val_file "/path/to/dev/file" --test_file "/path/to/test/file" --save_dir "/path/to/output/directory/" --seed "42" --model_name "bert-base-uncased"
```

## Finetuning
There are many other config options available. If you want to launch finetuning, you will have to pass model path and labels file, which get stored during training.

```
python3 tools/train.py --opts --model_start_path "/path/to/model.pth" --labels_json_path "/path/to/labels.json" --save_dir "/path/to/output/directory/" --model_name "model_name_used_in_training"
```

Note that if your finetuning fails, ensure that you provide correct model name that was used during pretraining.

## Multiple splits training
If you want to train model for multiple splits, you can launch the following command. The script will automatically create 30 splits and train one by one until all the models are trained. 

```
python3 tools/train_K-splits.py --opts --splits "30" --seed "42" --train_file "/path/to/train/file" --test_file "/path/to/test/file" --dev_file "/path/to/dev/file"
```

The script will merge train, test and validation data. And Create 30 splits, each having 80% of data as training, 10% for testing and 10% for validation.

## Custom training
If you want to train your own model with a different encoder or other parameters. You can change it in [config file](https://github.sec.samsung.net/d-bhatt/Multitask-RFG/blob/main/mtrfg/config/config.py), or you can override those config options via command line. You just need to pass `--<key_name> "value"` while running the script. 
For example, if you want to change `model_name`, `epochs` and `batch_size` while launching training, you can use run command as shown below,

```
python3 tools/train.py --opts --train_file "/path/to/train/file" --val_file "/path/to/dev/file" --test_file "/path/to/test/file" --save_dir "/path/to/output/directory/" --seed "42" --model_name "facebook/bart-base" --epochs "20" --batch_size "10"
```

This way, your config will be overridden through command line. Note that some params like `n_tags`, `n_edge_labels` are determined by the script based on dataset provided. Similary, `device` is determined based on GPU availability on a particular machine.



## Evaluation
Performing inference with mtrfg is simple. It also expect data to be in CoNLL-U format. Run below command to perform evaliation,

```
python3 tools/evaluate.py --opts --dir_name "/path/to/model/directory" --test_file "/path/to/test/file/in/conllu/format" --batch_size "16" --save_file_name "test_results.json" --use_pred_tags "True"
```

This will print results on the terminal, as well as save them in `test_results.json` file inside model directory.

## Inference:
You can perform inference on any new input in 3 different ways. 

### Method 1: Input as list of strings.
```
python3 tools/inference.py --opts --dir_name "/path/to/model/directory" --recipes "["text of input 1", "text of input 2"]" --output_file_path "/path/to/output/file/in/conllu/format" --use_pred_tags "True"
```

### Method 2: Input as text file, each input in a new line.
```
python3 tools/inference.py --opts --dir_name "/path/to/model/directory" --recipes_file_path "/path/to/recipe/txt/file" --output_file_path "/path/to/output/file/in/conllu/format" --use_pred_tags "True"
```

### Method 3: Infer using CoNLL-U file
```
python3 tools/inference.py --opts --dir_name "/path/to/model/directory" --conllu_test_path "/path/to/recipe/conllu/file" --output_file_path "/path/to/output/file/in/conllu/format" --use_pred_tags "True"
```

## Notebooks
There are some useful ipython notebooks inside `Multitask-RFG/notebooks`, which are useful for evaluation and analysis. The noteboooks are not upto date, so you may have to change some variables and paths that could be present.

# Contributors

* Dhaivat Bhatt 
* Ahmad Pourihosseini
  
For any queries about this work, feel free to send an email to [dhaivat1994@gmail.com](dhaivat1994@gmail.com).