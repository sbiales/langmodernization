# Language  Modernization

This repository is for the code base for Experimentation on Language Modernization.

## Ideas

## Dependencies Installation
```bash
python3 -m venv [env] 

source [env]/bin/activate 

git clone https://github.com/sbiales/langmodernization.git 

cd langmodernization

pip install -r requirements.txt 
```

## Data

## Framework

### Model
We use GPT-2 and BERT as our base model for training.

### Prerequisites

####For any Causal/Conditional LM training [CLM]: 

1. Activate the env and set `$PYTHONPATH` 

2. Prepare corrupted corpus – This step takes newer text and corrupts it with older words, changes the word order in the sentence (optinally removes stop words)

`python3 src/data/corrupt_corpus.py --input_data <new data text path> --model_path <fasttext model path (.bin)> --out_path <Output data path where you want to store the corrupted data> --threshold 0.7 `

3. Format the input text for training – We need to merge the new text and corrupted text (i.e. <New text> [SEP] <corrupted text>). This format is a requirement for CLM training. This script rewrites the data at the same path. 

`python3 src/data/merge_datafields.py --dataset_path <path of the corrupted dataset> --tokenizer <huggingface model that you plan to use for training> `

4. Define hyperparameters for CLM training and generate training args. Check all the parameters from `DEFAULT_TRAIN_ARGS` dict defined and overwrite the variables that needs to be changed. 

`python3 tools/generate_train_args_clm.py `

###CLM training: 

Causal LM training generates one token at a time using the previous tokens as context. We want to train a model that can modify/translate the old text to new text using the data we prepared and with the json containing hyperparameters. 

`python3 src/models/run_clm.py train_args.json` 

###CLM Inference: 

Causal LM inference converts the old text into the new text. It also prints the BLEU and ROUGE score. There's also a way to store the inferences, but right now it's just overwriting it so "be careful". We just pass the model path and manually created validation xlsx file.  

**Note**: Make sure to comment/uncomment lines based on the model you want to use (i.e. BERT/DistillGPT-2)

python3 src/inference/infer_clm_gpt.py --model_path <trained_model_path> --input_data <eval_data_path> --output_data <path to store the predictions>

### Results


