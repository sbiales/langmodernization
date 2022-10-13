# Language  Modernization

This repository is for the code base for Experimentation on Language Modernization.

## Ideas

Future work on this project would be to utilize embeddings other than fasttext, because fasttext gives us more phonetic similarities than semantic. In addition, using a different or older corpus, such as A Corpus of English Dialogues 1560-1760 (CED) or a cleaner corpus of modern texts instead of the Books corpus, may produce different results.

## Dependencies Installation
```bash
python3 -m venv [env] 

source [env]/bin/activate 

git clone https://github.com/sbiales/langmodernization.git 

cd langmodernization

pip install -r requirements.txt 
```

## Data

For this task, we use 10 million sentences from the Books corpus as our modern corpus. For our old texts, we use the CLMET corpus, from 1710-1850.

## Framework

### Model
We use GPT-2 and BERT as our base models for training.

### Prerequisites

#### For any Causal/Conditional LM training [CLM]: 

1. Activate the env and set `$PYTHONPATH` to the location of the repository

Windows: `set PYTHONPATH=C:\[...]\langmod`

Linux: `export PYTHONPATH=/[...]/langmod`

2. Prepare corrupted corpus – This step takes newer text and corrupts it with older words (optionally removes stop words)

Format: `python3 src/data/corrupt_corpus.py --input_data <modern corpus text path> --model_path <fasttext model path (.bin)> --out_path <location to store the corrupted data> --threshold <similarity threshhold for corrupting words>`

Example: `python3 src/data/corrupt_corpus.py --input_data bookscorpus/books.txt --model_path models/langmod/old_model_300d.bin --out_path data/langmod/corrupted/bookscorpus/ --threshold 0.6`

3. Format the input text for training – We need to merge the new text and corrupted text (i.e. <New text> [SEP] <corrupted text>). This format is a requirement for CLM training. This script overwrites the data at the same path. 

Format: `python3 src/data/merge_datafields.py --dataset_path <path of the corrupted dataset> --tokenizer <huggingface model that you plan to use for training> `

4. Define hyperparameters for CLM training and generate training args. Check all the parameters from `DEFAULT_TRAIN_ARGS` dict defined and overwrite the variables that needs to be changed. 

`python3 tools/generate_train_args_clm.py `

### CLM training: 

Causal LM training generates one token at a time using the previous tokens as context. We want to train a model that can modify/translate the old text to new text using the data we prepared and with the json containing hyperparameters. 

`python3 src/models/run_clm.py train_args.json` 

### CLM Inference: 

Causal LM inference converts the old text into the new text. It also prints the BLEU and ROUGE score. There's also a way to store the inferences, but right now it's just overwriting it so be careful. We just pass the model path and the manually created validation xlsx file.  

**Note**: Make sure to comment/uncomment lines based on the model you want to use (i.e. BERT/DistillGPT-2)

Format: `python3 src/inference/infer_clm_gpt.py --model_path <trained_model_path> --input_data <eval_data_path> --output_data <path to store the predictions>`

### Results

We have two sets of results to present and compare. The first set is using unaligned fasttext embeddings. The second set is using the old corpus aligned to the modern corpus.

In the best case scenario, the models should have more overlap with `gold modern text` and less overlap with `old text` because it should've learned to modernize the old text given to it. If it's otherwise, we can assume that the model is failing at the modernization task. Note that the overlap with `old text` can never be `0` because there'll always be some common words between old and modern text.

|                               | GPT-2   |       |   | BERT    |       |
|-------------------------------|---------|-------|---|---------|-------|
|                               | ROUGE-L | BLEU  |   | ROUGE-L | BLEU  |
| Overlap with gold modern text |  0.542  | 39.12 |   | 0.545   | 38.03 |
| Overlap with old text         |  0.826  | 83.05 |   | 0.816   | 81.70 |

After aligning our embeddings and removing any instances in the training data where the modern text and old text were identical, we obtained the following results:

|                               | GPT-2   |       |   | BERT    |       |
|-------------------------------|---------|-------|---|---------|-------|
|                               | ROUGE-L | BLEU  |   | ROUGE-L | BLEU  |
| Overlap with gold modern text |  0.318  | 27.04 |   | 0.481   | 32.98 |
| Overlap with old text         |  0.468  | 59.92 |   | 0.726   | 73.65 |

As shown in the table above, we do see significantly lesser overlap with the `old text` but we also see lesser overlap with the `gold modern text`. This requires further investigation into this method.
