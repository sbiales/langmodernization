import argparse
import pandas as pd
import torch

from rouge_metric import PyRouge
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from transformers import AutoModelForCausalLM as CausalLM, AutoTokenizer as Tokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for generating modernized text")
    parser.add_argument("-model_path", "--model_path", type=str, help="Model path")
    parser.add_argument("-input_data", "--input_data", type=str, help="Input Data path")
    parser.add_argument("-output_data", "--output_data", type=str, help="Input Data path")
    args = parser.parse_args()

    print(f"Loading data from {args.input_data}")
    if 'xls' in args.input_data:
        data = pd.read_excel(args.input_data)
    else:
        data = pd.read_csv(args.input_data, sep='\t')
    data = data.dropna()
    print(data.columns)
    
    MODEL = args.model_path
    print(f"Loading model {MODEL}")

    tokenizer = Tokenizer.from_pretrained(MODEL)
    model = CausalLM.from_pretrained(MODEL, is_decoder=True).to(device)
    old_sents = data['old'].to_list()
    new_sents = data['new'].to_list()
    gen_sents = []
    for sent in tqdm(old_sents):
        context = f"{sent} [SEP]"
        input_ids = tokenizer.encode(context, return_tensors='pt', add_special_tokens=True).to(device)
        # greedy
        output = model.generate(input_ids, max_length=int(len(input_ids[0]) * 2.3), return_dict_in_generate=True,
                                early_stopping=True)
        # print('*'*10)
        # print(output)
        # print('*' * 10)
        # beam_search
        # output = model.generate(input_ids, max_length=int(len(input_ids[0]) * 2.3), num_beams=5, early_stopping=True)
        # print(context)
        # print("----------")
        res = tokenizer.batch_decode(output[0], skip_special_tokens=False)
        # print(res)
        # print("Output: {}".format(post_processing(res)) + '\n' + 100 * '-')

        # For GPT-2
        # out = res[0].split("[SEP]")[1]
        # out = out.split('"``')[0]

        #For BERT
        out = res[0].split("[SEP]")[2]

        print(out)
        gen_sents.append(out)

        # print("----------")
    data['generated'] = gen_sents
    data.to_csv(args.output_data, sep='\t', index=False)

    bleu = BLEU()
    print(f"Model: {MODEL}")
    print("Overlap with gold modern text...")
    print(bleu.corpus_score(gen_sents, [new_sents]))

    print("Overlap with old text...")
    print(bleu.corpus_score(gen_sents, [old_sents]))

    rouge = PyRouge(rouge_n=(1, 2, 4), rouge_l=True)

    r_trans = lambda x: [x]

    refs = list(map(r_trans, new_sents))

    scores = rouge.evaluate(gen_sents, refs)
    print("Overlap with gold modern text ROUGE: {}".format(scores['rouge-l']))

    refs = list(map(r_trans, old_sents))
    scores = rouge.evaluate(gen_sents, refs)
    print("Overlap with old text ROUGE: {}".format(scores['rouge-l']))
