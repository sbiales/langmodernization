import argparse
import random
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm import tqdm

from src.constants import THRESHOLD, SHUFFLE_RATIO


def remove_stopwords(sentence):
    # sentence = tokenizer.tokenize(sentence)
    sentence = [word for word in sentence.split(' ')
                if word.lower() not in en_stopwords]
    sentence = ' '.join(sentence)
    # sentence = sentence.replace("''", '"').replace('``', '"')
    # sentence = detokenizer.detokenize(sentence.split())
    return sentence


def sentence_noising(sentence, model, thresh, sr):
    # 1. Synonym replacement
    words = sentence.split()
    # words = [shakespeare_dict[w] if w in shakespeare_dict.keys() else w for w in words]
    corrupted = []
    for w in tqdm(words):
        score, nearest = model.get_nearest_neighbors(w, k=1)[0]
        if score >= thresh:
            corrupted.append(nearest)
        else:
            corrupted.append(w)
        # else:
        #  corrupted.append(w)

    # 2. Random shuffling
    n_sr = max(1, int(len(corrupted)*sr))
    if random.random() < sr:
       random.shuffle(corrupted)

    return ' '.join(words)


def corrupt(obj, model, threshold, shuffle_ratio):
    obj['corrupted'] = [remove_stopwords(t) for t in obj['text']]
    obj['corrupted'] = [sentence_noising(t, model, thresh=threshold, sr=shuffle_ratio) for t in obj['corrupted']]
    return obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corrupt corpus for training paraphrasing model")
    parser.add_argument("-input_data", "--input_data", type=str, help="Input Data path")
    parser.add_argument("-model_path", "--model_path", type=str, help="Model path")
    parser.add_argument("-threshold", "--threshold", type=float, default=THRESHOLD,
                        help="Similarity threshold used for word corruptions")
    parser.add_argument("-shuffle_ratio", "--sr", type=float, default=SHUFFLE_RATIO,
                        help="Similarity threshold used for word corruptions")
    parser.add_argument("-out_path", "--out_path", type=str, help="Model path")

    args = parser.parse_args()

    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()
    en_stopwords = set(stopwords.words('english'))

    dataset = load_dataset('text', data_files={'train': args.ip_file})
    dataset_train = dataset.map(corrupt, fn_kwargs={"model": args.model_path,
                                                    "threshold": args.threshold,
                                                    "shuffle_ratio": args.shuffle_ratio},
                                batched=True)
    dataset_train.save_to_disk(args.out_path)
