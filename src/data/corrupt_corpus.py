import argparse
import json
import psutil
import random
from datasets import load_dataset

from src.constants import THRESHOLD, SHUFFLE_RATIO


def remove_stopchars(sentence):
    sentence = sentence.replace("``", "")
    sentence = sentence.replace("</s>", ".")
    sentence = sentence.replace('"', '')
    sentence = sentence.replace("''", "")
    return sentence


def load_syn_dict(f_path):
    f = open(f_path, 'r', encoding='UTF-8')
    syn = json.load(f)
    f.close()
    return syn


def sentence_noising_aligned(sentence, syn_dict, thresh, sr):
    sentence = remove_stopchars(sentence)

    # 1. Synonym replacement
    words = sentence.split()
    corrupted = []
    for w in words:
        if w.strip() in ['.', '!', '?']:
            corrupted.append(w.strip())
            continue
        score = syn_dict.get(w.strip(), {"dist": 0})["dist"]
        if score >= thresh:
            corrupted.append(syn_dict.get(w.strip())["word"])
        else:
            corrupted.append(w)

    # print(f"Corr: {' '.join(corrupted)} \nOrig: {sentence}")
    if corrupted == words:
        # print("None")
        return ""

    # 2. Random shuffling
    n_sr = max(1, int(len(corrupted)*sr))
    if random.random() < sr:
       random.shuffle(corrupted)

    corrupted = ' '.join(corrupted)
    corrupted = corrupted.replace("</s>", ".")
    return corrupted


def sentence_noising(sentence, thresh, sr):
    sentence = remove_stopchars(sentence)

    # 1. Synonym replacement
    words = sentence.split()
    # words = [shakespeare_dict[w] if w in shakespeare_dict.keys() else w for w in words]
    corrupted = []
    for w in words:
        if w.strip() in ['.', '!', '?']:
            corrupted.append(w.strip())
            continue
        score, nearest = model.get_nearest_neighbors(w, k=1)[0]
        if score >= thresh:
            corrupted.append(nearest)
        else:
            corrupted.append(w)

    if corrupted == words:
        return ""

    # 2. Random shuffling
    n_sr = max(1, int(len(corrupted)*sr))
    if random.random() < sr:
       random.shuffle(corrupted)

    corrupted = ' '.join(corrupted)
    corrupted = corrupted.replace("</s>", ".")
    return corrupted


def corrupt(obj, threshold, shuffle_ratio, syn_dict):
    obj['corrupted'] = [sentence_noising_aligned(t, syn_dict=syn_dict,
                                                 thresh=threshold, sr=shuffle_ratio) for t in obj['text']]
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
    parser.add_argument("-syn_dict", "--syn_dict", type=str, help="synonym dict path")

    args = parser.parse_args()

    dataset = load_dataset('text', data_files={'train': args.input_data})
    if args.syn_dict:
        syn_dict = load_syn_dict(args.syn_dict)
        dataset_train = dataset.map(corrupt, fn_kwargs={"threshold": args.threshold,
                                                        "shuffle_ratio": args.sr,
                                                        "syn_dict": syn_dict},
                                    batched=True, num_proc=int(psutil.cpu_count() * .8))
    elif args.model_path:
        model = fasttext.load_model(args.model_path)

        dataset_train = dataset.map(corrupt, fn_kwargs={"model": model,
                                                        "threshold": args.threshold,
                                                        "shuffle_ratio": args.sr},
                                    batched=True, num_proc=int(psutil.cpu_count()*.8))
    else:
        raise ValueError("Pass at least syn_dict or model_path")
    print(dataset_train)
    dataset_train = dataset_train.filter(lambda example: example['corrupted'] != "")
    print(dataset_train)
    dataset_train.save_to_disk(args.out_path)
