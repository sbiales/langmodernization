import argparse
import json
import numpy as np
import random


def fasttext_model_to_vocab_set(model_path):
    vocab = []

    if ".vec" not in model_path:
        raise NotImplementedError

    with open(model_path, 'r') as model:
        print(f"Reading fasttext model from {model}. \nNumber of Words, Vector Dims: {next(model)}")

        for _, row in enumerate(model):
            token, _ = row.split(" ", 1)
            vocab.append(token)

    return set(vocab)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create parallel token dict aligned fasttext models")
    parser.add_argument("-old_model_path", "--old_model_path", type=str, help="Aligned Old Model path")
    parser.add_argument("-new_model_path", "--new_model_path", type=str, help="Aligned New Model path")
    parser.add_argument("-out_path", "--out_path", type=str, help="output path of json containing parallel word dict")
    parser.add_argument("-max_count", "--max_count", type=str, help="max no of parallel words", default=2000)

    args = parser.parse_args()

    new_vocab_set = fasttext_model_to_vocab_set(args.new_model_path)
    old_vocab_set = fasttext_model_to_vocab_set(args.old_model_path)

    intersection_vocab = list(new_vocab_set.intersection(old_vocab_set))

    cnt = 0
    with open(args.out_path, "w") as out_file:
        for w in intersection_vocab:
            if len(w) > 2 and random.choice([True, False]):
                out_file.write(f"{w} {w}\n")
                cnt += 1

            if cnt >= args.max_count:
                break
