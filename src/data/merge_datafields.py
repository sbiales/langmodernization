import argparse
import datasets
from transformers import AutoTokenizer


def merge_text(data, merge_column_name='text'):
    if data['text'].strip() and data['corrupted'].strip():
        data[merge_column_name] = "{} {} {}".format(data['corrupted'],
                                                    "[SEP]",
                                                    data['text'])
    else:
        print("Error Occurred!")
    return data

#
# def add_random_column_to_dataset(dataset):
#     new_column = ["random"] * len(dataset)
#     return dataset.add_column("text", new_column)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corrupt corpus for training paraphrasing model")
    parser.add_argument("-dataset_path", "--dataset_path", type=str, help="Input Data path")
    parser.add_argument("-tokenizer", "--tokenizer", type=str, default="distilgpt2",
                        help="Tokenizer path or HuggingFace tokenizer name")
    args = parser.parse_args()
    
    dataset = datasets.load_from_disk(args.dataset_path)

    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # print(dataset['train'][0])
    # print(merge_text(dataset['train'][0]))
    DROP_COLS = list(set(dataset['train'].column_names) - set(['text']))

    func = merge_text

    dataset = dataset['train'].train_test_split(test_size=0.05)
    dataset['validation'] = dataset['test']
    
    dataset['test'] = dataset['test'].map(func)
    dataset['train'] = dataset['train'].map(func)
    dataset['validation'] = dataset['validation'].map(func)

    dataset['test'] = dataset['test'].remove_columns(DROP_COLS)
    dataset['train'] = dataset['train'].remove_columns(DROP_COLS)
    dataset['validation'] = dataset['validation'].remove_columns(DROP_COLS)
    
    print(dataset)
    # 
    # print(dataset['test']['text'][:5])
    # print(dataset['train']['text'][:5])
    # print(dataset['validation']['text'][:5])
    # 
    dataset.save_to_disk(args.dataset_path)
