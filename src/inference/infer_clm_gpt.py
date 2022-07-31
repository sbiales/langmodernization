import torch

from tqdm import tqdm
from transformers import AutoModelForCausalLM as CausalLM, AutoTokenizer as Tokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == "__main__":
    OUT_MODEL = "/home/mayank/models/langmod/run_0/checkpoint-5000/"
    print(f"Loading model {OUT_MODEL}")

    tokenizer = Tokenizer.from_pretrained(OUT_MODEL)
    model = CausalLM.from_pretrained(OUT_MODEL, is_decoder=True).to(device)

    data = open("test.txt").readlines()

    for row in tqdm(data):
        context = f"{row} [SEP]"

        input_ids = tokenizer.encode(context, return_tensors='pt', add_special_tokens=True).to(device)
        # greedy
        output = model.generate(input_ids, max_length=int(len(input_ids[0]) * 2.3), return_dict_in_generate=True, early_stopping=True)
        print('*'*10)
        print(output)
        print('*' * 10)
        # beam_search
        # output = model.generate(input_ids, max_length=int(len(input_ids[0]) * 2.3), num_beams=5, early_stopping=True)
        print(context)
        print("----------")
        res = tokenizer.decode(output[0], skip_special_tokens=False)
        # print("Output: {}".format(post_processing(res)) + '\n' + 100 * '-')
        print(res.split("[SEP]")[1])
        print("----------")

