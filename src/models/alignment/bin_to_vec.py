import argparse
from tqdm import tqdm
from fasttext import load_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert fasttext model from bin to vec")
    parser.add_argument("-input_path", "--input_path", type=str, help="Input Model path")
    parser.add_argument("-out_path", "--out_path", type=str, help="Output Model path")
    args = parser.parse_args()

    # original BIN model loading
    f = load_model(args.input_path)
    lines = []

    # get all words from model
    words = f.get_words()

    with open(args.out_path, 'w') as file_out:
        # the first line must contain number of total words and vector dimension
        file_out.write(str(len(words)) + " " + str(f.get_dimension()) + "\n")

        # line by line, you append vectors to VEC file
        for w in tqdm(words):
            v = f.get_word_vector(w)
            vstr = ""
            for vi in v:
                vstr += " " + str(vi)
            try:
                file_out.write(w + vstr + '\n')
            except:
                pass
