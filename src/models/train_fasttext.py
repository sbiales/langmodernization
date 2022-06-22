import argparse
import fasttext
import os


def train_fasttext(datapath, dims, model_out, model_name):
    model = fasttext.train_unsupervised(datapath, dim=dims)
    model.save_model(os.path.join(model_out, model_name if ".bin" in model_name else model_name+".bin"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FastText model")
    # args.add_argument("-seed", "--seed", type=int, default=5, help="Seed to be set")
    parser.add_argument("-input_data", "--input_data", type=str, help="Input Data path")
    parser.add_argument("-dims", "--dims", type=int, default=300, help="Fasttext vector dims")
    parser.add_argument("-model_out_path", "--model_out_path", type=str, help="path of the fasttext model to be stored")
    parser.add_argument("-out_model_name", "--out_model_name", type=str, help="Name of the fasttext model to be stored")
    args = parser.parse_args()

    train_fasttext(args.input_data, args.dims, args.model_out_path, args.out_model_name)