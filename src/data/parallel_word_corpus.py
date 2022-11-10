import argparse
import json
import numpy as np
import scann
import time


def get_cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def _fasttext_model_to_dict(model_path):
    idx_to_word, embedding_matrix = {}, []

    if ".vec" not in model_path:
        raise NotImplementedError

    with open(model_path, 'r') as model:
        print(f"Reading fasttext model from {model}. \nNumber of Words, Vector Dims: {next(model)}")

        for t_idx, row in enumerate(model):
            token, vec = row.split(" ", 1)
            vec = np.fromstring(vec, np.float32, sep=' ')
            idx_to_word[t_idx] = token
            embedding_matrix.append(vec)

    return idx_to_word, np.array(embedding_matrix)


def fasttext_models_to_dicts(new_model_path, old_model_path):
    old_word_idx, old_model_embeddings = _fasttext_model_to_dict(old_model_path)
    new_word_idx, new_model_embeddings = _fasttext_model_to_dict(new_model_path)

    normalized_dataset = old_model_embeddings / np.linalg.norm(old_model_embeddings, axis=1)[:, np.newaxis]
    normalized_queries = new_model_embeddings / np.linalg.norm(new_model_embeddings, axis=1)[:, np.newaxis]

    searcher = scann.scann_ops_pybind.builder(normalized_dataset, 10, "dot_product").tree(
        num_leaves=2000, num_leaves_to_search=100, training_sample_size=25000).score_ah(
        2, anisotropic_quantization_threshold=0.2).reorder(100).build()

    start = time.time()
    neighbors, distances = searcher.search_batched(normalized_queries, leaves_to_search=200,
                                                   pre_reorder_num_neighbors=250, final_num_neighbors=1)

    print(f"Retrieved Nearest neighbours for {len(new_model_embeddings)} in {time.time() - start} secs.")

    parallel_dict = {}
    for n_idx, neighbor in enumerate(neighbors):
        if 0.55 < distances[n_idx] < 0.6:
            print(neighbor)
            print(f"Approximated distance: {distances[n_idx]}, "
                  f"Actual distance: {get_cos_sim(new_model_embeddings[n_idx], old_model_embeddings[neighbor[0]])}")
            print(f"New word: {new_word_idx[n_idx]}, Nearest old word: {old_word_idx[neighbor[0]]}")

        parallel_dict[new_word_idx[n_idx]] = {"word": old_word_idx[neighbor[0]], "dist": distances[n_idx][0].item()}

    return parallel_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create parallel token dict aligned fasttext models")
    parser.add_argument("-old_model_path", "--old_model_path", type=str, help="Aligned Old Model path")
    parser.add_argument("-new_model_path", "--new_model_path", type=str, help="Aligned New Model path")
    parser.add_argument("-out_path", "--out_path", type=str, help="output path of json containing parallel word dict")

    args = parser.parse_args()

    p_dict = fasttext_models_to_dicts(args.new_model_path, args.old_model_path)

    with open(args.out_path, "w") as out_file:
        json.dump(p_dict, out_file, indent=4)
