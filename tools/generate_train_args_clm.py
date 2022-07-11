import json


DEFAULT_TRAIN_ARGS = {
    "model_name_or_path": "distilgpt2",
    "output_dir": "/home/mayank/models/langmod/run_0/",
    "dataset_path": "/home/mayank/data/langmod/merged/",
    "num_train_epochs": 6,
    "max_train_samples": 150000,
    "max_eval_samples": 10000,
    "do_train": True,
    "do_eval": True,
    "learning_rate": 5e-5,
    "per_device_train_batch_size": 12,
    "per_device_eval_batch_size": 12,
    "warmup_steps": 100,
    "report_to": "wandb",
    "run_name": "langmod-distilgpt2",
    "save_steps": 10000,
    "eval_steps": 10000
}

if __name__ == "__main__":
    OUT_PATH = 'train_args.json'
    MAX_TRAIN_SAMPLES = 150000

    if 'train' in OUT_PATH:
        conf = DEFAULT_TRAIN_ARGS
        # conf['run_name'] = "{}".format(RUN_TYPE)
        # conf['output_dir'] = OUT_MODEL.format(conf['run_name'], TOTAL_DATAPOINTS)
        # conf['dataset_path'] = MERGED_DATA_PATH
        conf['max_train_samples'] = MAX_TRAIN_SAMPLES
        conf['num_train_epochs'] = 8
        conf['save_steps'] = 5000
        conf['eval_steps'] = 5000

        json.dump(conf, open(OUT_PATH, 'w'), indent=4)
