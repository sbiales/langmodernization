import argparse

import constants as C


class Args:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Language Modernization")

    def get_generic_args(self):
        generic_args = self.parser.add_argument_group("generic_args")

        generic_args.add_argument("-seed", "--seed", type=int, default=42, help="Seed to be set")
        generic_args.add_argument(
            "-gpu",
            "--gpu",
            type=int,
            default=0,
            help="Number of gpus to use",
        )
        generic_args.add_argument(
            "-tpu_cores",
            "--tpu_cores",
            type=int,
            default=0,
            help="TPU Cores to use",
        )
        generic_args.add_argument(
            "-print_params",
            "--print_params",
            action="store_true",
            help="Print the parameters",
        )
        generic_args.add_argument(
            "-dataset",
            "--dataset",
            default=["bookscorpus"],
            nargs="+",
            help="Dataset for training and evaluation",
        )

        return self.parser

    def get_callback_args(self):
        callback_args = self.parser.add_argument_group("callback_args")

        callback_args.add_argument(
            "-monitor",
            "--monitor",
            default="val_loss",
            choices=["val_loss", "val_acc"],
            help="What metric to monitor",
        )
        callback_args.add_argument(
            "-patience",
            "--patience",
            type=int,
            default=5,
            help="Number of epochs till no improvement for training to stop",
        )
        callback_args.add_argument(
            "-save_top_k",
            "--save_top_k",
            type=int,
            default=1,
            help="Best k models according to monitor",
        )
        callback_args.add_argument(
            "-metric_mode",
            "--metric_mode",
            default="min",
            choices=["min", "max", "auto"],
            help="Metric mode for model saving",
        )

        return self.parser

    def get_optimizer_args(self):
        optimizer_args = self.parser.add_argument_group("optimizer_args")

        optimizer_args.add_argument(
            "-wd",
            "--weight_decay",
            type=float,
            default=0.1,
            help="Weight decay term during optimization of loss",
        )
        optimizer_args.add_argument(
            "-lr",
            "--learning_rate",
            type=float,
            default=5e-5,
            help="Learning rate for the optimizer",
        )

        optimizer_args.add_argument(
            "-eps",
            "--epsilon",
            type=float,
            default=1e-8,
            help="Max allowable epsilon",
        )
        optimizer_args.add_argument(
            "-ws",
            "--warmup_steps",
            type=int,
            default=1000,
            help="Warm up steps for lr scheduler",
        )
        optimizer_args.add_argument(
            "-sch",
            "--scheduler",
            action="store_true",
            help="Scheduler on or off?",
        )
        return self.parser

    def get_trainer_args(self):
        trainer_args = self.parser.add_argument_group("trainer_args")

        trainer_args.add_argument(
            "-gc",
            "--gradient_clip_val",
            type=float,
            default=0,
            help="Gradient clipping value",
        )
        trainer_args.add_argument(
            "-dpout",
            "--dropout_prob",
            type=float,
            default=0.1,
            help="Drop out probability",
        )
        trainer_args.add_argument(
            "-maxe",
            "--max_epochs",
            type=int,
            default=8,
            help="Number of epochs for training",
        )
        trainer_args.add_argument(
            "-mine",
            "--min_epochs",
            type=int,
            default=4,
            help="Minimum Number of epochs for training",
        )
        trainer_args.add_argument(
            "-strategy",
            "--strategy",
            default="ddp_find_unused_parameters_false",
            choices=["dp", "ddp", "horovod", "ddp_spawn", "ddp_find_unused_parameters_false", "tpu_spawn_debug"],
            help="Accelerator to use",
        )
        trainer_args.add_argument(
            "-ofit",
            "--overfit_batches",
            type=float,
            default=0.0,
            help="Overfit a percent of training batches",
        )
        trainer_args.add_argument(
            "-prec",
            "--precision",
            type=int,
            default=32,
            choices=[16, 32],
            help="Precision",
        )
        trainer_args.add_argument(
            "-check_val_ep",
            "--check_val_every_n_epoch",
            type=int,
            default=1,
            help="Check val every n epoch",
        )
        trainer_args.add_argument(
            "-acc_grad_batch",
            "--accumulate_grad_batches",
            type=int,
            default=1,
            help="Gradient accumulation batches",
        )
        trainer_args.add_argument(
            "-log_every_n_steps",
            "--log_every_n_steps",
            type=int,
            default=100,
            help="Log in every n steps",
        )
        trainer_args.add_argument(
            "-limit_train_batches",
            "--limit_train_batches",
            type=float,
            default=1.0,
            help="Fraction of training set to be used",
        )
        trainer_args.add_argument(
            "-profiler",
            "--profiler",
            action="store_true",
            help="Enable profiler",
        )
        trainer_args.add_argument(
            "-suffix",
            "--suffix",
            default=None,
            help="suffix to be added in modeldirname",
        )

        trainer_args.add_argument(
            "-debug",
            "--debug",
            action="store_true",
            help="If debug mode is on then the following params will be set. Otherwise val_check_interval=1.0\
                                                                                  limit_val_batches=1.0",
        )
        trainer_args.add_argument(
            "-val_check_interval",
            "--val_check_interval",
            type=float,
            default=1,
            help="Run validation within epoch (1/x) times [Default = 4]",
        )
        trainer_args.add_argument(
            "-limit_val_batches",
            "--limit_val_batches",
            type=int,
            default=0,
            help="Number of batches on which validation needs to run",
        )
        trainer_args.add_argument(
            "-num_sanity_val_steps",
            "--num_sanity_val_steps",
            type=int,
            default=0,
            help="Number of validation steps - sanity",
        )
        trainer_args.add_argument(
            "-tbs",
            "--train_batch_size",
            type=int,
            default=1,
            help="Batch size for training",
        )
        trainer_args.add_argument(
            "-ebs",
            "--eval_batch_size",
            type=int,
            default=8,
            help="Batch size for validation",
        )

        retraining_args = self.parser.add_argument_group("retraining_args")

        retraining_args.add_argument(
            "-ckpt",
            "--checkpoint",
            default=None,
            help="Checkpoint to load for re-training",
        )

        return self.parser

    def get_inference_args(self):
        inference_args = self.parser.add_argument_group("inference_args")

        inference_args.add_argument(
            "-ckpt",
            "--checkpoint",
            required=True,
            help="Checkpoint to load for inference",
        )
        inference_args.add_argument("-nb", "--num_beams", type=int, default=5, help="Beam Width to use")
        inference_args.add_argument(
            "-top_k",
            "--top_k",
            type=int,
            default=10,
            help="Top - k Parameter",
        )
        inference_args.add_argument(
            "-top_p",
            "--top_p",
            type=float,
            default=0.95,
            help="Top p for nucleus sampling",
        )
        inference_args.add_argument(
            "-ns",
            "--num_return_sequences",
            type=int,
            default=1,
            help="Num of sequences to output",
        )
        inference_args.add_argument("-ds", "--do_sample", action="store_true", help="Sample ?")
        inference_args.add_argument(
            "-temp",
            "--temperature",
            type=float,
            default=0.9,
            help="Temperature for softmax reparameterization",
        )
        inference_args.add_argument(
            "-lts",
            "--limit_test_set",
            type=float,
            default=1.0,
            help="% of Test set to be output. Should be between 0 and 1",
        )
        inference_args.add_argument(
            "-max_length",
            "--max_length",
            type=int,
            default=200,
            help="Max length of summary",
        )

        return self.parser

    def get_model_args(self):
        model_args = self.parser.add_argument_group("model_args")

        model_args.add_argument(
            "-model",
            "--model",
            default="bart",
            choices=["bart", "t5"],
            help="Which model to use for learning?",
        )

        return self.parser

    def get_evaluation_args(self):
        evaluation_args = self.parser.add_argument_group("evaluation_args")

        evaluation_args.add_argument(
            "-i",
            "--input_file",
            required=True,
            help="json file name for evaluation",
        )
        evaluation_args.add_argument(
            "-m",
            "--metrics",
            default=["rouge", "bleu", "meteor", "bertscore", "mauve"],
            nargs="+",
            help="Metrics to be evaluated on",
        )
        evaluation_args.add_argument(
            "-gpu",
            "--gpu",
            type=int,
            default=0, # Will use CPU if -gpu = 0
            help="Number of gpus to use",
        )

        return self.parser