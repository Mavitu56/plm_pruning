import json
import logging
import os

from argparse import ArgumentParser
from pathlib import Path
from transformers import AutoConfig, AutoModelForSequenceClassification

from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from syne_tune.config_space import randint, choice
from syne_tune.experiments import load_experiment

from baselines import MethodArguments, methods

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset_seed", type=int)
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--runtime", type=int)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--iterations", type=int, default=-1)
    parser.add_argument("--method", type=str, default="random_search")
    parser.add_argument("--torch_dtype", type=str, default=None, choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code when loading models")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit quantization")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit quantization")

    args, _ = parser.parse_known_args()

    # Check if model is a Llama model
    is_llama = "llama" in args.model_name.lower()
    
    # Basic config space for all models
    config_space = {
        "num_layers": randint(0, 12),
        "num_heads": randint(0, 12),
        "num_units": randint(0, 3072),
        "model_name_or_path": args.model_name,
        "output_dir": "./nas_output",
        "task_name": args.dataset,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": 2e-05,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 8,
        "seed": args.seed,
        "dataset_seed": args.dataset_seed,
    }

    # Add Llama specific configurations if needed
    if is_llama:
        if args.torch_dtype:
            config_space["torch_dtype"] = args.torch_dtype
        if args.trust_remote_code:
            config_space["trust_remote_code"] = True
        
        # For Llama, we might want different padding options
        config_space["padding_side"] = "left"
        
        # If using FlashAttention-2, add it as an option
        if torch.cuda.is_available():
            config_space["attn_implementation"] = choice(["eager", "sdpa", "flash_attention_2"])
        
        if args.load_in_8bit:
            config_space["load_in_8bit"] = True
        if args.load_in_4bit:
            config_space["load_in_4bit"] = True

    if args.dataset == "stsb":
        config_space["is_regression"] = True

    base_scheduler = methods[args.method](
        MethodArguments(
            config_space=config_space,
            metrics=["valid", "params"],
            mode=["min", "min"],
            random_seed=args.seed,
        )
    )
    if args.iterations > -1:
        stop_criterion = StoppingCriterion(max_num_trials_finished=args.iterations)
    else:
        stop_criterion = StoppingCriterion(max_wallclock_time=args.runtime)

    tuner = Tuner(
        trial_backend=LocalBackend(
            entry_point=str(Path(__file__).parent / "run_from_scratch_nas.py")
        ),
        scheduler=base_scheduler,
        stop_criterion=stop_criterion,
        n_workers=1,
    )
    tuner.run()

    df = load_experiment(tuner.name).results

    runtime_traj = []
    params = []
    test_error = []
    valid_error = []
    configs = []

    # Load model to calculate total parameters - with appropriate configs for Llama
    try:
        if is_llama:
            # For Llama models, we need additional arguments
            config = AutoConfig.from_pretrained(
                args.model_name,
                trust_remote_code=args.trust_remote_code
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name,
                config=config,
                trust_remote_code=args.trust_remote_code,
                torch_dtype="auto" if args.torch_dtype == "auto" else None
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    except Exception as e:
        logging.warning(f"Could not load model to calculate total parameters: {e}")
        # Fallback to a reasonable estimate for parameter count
        if is_llama:
            # Rough estimate for Llama-2-7b
            total_params = 7000000000 if "7b" in args.model_name.lower() else 13000000000
        else:
            # BERT-base size estimate
            total_params = 110000000

    for trial, trial_df in df.groupby("trial_id"):
        idx = trial_df.valid.argmin()
        params.append(float(trial_df.params.iloc[0]) * total_params)
        valid_error.append(float(trial_df.valid.iloc[idx]))
        runtime_traj.append(float(trial_df.st_tuner_time.iloc[-1]))
        test_error.append(float(trial_df.test.iloc[idx]))

        config = []
        for hyper in config_space.keys():
            if hyper in trial_df.columns:
                c = trial_df[hyper].iloc[idx]
                config.append(c)
        configs.append(config)

    results = {
        "runtime_traj": runtime_traj,
        "params": params,
        "valid_error": valid_error,
        "test_error": test_error,
        "configs": configs,  # Save configs too for better analysis
    }
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(results, open(args.output_dir + "/results.json", "w"))