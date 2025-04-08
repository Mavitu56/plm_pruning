import logging
import argparse

from itertools import product
from pathlib import Path

from slurmpilot.config import load_config
from slurmpilot.slurm_wrapper import SlurmWrapper, JobCreationInfo
from slurmpilot.util import unify
from experiment_configs import (
    experiment_config,
    search_hyperparameters,
    create_checkpoint_dir,
    create_output_dir,
)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_methods", type=str, default=None)
    parser.add_argument("--sampling_strategy", type=str, default=None)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--search_space", type=str, default=None)
    parser.add_argument("--cluster", type=str, default=None)
    parser.add_argument("--partition", type=str, default=None)
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit quantization")
    parser.add_argument("--torch_dtype", type=str, choices=["auto", "float16", "bfloat16"], default=None)
    parser.add_argument("--attn_implementation", type=str, choices=["eager", "sdpa", "flash_attention_2"], default=None)
    parser.add_argument("--padding_side", type=str, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")

    args, _ = parser.parse_known_args()

    max_runtime_minutes = int(60 * 24)
    user_path = Path(__file__).parent

    cluster = args.cluster
    partition = args.partition

    experiments = experiment_config
    hparams = search_hyperparameters
    config = load_config(user_path=user_path)
    slurm = SlurmWrapper(config=config, clusters=[cluster])

    # overwrite defaults
    for key, value in vars(args).items():
        if key in experiments and value is not None:
            experiments[key] = [value]

    keys = list(experiments.keys())
    for exp in product(*[value for value in experiments.values()]):

        jobname = unify("plm_search", method="coolname")
        tag = "run"
        for ti in exp:
            tag += f"-{ti}"

        params = {key: value for key, value in zip(keys, exp)}

        # add hyperparameters
        for name, hparam in hparams.items():
            params[name] = hparam

        params["checkpoint_dir"] = create_checkpoint_dir(params)
        params["output_dir"] = create_output_dir(params)

        # Configure parameters for Llama
        if "llama" in params.get("model_types", ""):
            # Default settings for Llama if not specified
            if args.load_in_8bit:
                params["load_in_8bit"] = True
            if args.torch_dtype:
                params["torch_dtype"] = args.torch_dtype
            if args.attn_implementation:
                params["attn_implementation"] = args.attn_implementation
            if args.padding_side:
                params["padding_side"] = args.padding_side
            if args.trust_remote_code:
                params["trust_remote_code"] = True

        # Adjust memory requirements for Llama
        if "llama" in params.get("model_types", ""):
            mem_requirement = 1024 * 32  # 32GB for Llama
        else:
            mem_requirement = 1024 * 8   # 8GB for BERT/RoBERTa

        jobinfo = JobCreationInfo(
            cluster=cluster,
            partition=partition,
            jobname=jobname,
            entrypoint="run_sub_network_search.sh",
            src_dir=str(Path(__file__).parent.parent / "src"),
            n_cpus=1,
            n_gpus=1,
            mem=mem_requirement,
            max_runtime_minutes=max_runtime_minutes,
            # Shows how to pass an environment variable to the running script
            env={key.upper(): value for key, value in params.items()},
        )
        slurm.schedule_job(job_info=jobinfo)
        print(jobname)
