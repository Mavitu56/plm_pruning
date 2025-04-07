import logging
import sys
import time

from dataclasses import dataclass, field

import torch
import datasets
import evaluate
import numpy as np
from syne_tune.report import Reporter
from torch.optim import AdamW

import transformers
from transformers import (
    AutoConfig,
    get_scheduler,
    HfArgumentParser,
    TrainingArguments,
    BitsAndBytesConfig,
)

from hf_args import DataTrainingArguments, ModelArguments, parse_model_name
from data_wrapper import Glue, IMDB, SWAG
from data_wrapper.task_data import GLUE_TASK_INFO
from estimate_efficency import compute_parameters
from model_data import get_model_data
from train_supernet import model_types


logger = logging.getLogger(__name__)

report = Reporter()


@dataclass
class PruningArguments:
    prune_top_n_layers: int = field(default=2)


@dataclass
class NASArguments:
    do_nas: bool = field(default=False)

    num_layers: int = field(default=12)
    num_heads: int = field(default=12)
    num_units: int = field(default=3072)
    st_checkpoint_dir: str = field(default=".")


def main():
    start_time = time.time()
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataTrainingArguments,
            TrainingArguments,
            PruningArguments,
            NASArguments,
        )
    )

    (
        model_args,
        data_args,
        training_args,
        pruning_args,
        nas_args,
    ) = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Load data
    if data_args.task_name in GLUE_TASK_INFO:
        data = Glue(
            training_args=training_args, model_args=model_args, data_args=data_args
        )
        metric = evaluate.load("glue", data_args.task_name)
        metric_name = GLUE_TASK_INFO[data_args.task_name]["metric"]
    elif data_args.task_name == "imdb":
        data = IMDB(
            training_args=training_args, model_args=model_args, data_args=data_args
        )
        metric = evaluate.load("accuracy")
        metric_name = "accuracy"
    elif data_args.task_name == "swag":
        data = SWAG(
            training_args=training_args, model_args=model_args, data_args=data_args
        )
        metric = evaluate.load("accuracy")
        metric_name = "accuracy"

    train_dataloader, eval_dataloader, test_dataloader = data.get_data_loaders()
    num_labels = data.num_labels

    # Define model
    model_type = parse_model_name(model_args)

    # Configure torch dtype for better memory efficiency with Llama models
    torch_dtype = None
    if hasattr(model_args, "torch_dtype"):
        if model_args.torch_dtype == "auto":
            torch_dtype = "auto"
        elif model_args.torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif model_args.torch_dtype == "float16":
            torch_dtype = torch.float16
        elif model_args.torch_dtype == "float32":
            torch_dtype = torch.float32

    # Configure attention implementation for Llama
    attn_implementation = None
    if hasattr(model_args, "attn_implementation"):
        attn_implementation = model_args.attn_implementation

    config = AutoConfig.from_pretrained(
        model_type,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=True if hasattr(model_args, "trust_remote_code") and model_args.trust_remote_code else None,
    )

    # Determine the model family for proper selection
    if model_type.startswith("bert"):
        model_family = "bert"
    elif model_type.startswith("roberta"):
        model_family = "roberta"
    elif "llama" in model_type.lower():
        model_family = "llama"
    else:
        logging.error(
            f"Model type {model_type} are not supported. "
            f"We only support models of the BERT, RoBERTa, or Llama family."
        )
        raise NotImplementedError

    # Set appropriate model class based on the task and model family
    if model_family == "llama":
        if data_args.task_name in ["swag"]:
            # For multiple choice tasks
            model_cls = model_types["llama"]["seq_classification"]["small"]
            # Llama doesn't have a dedicated multiple choice class, using seq classification
        else:
            model_cls = model_types["llama"]["seq_classification"]["small"]
    else:
        if data_args.task_name in ["swag"]:
            model_cls = model_types[model_family]["multiple_choice"]["small"]
        else:
            model_cls = model_types[model_family]["seq_classification"]["small"]

    # Configure padding side for causal models like Llama
    if model_family == "llama" and hasattr(data_args, "padding_side"):
        data.tokenizer.padding_side = data_args.padding_side

    # Configure quantization for Llama models
    quantization_config = None
    if model_family == "llama":
        if hasattr(model_args, "load_in_8bit") and model_args.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        elif hasattr(model_args, "load_in_4bit") and model_args.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif hasattr(model_args, "quantization_config") and model_args.quantization_config:
            quantization_config = BitsAndBytesConfig(**model_args.quantization_config)

    # Load the model with appropriate configurations
    model = model_cls.from_pretrained(
        model_type,
        from_tf=bool(".ckpt" in model_type),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        quantization_config=quantization_config,
        trust_remote_code=True if hasattr(model_args, "trust_remote_code") and model_args.trust_remote_code else None,
    )

    model_data = get_model_data(model)

    attention_head_size = model_data["attention_head_size"]
    n_params_emb = model_data["n_params_emb"]
    n_params_classifier = model_data["n_params_classifier"]
    attention_size = model_data["attention_size"]

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model.select_sub_network(
        {
            "num_layers": nas_args.num_layers,
            "num_heads": nas_args.num_heads,
            "num_units": nas_args.num_units,
        }
    )
    model.to(device)

    is_regression = True if data_args.task_name == "stsb" else False

    # compute number of parameters
    n_params_model = compute_parameters(
        dmodel=attention_size,
        dhead=attention_head_size,
        num_heads_per_layer=np.ones(nas_args.num_layers) * nas_args.num_heads,
        num_neurons_per_layer=np.ones(nas_args.num_layers) * nas_args.num_units,
        model_type=model_family,  # Pass model_family to use correct parameter calculation
    )
    n_params = n_params_emb + n_params_model + n_params_classifier

    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

    num_training_steps = int(training_args.num_train_epochs * len(train_dataloader))
    warmup_steps = int(training_args.warmup_ratio * num_training_steps)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Training
    for epoch in range(int(training_args.num_train_epochs)):
        model.train()

        train_loss = 0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += loss
        runtime = time.time() - start_time
        print(
            f"epoch {epoch}: training loss = {train_loss / len(train_dataloader)}, "
            f"runtime = {runtime}"
        )

        model.eval()

        results = {}
        for mode, dataloader in zip(
            ["valid", "test"], [eval_dataloader, test_dataloader]
        ):
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(batch)

                logits = outputs.logits
                predictions = (
                    torch.squeeze(logits)
                    if is_regression
                    else torch.argmax(logits, dim=-1)
                )

                metric.add_batch(predictions=predictions, references=batch["labels"])

            error = 1 - metric.compute()[metric_name]
            if np.isnan(error) and is_regression:
                error = 1
            results[mode] = error
        report(**results, params=n_params / total_params, epoch=epoch)


if __name__ == "__main__":
    main()