from dataclasses import dataclass, field
from typing import Optional


def parse_model_name(model_args):
    if model_args.model_name_or_path in ["bert-small", "bert-medium", "bert-tiny"]:
        model_type = "prajjwal1/" + model_args.model_name_or_path
    elif model_args.model_name_or_path in ["electra-base"]:
        model_type = "google/electra-base-discriminator"
    elif model_args.model_name_or_path in ["electra-small"]:
        model_type = "google/electra-small-discriminator"
    elif model_args.model_name_or_path.startswith("pythia"):
        model_type = "EleutherAI/" + model_args.model_name_or_path
    elif model_args.model_name_or_path.startswith("llama-"):
        # Handle common Llama model references
        size = model_args.model_name_or_path.split("-")[1]
        if "meta" in model_args.model_name_or_path:
            model_type = f"meta-llama/Llama-2-{size}-hf"
        else:
            # Default to Meta's Llama 2 models
            model_type = f"meta-llama/Llama-2-{size}-hf"
    elif model_args.model_name_or_path.startswith("tiiuae/falcon"):
        # Direct mapping for Falcon models
        model_type = model_args.model_name_or_path
    else:
        model_type = model_args.model_name_or_path
    return model_type


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    dataset_seed: int = field(
        default=128, metadata={"help": "Seed for the dataset sampling"}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    is_regression: bool = field(
        default=False,
        metadata={"help": "Specifies if dataset is a regression dataset."},
    )
    padding_side: str = field(
        default="right",
        metadata={
            "help": "Padding side for tokenization (right for BERT/Electra, left for causal LMs like Llama)"
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether to trust remote code when loading models from HuggingFace Hub"
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override the default `torch.dtype` and load the model under a specific dtype. "
            "Available options: 'auto', 'bfloat16', 'float16', 'float32'",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Implementation of attention to use for Llama models "
            "Available options: 'eager', 'sdpa', 'flash_attention_2'",
            "choices": ["eager", "sdpa", "flash_attention_2"],
        },
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to load the model in 8-bit quantization."},
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Whether to load the model in 4-bit quantization."},
    )
    quantization_config: Optional[dict] = field(
        default=None,
        metadata={"help": "Additional quantization configuration parameters."},
    )