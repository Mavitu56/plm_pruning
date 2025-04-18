import logging

from torch.utils.data import DataLoader

from transformers import AutoTokenizer, DataCollatorWithPadding, default_data_collator

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hf_args import parse_model_name

logger = logging.getLogger(__name__)


class DataWrapper:
    def __init__(self, training_args, model_args, data_args):
        self.training_args = training_args
        self.model_args = model_args
        self.data_args = data_args
        self.model_type = parse_model_name(model_args)

        # Load tokenizer
        self.tokenizer = self.get_tokenizer()

        # Configuração específica para modelos LLM como GPT-2 e Llama
        if (
            self.model_type.startswith("gpt2")
            or "pythia" in self.model_type
            or self.model_type.startswith("distilgpt2")
        ):
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configuração específica para Llama
        elif "llama" in self.model_type.lower():
            # Definir tokens especiais para Llama se não estiverem definidos
            special_tokens_dict = dict()
            if self.tokenizer.pad_token is None:
                special_tokens_dict["pad_token"] = "[PAD]"
            if self.tokenizer.eos_token is None:
                special_tokens_dict["eos_token"] = "</s>"
            if self.tokenizer.bos_token is None:
                special_tokens_dict["bos_token"] = "<s>"
            if self.tokenizer.unk_token is None:
                special_tokens_dict["unk_token"] = "<unk>"
                
            # Adicionar tokens especiais se necessário
            if special_tokens_dict:
                self.tokenizer.add_special_tokens(special_tokens_dict)
            
            # Definir o lado de padding (importante para modelos causais como Llama)
            if hasattr(self.data_args, "padding_side"):
                self.tokenizer.padding_side = self.data_args.padding_side
            else:
                self.tokenizer.padding_side = "left"  # Default recomendado para modelos causais

        # Padding strategy
        if self.data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            self.padding = False

        # Determine max_seq_length
        if self.data_args.max_seq_length > self.tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({self.data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({self.tokenizer.model_max_length}). Using max_seq_length={self.tokenizer.model_max_length}."
            )
        self.max_seq_length = min(
            self.data_args.max_seq_length, self.tokenizer.model_max_length
        )

        self.train_data, self.valid_data, self.test_data = self._load_data()

        data_collator = self.get_data_collator()

        self.train_dataloader = DataLoader(
            self.train_data,
            batch_size=self.training_args.per_device_eval_batch_size,
            collate_fn=data_collator,
        )

        self.eval_dataloader = DataLoader(
            self.valid_data,
            batch_size=self.training_args.per_device_eval_batch_size,
            collate_fn=data_collator,
        )

        self.test_dataloader = DataLoader(
            self.test_data,
            batch_size=self.training_args.per_device_eval_batch_size,
            collate_fn=data_collator,
        )

        self.num_labels = self.get_num_labels(self.data_args)

    def get_num_labels(self, data_args):
        if data_args.is_regression:
            num_labels = 1
        else:
            label_list = self.train_data.features["label"].names
            num_labels = len(label_list)
        return num_labels

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.model_type,
            cache_dir=self.model_args.cache_dir,
            use_fast=self.model_args.use_fast_tokenizer,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )

    def get_data_loaders(self):
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

    def get_data_collator(self):
        if self.data_args.pad_to_max_length:
            data_collator = default_data_collator
        elif self.training_args.fp16:
            data_collator = DataCollatorWithPadding(
                self.tokenizer, pad_to_multiple_of=8
            )
        else:
            data_collator = None
        return data_collator

    def _load_data(self):
        pass
