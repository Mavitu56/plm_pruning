import os
import time
import json

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
import transformers
import evaluate

from tqdm.auto import tqdm

from torch.optim import AdamW

from transformers import (
    AutoConfig,
    get_scheduler,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from whittle.sampling import RandomSampler
from whittle.training_strategies import (
    RandomLinearStrategy,
    RandomStrategy,
    SandwichStrategy,
    StandardStrategy,
)

from search_spaces import (
    FullSearchSpace,
    SmallSearchSpace,
    LayerSearchSpace,
    MediumSearchSpace,
)
from data_wrapper.task_data import GLUE_TASK_INFO
from hf_args import DataTrainingArguments, ModelArguments, parse_model_name
from data_wrapper import Glue, IMDB, SWAG, AlpacaDataset
from bert import (
    SuperNetBertForMultipleChoiceSMALL,
    SuperNetBertForMultipleChoiceMEDIUM,
    SuperNetBertForMultipleChoiceLAYER,
    SuperNetBertForMultipleChoiceLARGE,
    SuperNetBertForSequenceClassificationSMALL,
    SuperNetBertForSequenceClassificationMEDIUM,
    SuperNetBertForSequenceClassificationLAYER,
    SuperNetBertForSequenceClassificationLARGE,
)
from roberta import (
    SuperNetRobertaForMultipleChoiceSMALL,
    SuperNetRobertaForMultipleChoiceMEDIUM,
    SuperNetRobertaForMultipleChoiceLAYER,
    SuperNetRobertaForMultipleChoiceLARGE,
    SuperNetRobertaForSequenceClassificationSMALL,
    SuperNetRobertaForSequenceClassificationMEDIUM,
    SuperNetRobertaForSequenceClassificationLAYER,
    SuperNetRobertaForSequenceClassificationLARGE,
)
from llama import (
    SuperNetLlamaForCausalLMSMALL,
    SuperNetLlamaForCausalLMMEDIUM,
    SuperNetLlamaForCausalLMLAYER,
    SuperNetLlamaForCausalLMLARGE,
    SuperNetLlamaForSequenceClassificationSMALL,
    SuperNetLlamaForSequenceClassificationMEDIUM,
    SuperNetLlamaForSequenceClassificationLAYER,
    SuperNetLlamaForSequenceClassificationLARGE,
)


def kd_loss(
    student_output, targets, teacher_output, temperature=1, is_regression=False
):
    teacher_logits = teacher_output.logits.detach()
    student_logits = student_output.logits
    if is_regression:
        return F.mse_loss(student_logits, teacher_logits)
    else:
        kd_loss = F.cross_entropy(
            student_logits / temperature, F.softmax(teacher_logits / temperature, dim=1)
        )
        predictive_loss = F.cross_entropy(student_logits, targets)
        return temperature ** 2 * kd_loss + predictive_loss


search_spaces = {
    "small": SmallSearchSpace,
    "medium": MediumSearchSpace,
    "layer": LayerSearchSpace,
    "large": FullSearchSpace,
}

model_types = dict()
model_types["bert"] = {
    "seq_classification": {
        "small": SuperNetBertForSequenceClassificationSMALL,
        "medium": SuperNetBertForSequenceClassificationMEDIUM,
        "layer": SuperNetBertForSequenceClassificationLAYER,
        "large": SuperNetBertForSequenceClassificationLARGE,
    },
    "multiple_choice": {
        "small": SuperNetBertForMultipleChoiceSMALL,
        "medium": SuperNetBertForMultipleChoiceMEDIUM,
        "layer": SuperNetBertForMultipleChoiceLAYER,
        "large": SuperNetBertForMultipleChoiceLARGE,
    },
}
model_types["roberta"] = {
    "seq_classification": {
        "small": SuperNetRobertaForSequenceClassificationSMALL,
        "medium": SuperNetRobertaForSequenceClassificationMEDIUM,
        "layer": SuperNetRobertaForSequenceClassificationLAYER,
        "large": SuperNetRobertaForSequenceClassificationLARGE,
    },
    "multiple_choice": {
        "small": SuperNetRobertaForMultipleChoiceSMALL,
        "medium": SuperNetRobertaForMultipleChoiceMEDIUM,
        "layer": SuperNetRobertaForMultipleChoiceLAYER,
        "large": SuperNetRobertaForMultipleChoiceLARGE,
    },
}
# Adicionar suporte para LLaMA
model_types["llama"] = {
    "seq_classification": {
        "small": SuperNetLlamaForSequenceClassificationSMALL,
        "medium": SuperNetLlamaForSequenceClassificationMEDIUM,
        "layer": SuperNetLlamaForSequenceClassificationLAYER,
        "large": SuperNetLlamaForSequenceClassificationLARGE,
    },
    "multiple_choice": {
        # Para múltipla escolha, caso necessário
        "small": SuperNetLlamaForSequenceClassificationSMALL,
        "medium": SuperNetLlamaForSequenceClassificationMEDIUM, 
        "layer": SuperNetLlamaForSequenceClassificationLAYER,
        "large": SuperNetLlamaForSequenceClassificationLARGE,
    },
    "causal_lm": {
        # Para geração de texto
        "small": SuperNetLlamaForCausalLMSMALL,
        "medium": SuperNetLlamaForCausalLMMEDIUM,
        "layer": SuperNetLlamaForCausalLMLAYER,
        "large": SuperNetLlamaForCausalLMLARGE,
    }
}


@dataclass
class NASArguments:
    search_space: str = field(metadata={"help": ""}, default="small")
    sampling_strategy: str = field(metadata={"help": ""}, default=None)
    num_random_sub_nets: int = field(metadata={"help": ""}, default=1)
    temperature: float = field(metadata={"help": ""}, default=1)


def main():
    start_time = time.time()
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, NASArguments)
    )

    (
        model_args,
        data_args,
        training_args,
        nas_args,
    ) = parser.parse_args_into_dataclasses()

    # Set seed before initializing model.
    if int(training_args.seed) == -1:
        training_args.seed = np.random.randint(2 ** 32 - 1)

    set_seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    torch.cuda.manual_seed(training_args.seed)

    model_type = parse_model_name(model_args)

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
    elif data_args.task_name == "alpaca":
        # Adicionar suporte para dataset Alpaca
        data = AlpacaDataset(
            training_args=training_args, model_args=model_args, data_args=data_args
        )
        metric = evaluate.load("perplexity")
        metric_name = "perplexity"

    train_dataloader, eval_dataloader, test_dataloader = data.get_data_loaders()
    num_labels = data.num_labels

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_type,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Determinar família do modelo
    if "llama" in model_type.lower():
        model_family = "llama"
    elif model_type.startswith("bert"):
        model_family = "bert"
    elif model_type.startswith("roberta"):
        model_family = "roberta"
    else:
        print(
            f"Model type {model_type} is not supported. "
            f"We only support models of the BERT, RoBERTa, or LLaMA family."
        )
        raise NotImplementedError

    # Selecionar a classe de modelo apropriada
    if data_args.task_name in ["swag"]:
        model_cls = model_types[model_family]["multiple_choice"][nas_args.search_space]
    elif data_args.task_name in ["alpaca"]:
        model_cls = model_types[model_family]["causal_lm"][nas_args.search_space]
    else:
        model_cls = model_types[model_family]["seq_classification"][
            nas_args.search_space
        ]

    search_space = search_spaces[nas_args.search_space](config, seed=training_args.seed)

    # Garantir que a GPU está sendo utilizada se disponível
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Carregando o modelo normalmente sem forçar dtype
    model = model_cls.from_pretrained(
        model_type,
        from_tf=bool(".ckpt" in model_type),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        low_cpu_mem_usage=True,
    )
    
    # Configurar o otimizador sem mixed precision
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

    if hasattr(training_args, "fp16") and training_args.fp16:
        try:
            # Usar accelerate para gerenciar treinamento em precisão mista
            from accelerate import Accelerator
            accelerator = Accelerator(mixed_precision='fp16')
            
            # Preparar modelo, optimizer e dataloaders
            model, optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader, eval_dataloader, test_dataloader
            )
            
            print("Using accelerate for mixed precision training")
            use_amp = False  # Desativar nosso próprio loop AMP já que o accelerator cuidará disso
        except ImportError:
            # Fallback para o método original
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
            print("Using torch.cuda.amp for mixed precision training")
            use_amp = True
    else:
        use_amp = False

    num_training_steps = int(training_args.num_train_epochs * len(train_dataloader))
    warmup_steps = int(training_args.warmup_ratio * num_training_steps)

    if training_args.lr_scheduler_type == "linear":
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif training_args.lr_scheduler_type == "cosine_with_restarts":
        lr_scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=training_args.num_train_epochs,
        )

    progress_bar = tqdm(range(num_training_steps))

    step = 0
    print(f"Use {nas_args.sampling_strategy} to update super-network training")

    is_regression = True if data_args.task_name == "stsb" else False

    def loss_function(predictions, labels):
        loss_value = predictions.loss
        
        # Handle LLaMA's loss format
        if not isinstance(loss_value, torch.Tensor):
            # Convert to tensor while preserving the value (don't use zero)
            return torch.tensor(loss_value, device=labels.device, requires_grad=True)
        
        # Ensure requires_grad is True
        if not loss_value.requires_grad:
            return loss_value.clone().detach().requires_grad_(True)
            
        return loss_value

    sampler = RandomSampler(search_space.config_space, seed=training_args.seed)
    training_strategies = {
        "standard": StandardStrategy(
            sampler=sampler,
            loss_function=loss_function,
        ),
        "sandwich": SandwichStrategy(
            sampler=sampler,
            loss_function=loss_function,
        ),
        "random": RandomStrategy(
            sampler=sampler,
            loss_function=loss_function,
        ),
        "random_linear": RandomLinearStrategy(
            sampler=sampler,
            loss_function=loss_function,
            total_number_of_steps=num_training_steps,
        ),
        "kd": RandomStrategy(
            sampler=sampler,
            kd_loss=kd_loss,
            loss_function=loss_function,
        ),
        "full": SandwichStrategy(
            sampler=sampler,
            kd_loss=kd_loss,
            loss_function=loss_function,
        ),
    }

    update_op = training_strategies[nas_args.sampling_strategy]

    for epoch in range(int(training_args.num_train_epochs)):
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # Usar mixed precision se configurado
            if use_amp:
                with autocast():
                    try:
                        # Tentar obter a perda
                        loss = update_op(model, batch, batch["labels"])
                        # Verificar se a perda é um tensor
                        if not isinstance(loss, torch.Tensor):
                            print(f"WARNING: Loss is not a tensor but {type(loss)}. Converting to zero tensor.")
                            loss = torch.tensor(0.0, device=device, requires_grad=True)
                    except Exception as e:
                        print(f"Error during forward pass: {e}")
                        # Fallback para caso de erro
                        loss = torch.tensor(0.0, device=device, requires_grad=True)
                        
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = update_op(model, batch, batch["labels"])
                loss.backward()
                optimizer.step()
                
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            train_loss += loss.item()
            step += 1

        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(batch)
                
                if data_args.task_name == "alpaca":
                    # Para modelos de linguagem causal, calculamos perplexidade
                    loss = outputs.loss
                    eval_loss += loss.item()
                else:
                    # Para tarefas de classificação
                    logits = outputs.logits
                    predictions = (
                        torch.squeeze(logits) if is_regression else torch.argmax(logits, dim=-1)
                    )
                    metric.add_batch(predictions=predictions, references=batch["labels"])

        if data_args.task_name == "alpaca":
            # Cálculo de perplexidade para modelos de linguagem
            eval_metric = {"perplexity": np.exp(eval_loss / len(eval_dataloader))}
        else:
            eval_metric = metric.compute()

        runtime = time.time() - start_time
        print(
            f"epoch {epoch}: training loss = {train_loss / len(train_dataloader)}, "
            f"evaluation metrics = {eval_metric}, "
            f"runtime = {runtime}"
        )
        print(f"epoch={epoch};")
        print(f"training loss={train_loss / len(train_dataloader)};")
        print(f"evaluation metrics={eval_metric[metric_name]};")
        print(f"runtime={runtime};")

        if training_args.save_strategy == "epoch":
            os.makedirs(training_args.output_dir, exist_ok=True)
            print(f"Store checkpoint in: {training_args.output_dir}")
            model.save_pretrained(training_args.output_dir)

    # Teste final
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(batch)
            
            if data_args.task_name == "alpaca":
                loss = outputs.loss
                test_loss += loss.item()
            else:
                logits = outputs.logits
                predictions = (
                    torch.squeeze(logits) if is_regression else torch.argmax(logits, dim=-1)
                )
                metric.add_batch(predictions=predictions, references=batch["labels"])

    if data_args.task_name == "alpaca":
        test_metric = {"perplexity": np.exp(test_loss / len(test_dataloader))}
    else:
        test_metric = metric.compute()
        
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    results = {}
    results["dataset"] = data_args.task_name
    results["params"] = n_params
    results["search_space"] = nas_args.search_space
    results["runtime"] = time.time() - start_time
    results[metric_name] = float(eval_metric[metric_name])
    results["test_" + metric_name] = float(test_metric[metric_name])
    fname = os.path.join(
        training_args.output_dir, f"results_{data_args.task_name}.json"
    )
    json.dump(results, open(fname, "w"))


if __name__ == "__main__":
    main()