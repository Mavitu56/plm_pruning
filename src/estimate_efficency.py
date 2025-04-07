import torch
import numpy as np


def mac_per_head(
    seq_len,
    hidden_size,
    attention_head_size,
):
    per_head_qkv = lambda seq_len: 3 * seq_len * hidden_size * attention_head_size
    per_head_attn = lambda seq_len: 2 * seq_len * seq_len * attention_head_size
    per_head_output = lambda seq_len: seq_len * attention_head_size * hidden_size
    mac = per_head_qkv(seq_len) + per_head_attn(seq_len) + per_head_output(seq_len)
    return mac


def mac_per_neuron(seq_len, hidden_size):
    return 2 * seq_len * hidden_size


def compute_mac(
    num_heads_per_layer,
    num_neurons_per_layer,
    seq_len,
    hidden_size,
    attention_head_size,
):
    mac = 0.0
    for num_heads, num_neurons in zip(num_heads_per_layer, num_neurons_per_layer):
        attention_mac = num_heads * mac_per_head(
            seq_len, hidden_size, attention_head_size
        )
        ffn_mac = num_neurons * mac_per_neuron(seq_len, hidden_size)
        mac += attention_mac + ffn_mac
    return mac


def compute_parameters(dmodel, dhead, num_heads_per_layer, num_neurons_per_layer, model_type="bert"):
    """
    Compute parameters for a transformer model
    
    Args:
        dmodel: hidden size dimension
        dhead: attention head dimension
        num_heads_per_layer: array of number of heads per layer
        num_neurons_per_layer: array of number of neurons per layer
        model_type: type of model ('bert', 'gpt', or 'llama')
    """
    num_layers = num_heads_per_layer.shape[0]
    assert num_layers == num_neurons_per_layer.shape[0]

    num_parameters = 0
    for layer in range(num_layers):
        # Different models use different layer norm patterns
        if model_type == "llama":
            # Llama uses RMSNorm instead of LayerNorm
            n_layer_norm = dmodel  # Only one parameter per feature (no bias in RMSNorm)
            n_layer_norm_count = 2  # Pre-attention and pre-MLP
        else:
            # BERT/GPT style models use LayerNorm with bias
            n_layer_norm = 2 * dmodel  # weight and bias
            n_layer_norm_count = 2  # Post-attention and post-MLP
            
        if num_heads_per_layer[layer] > 0:
            if model_type == "llama":
                # Llama uses rotary embeddings and has different attention structure
                n_attention = (
                    (dmodel * dhead * num_heads_per_layer[layer]) * 3  # Q, K, V projections (no bias in Llama)
                )
                n_attention += dmodel * dmodel  # Output projection (no bias in Llama) 
            else:
                n_attention = (
                    (dmodel * dhead + dhead) * num_heads_per_layer[layer] * 3
                )  # attention with bias
                n_attention += dmodel * dmodel + dmodel  # output with bias
                
            n_attention += n_layer_norm * n_layer_norm_count
        else:
            n_attention = 0
            
        if num_neurons_per_layer[layer] > 0:
            if model_type == "llama":
                # Llama MLP: up_proj (no bias), gate_proj (no bias), down_proj (no bias)
                n_ffn = (
                    dmodel * num_neurons_per_layer[layer] +  # up_proj
                    dmodel * num_neurons_per_layer[layer] +  # gate_proj 
                    num_neurons_per_layer[layer] * dmodel    # down_proj
                )
            else:
                n_ffn = (
                    2 * dmodel * num_neurons_per_layer[layer] +  # intermediate and output projs
                    dmodel + num_neurons_per_layer[layer]        # biases
                )
                
            n_ffn += n_layer_norm * n_layer_norm_count
        else:
            n_ffn = 0

        num_parameters += n_attention + n_ffn
        
    return int(num_parameters)


def compute_latency(model, tokenizer, batch, device):
    # train_dataset[0][sentence1_key],
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    
    # Handle different tokenizers/inputs between BERT and Llama
    if hasattr(model.config, "model_type") and model.config.model_type == "llama":
        # Llama may require padding_side="left" for better performance
        tokenizer.padding_side = "left"
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
    else:
        encoded = tokenizer(batch, return_tensors="pt").to(device)
    
    # warm-up GPU
    for _ in range(10):
        _ = model(**encoded)
        
    # measure latency
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(**encoded)
            ender.record()
            # synchronize GPU
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions

    return mean_syn