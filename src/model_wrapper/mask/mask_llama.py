from .utils import get_backbone, register_drop_attention_layer


def get_ffn2(model, index):
    layer = get_layers(model)[index]
    # LLaMA uses a different naming scheme for MLP layers
    # The second feed-forward layer in LLaMA is typically the down-projection
    ffn2 = layer.mlp.down_proj
    return ffn2


def register_drop_layer(module):
    hook = lambda _, inputs, output: (inputs[0], output[1:])
    handle = module.register_forward_hook(hook)
    return handle


def get_mlp(model, index):
    layer = get_layers(model)[index]
    return layer.mlp


def get_attention_output(model, index):
    layer = get_layers(model)[index]
    # In LLaMA, the attention module is named self_attn
    output = layer.self_attn
    return output


def get_layers(model):
    # LLaMA transformer layers are typically in model.model.layers
    # or directly in model.layers depending on the implementation
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        # Fallback to direct access
        layers = model.layers
    return layers


def register_mask(module, mask):
    hook = lambda _, inputs: (inputs[0] * mask)
    handle = module.register_forward_pre_hook(hook)
    return handle


def register_drop_mlp_layer(module):
    hook = lambda _, input, output: input[0]
    handle = module.register_forward_hook(hook)
    return handle


def mask_llama(model, neuron_mask, head_mask):
    """
    Apply neuron and attention head masks to a LLaMA model.
    
    Args:
        model: LLaMA model
        neuron_mask: Binary mask for neurons in feed-forward layers [num_layers, ffn_dim]
        head_mask: Binary mask for attention heads [num_layers, num_heads]
        
    Returns:
        List of handles to the registered hooks
    """
    num_hidden_layers = neuron_mask.shape[0]
    
    assert head_mask.shape[0] == num_hidden_layers
    
    handles = []
    for layer_idx in range(num_hidden_layers):
        # Mask neurons in feed-forward network
        ffn2 = get_ffn2(model, layer_idx)
        handle = register_mask(ffn2, neuron_mask[layer_idx])
        handles.append(handle)
        
        # Handle different pruning scenarios
        if neuron_mask[layer_idx].sum() == 0 and head_mask[layer_idx].sum() == 0:
            # If both FFN and attention are fully pruned, drop the entire layer
            layer = get_layers(model)[layer_idx]
            handle = register_drop_layer(layer)
            handles.append(handle)
            
        elif neuron_mask[layer_idx].sum() == 0:
            # If only FFN is fully pruned, skip the MLP
            mlp = get_mlp(model, layer_idx)
            handle = register_drop_layer(mlp)
            handles.append(handle)
            
        elif head_mask[layer_idx].sum() == 0:
            # If only attention is fully pruned, skip attention
            attention = get_attention_output(model, layer_idx)
            handle = register_drop_attention_layer(attention)
            handles.append(handle)
    
    return handles