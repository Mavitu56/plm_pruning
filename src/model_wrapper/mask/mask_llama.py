from .utils import get_backbone, register_drop_attention_layer
import torch


def get_ffn_components(model, index):
    """
    Obtém todos os componentes da camada feed-forward do LLaMA:
    gate_proj, up_proj, down_proj
    """
    layer = get_layers(model)[index]
    # LLaMA usa SwiGLU com três componentes principais
    return {
        'gate_proj': layer.mlp.gate_proj,
        'up_proj': layer.mlp.up_proj, 
        'down_proj': layer.mlp.down_proj
    }


def get_ffn2(model, index):
    """
    Mantida para compatibilidade, retorna down_proj
    """
    layer = get_layers(model)[index]
    # A segunda camada feed-forward no LLaMA é tipicamente a down-projection
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
    # No LLaMA, o módulo de atenção é chamado self_attn
    output = layer.self_attn
    return output


def get_layers(model):
    # Camadas transformer do LLaMA estão tipicamente em model.model.layers
    # ou diretamente em model.layers dependendo da implementação
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        # Fallback para acesso direto
        layers = model.layers
    return layers


def register_mask(module, mask):
    hook = lambda _, inputs: (inputs[0] * mask)
    handle = module.register_forward_pre_hook(hook)
    return handle


def register_swiglu_mask(gate_proj, up_proj, down_proj, neuron_mask):
    """
    Registra um hook especial que aplica a máscara de neurônios
    para a arquitetura SwiGLU do LLaMA, que tem três projeções
    lineares (gate, up, down).
    
    A máscara é aplicada na entrada do down_proj, mas também
    requer modificações consistentes nas saídas de gate_proj e up_proj.
    """
    # Criamos uma máscara complementar - 0 onde queremos pruning
    pruning_indices = ~neuron_mask.bool()
    
    def swiglu_hook(module, inputs, outputs):
        if pruning_indices.sum() > 0:  # Se houver algo para podar
            # Garantimos que as dimensões sejam compatíveis
            if outputs.size(-1) != pruning_indices.size(0):
                # Adequamos a máscara às dimensões do output
                # (importante para modelos como LLaMA3)
                indices = torch.arange(outputs.size(-1), device=outputs.device)
                indices_to_mask = indices % pruning_indices.size(0)
                pruning_mask = pruning_indices[indices_to_mask]
                # Zeramos as saídas das projeções gate/up para neurônios mascarados
                outputs.index_fill_(-1, indices[pruning_mask], 0.0)
            else:
                # Caso simples: dimensões compatíveis
                outputs.index_fill_(-1, torch.nonzero(pruning_indices).squeeze(), 0.0)
        return outputs
    
    # Registramos o hook nas duas primeiras projeções
    handles = []
    handles.append(gate_proj.register_forward_hook(swiglu_hook))
    handles.append(up_proj.register_forward_hook(swiglu_hook))
    
    # Para down_proj, usamos a máscara diretamente na entrada
    handles.append(register_mask(down_proj, neuron_mask))
    
    return handles


def register_gqa_mask(self_attn, head_mask):
    """
    Registra um hook que aplica mascaramento para atenção
    agrupada (Grouped-Query Attention) usada no LLaMA3.
    
    Lida com a diferente quantidade de cabeças para queries vs key/values.
    """
    # Extrair configurações de atenção
    device = next(self_attn.parameters()).device
    
    # Detectar se o modelo usa GQA verificando atributos específicos
    uses_gqa = hasattr(self_attn, "num_key_value_heads") and hasattr(self_attn, "num_heads")
    num_heads = self_attn.num_heads if hasattr(self_attn, "num_heads") else self_attn.num_attention_heads
    num_kv_heads = self_attn.num_key_value_heads if uses_gqa else num_heads
    
    # Verificar e ajustar a máscara para compatibilidade
    if head_mask.shape[0] != num_heads and num_heads % head_mask.shape[0] == 0:
        print(f"Ajustando máscara de atenção: {head_mask.shape[0]} → {num_heads}")
        repeats = num_heads // head_mask.shape[0]
        expanded_mask = head_mask.repeat_interleave(repeats)
        head_mask = expanded_mask
    
    # Criar máscara para key/value heads
    if uses_gqa and num_kv_heads != num_heads:
        # No GQA, múltiplas query heads compartilham uma key/value head
        heads_per_kv = num_heads // num_kv_heads
        kv_head_mask = torch.ones(num_kv_heads, device=device)
        
        # Para cada grupo de query heads, se todas estiverem podadas,
        # podamos a key/value head correspondente
        for kv_idx in range(num_kv_heads):
            q_start = kv_idx * heads_per_kv
            q_end = (kv_idx + 1) * heads_per_kv
            if head_mask[q_start:q_end].sum() == 0:  # Todas as queries podadas
                kv_head_mask[kv_idx] = 0
        
        print(f"GQA detectado: {num_heads} Q-heads, {num_kv_heads} KV-heads")
        print(f"KV head mask: {kv_head_mask.sum().item()}/{num_kv_heads} cabeças ativas")
    else:
        kv_head_mask = head_mask  # Sem GQA, usamos a mesma máscara
    
    # Registrar hooks para atenção
    handles = []
    
    # Hook para q_proj (queries)
    if hasattr(self_attn, "q_proj"):
        def q_mask_hook(module, inputs, outputs):
            # Reshape para (batch, seq, heads, dim_per_head)
            orig_shape = outputs.shape
            head_dim = orig_shape[-1] // num_heads
            outputs = outputs.view(-1, num_heads, head_dim)
            
            # Aplicar máscara nas cabeças
            for head_idx in range(num_heads):
                if head_mask[head_idx] == 0:
                    outputs[:, head_idx, :] = 0
                    
            # Restaurar formato original
            outputs = outputs.view(orig_shape)
            return outputs
            
        handles.append(self_attn.q_proj.register_forward_hook(q_mask_hook))
    
    # Hook para k_proj e v_proj (keys e values)
    for proj_name in ["k_proj", "v_proj"]:
        if hasattr(self_attn, proj_name):
            def kv_mask_hook(module, inputs, outputs):
                # Reshape para (batch, seq, kv_heads, dim_per_head)
                orig_shape = outputs.shape
                head_dim = orig_shape[-1] // num_kv_heads
                outputs = outputs.view(-1, num_kv_heads, head_dim)
                
                # Aplicar máscara nas cabeças
                for head_idx in range(num_kv_heads):
                    if kv_head_mask[head_idx] == 0:
                        outputs[:, head_idx, :] = 0
                        
                # Restaurar formato original
                outputs = outputs.view(orig_shape)
                return outputs
            
            handles.append(getattr(self_attn, proj_name).register_forward_hook(kv_mask_hook))
    
    return handles


def register_drop_mlp_layer(module):
    hook = lambda _, input, output: input[0]
    handle = module.register_forward_hook(hook)
    return handle


def mask_llama(model, neuron_mask, head_mask):
    """
    Aplica máscaras de neurônios e cabeças de atenção a um modelo LLaMA.
    Versão aprimorada para lidar com SwiGLU e GQA (LLaMA3).
    
    Args:
        model: modelo LLaMA
        neuron_mask: Máscara binária para neurônios nas camadas feed-forward [num_layers, ffn_dim]
        head_mask: Máscara binária para cabeças de atenção [num_layers, num_heads]
        
    Returns:
        Lista de handles para os hooks registrados
    """
    num_hidden_layers = neuron_mask.shape[0]
    
    assert head_mask.shape[0] == num_hidden_layers
    
    # Obter dimensões do modelo
    sample_ffn = get_ffn2(model, 0)
    actual_ffn_dim = sample_ffn.weight.shape[1]  # Entrada do FFN2 (down_proj)
    
    print(f"Neuron mask shape: {neuron_mask.shape}")
    print(f"Actual FFN dimension: {actual_ffn_dim}")
    
    # Redimensionar máscara se necessário
    if neuron_mask.shape[1] != actual_ffn_dim:
        print(f"WARNING: Resizing neuron mask from {neuron_mask.shape[1]} to {actual_ffn_dim}")
        resized_mask = torch.ones((num_hidden_layers, actual_ffn_dim), 
                                  device=neuron_mask.device, 
                                  dtype=neuron_mask.dtype)
        # Copiar valores até o tamanho mínimo
        min_dim = min(neuron_mask.shape[1], actual_ffn_dim)
        for i in range(num_hidden_layers):
            resized_mask[i, :min_dim] = neuron_mask[i, :min_dim]
        neuron_mask = resized_mask
    
    handles = []
    for layer_idx in range(num_hidden_layers):
        # Obter componentes da camada feed-forward
        ffn_components = get_ffn_components(model, layer_idx)
        
        # Aplicar máscara para SwiGLU (tratando todas as projeções)
        swiglu_handles = register_swiglu_mask(
            ffn_components['gate_proj'],
            ffn_components['up_proj'],
            ffn_components['down_proj'],
            neuron_mask[layer_idx]
        )
        handles.extend(swiglu_handles)
        
        # Aplicar máscara para atenção com suporte a GQA
        attention = get_attention_output(model, layer_idx)
        gqa_handles = register_gqa_mask(attention, head_mask[layer_idx])
        handles.extend(gqa_handles)
        
        # Tratamento de cenários de poda
        if neuron_mask[layer_idx].sum() == 0 and head_mask[layer_idx].sum() == 0:
            # Se FFN e atenção estiverem completamente podados, eliminar a camada inteira
            layer = get_layers(model)[layer_idx]
            handle = register_drop_layer(layer)
            handles.append(handle)
            
        elif neuron_mask[layer_idx].sum() == 0:
            # Se apenas FFN estiver completamente podado, pular o MLP
            mlp = get_mlp(model, layer_idx)
            handle = register_drop_mlp_layer(mlp)
            handles.append(handle)
            
        elif head_mask[layer_idx].sum() == 0:
            # Se apenas atenção estiver completamente podada, pular atenção
            handle = register_drop_attention_layer(attention)
            handles.append(handle)
    
    return handles