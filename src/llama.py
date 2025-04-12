from transformers.models.llama.modeling_llama import (
    LlamaForSequenceClassification,
    LlamaModel,
)

from search_spaces import (
    SmallSearchSpace,
    LayerSearchSpace,
    FullSearchSpace,
    MediumSearchSpace,
    LlamaAdaptiveSearchSpace,
)

import torch

# Função simples para mascaramento do Llama (substitui a importação de model_wrapper.mask)
def mask_llama(model, ffn_mask, head_mask):
    """Aplicar máscaras de FFN e atenção ao modelo Llama."""
    handles = []
    
    # Funções de hook para mascaramento
    def get_attention_hook(layer_idx):
        def attention_hook(module, input, output):
            # Skip se não há máscara
            if head_mask is None:
                return output
            
            try:
                # Garantir que layer_idx está dentro dos limites
                if layer_idx >= len(head_mask):
                    return output
                    
                # Verificar se output é uma tupla
                if isinstance(output, tuple):
                    output_tensor = output[0]
                else:
                    output_tensor = output
                
                # Obter dimensões
                batch_size, seq_len, hidden_dim = output_tensor.shape
                
                # Obter número de cabeças de atenção
                if hasattr(module, "num_heads"):
                    num_heads = module.num_heads
                else:
                    num_heads = getattr(module, "num_attention_heads", 
                                      model.config.num_attention_heads)
                    
                head_dim = hidden_dim // num_heads
                
                # Verificar compatibilidade
                if num_heads > head_mask.shape[1]:
                    return output
                
                # Reshape para obter cabeças
                output_reshaped = output_tensor.view(batch_size, seq_len, num_heads, head_dim)
                
                # Aplicar máscara
                for head_idx in range(num_heads):
                    if head_idx < head_mask.shape[1] and head_mask[layer_idx][head_idx] == 0:
                        output_reshaped[:, :, head_idx, :] = 0
                
                # Reshape de volta
                if isinstance(output, tuple):
                    output = list(output)
                    output[0] = output_reshaped.view(batch_size, seq_len, hidden_dim)
                    output = tuple(output)
                else:
                    output = output_reshaped.view(batch_size, seq_len, hidden_dim)
                
                return output
            except Exception as e:
                # Retornar output original em caso de erro
                return output
        return attention_hook

    def get_ffn_hook(layer_idx):
        def ffn_hook(module, input, output):
            # Skip se não há máscara
            if ffn_mask is None:
                return output
            
            try:
                # Garantir que layer_idx está dentro dos limites
                if layer_idx >= len(ffn_mask):
                    return output
                
                # Verificar se output é uma tupla
                if isinstance(output, tuple):
                    return output
                    
                # Mover a máscara para o dispositivo correto
                mask = ffn_mask[layer_idx].to(output.device)
                
                # Para o Llama
                if output.dim() == 3:
                    mask = mask.view(1, 1, -1)
                    # Garantir que as dimensões são compatíveis
                    if mask.shape[2] <= output.shape[2]:
                        output[:, :, :mask.shape[2]] = output[:, :, :mask.shape[2]] * mask
                else:
                    mask = mask.view(1, -1)
                    if mask.shape[1] <= output.shape[1]:
                        output[:, :mask.shape[1]] = output[:, :mask.shape[1]] * mask
                            
                return output
            except Exception as e:
                # Retornar output original em caso de erro
                return output
        return ffn_hook
    
    # Registrar hooks para cada camada do modelo
    for i, layer in enumerate(model.layers):
        # Aplicar máscara de atenção
        if hasattr(layer, "self_attn"):
            handle = layer.self_attn.register_forward_hook(get_attention_hook(i))
            handles.append(handle)
        
        # Aplicar máscara FFN
        if hasattr(layer, "mlp"):
            handle = layer.mlp.register_forward_hook(get_ffn_hook(i))
            handles.append(handle)
    
    return handles


class LlamaSuperNetMixin:
    search_space = None
    handles = None

    def __init__(self, config):
        super().__init__(config)
        self.handles = []

    def select_sub_network(self, sub_network_config):
        head_mask, ffn_mask = self.search_space.config_to_mask(sub_network_config)
        head_mask = head_mask.to(device=next(self.parameters()).device, dtype=self.dtype)
        ffn_mask = ffn_mask.to(device=next(self.parameters()).device, dtype=self.dtype)
        self.handles = mask_llama(self.model, ffn_mask, head_mask)

    def reset_super_network(self):
        if hasattr(self, 'handles') and self.handles is not None:
            for handle in self.handles:
                handle.remove()
        self.handles = []


class LlamaSuperNetMixinLAYERSpace(LlamaSuperNetMixin):
    @property
    def search_space(self):
        return LayerSearchSpace(self.config)


class LlamaSuperNetMixinMEDIUMSpace(LlamaSuperNetMixin):
    @property
    def search_space(self):
        return MediumSearchSpace(self.config)


class LlamaSuperNetMixinLARGESpace(LlamaSuperNetMixin):
    @property
    def search_space(self):
        return FullSearchSpace(self.config)


class LlamaSuperNetMixinSMALLSpace(LlamaSuperNetMixin):
    @property
    def search_space(self):
        return SmallSearchSpace(self.config)


# Definição das classes faltantes
class SuperNetLlamaForSequenceClassificationSMALL(
    LlamaForSequenceClassification, LlamaSuperNetMixinSMALLSpace
):
    def forward(self, inputs=None, **kwargs):
        if isinstance(inputs, dict):
            return super().forward(**inputs)
        elif inputs is not None:
            return super().forward(**inputs)
        return super().forward(**kwargs)


class SuperNetLlamaModelSMALL(
    LlamaModel, LlamaSuperNetMixinSMALLSpace
):
    def forward(self, inputs=None, **kwargs):
        if isinstance(inputs, dict):
            return super().forward(**inputs)
        elif inputs is not None:
            return super().forward(**inputs)
        return super().forward(**kwargs)


class SuperNetLlamaForSequenceClassificationLAYER(
    LlamaForSequenceClassification, LlamaSuperNetMixinLAYERSpace
):
    def forward(self, inputs=None, **kwargs):
        if isinstance(inputs, dict):
            return super().forward(**inputs)
        elif inputs is not None:
            return super().forward(**inputs)
        return super().forward(**kwargs)


class SuperNetLlamaModelLAYER(
    LlamaModel, LlamaSuperNetMixinLAYERSpace
):
    def forward(self, inputs=None, **kwargs):
        if isinstance(inputs, dict):
            return super().forward(**inputs)
        elif inputs is not None:
            return super().forward(**inputs)
        return super().forward(**kwargs)


class SuperNetLlamaForSequenceClassificationMEDIUM(
    LlamaForSequenceClassification, LlamaSuperNetMixinMEDIUMSpace
):
    def forward(self, inputs=None, **kwargs):
        if isinstance(inputs, dict):
            return super().forward(**inputs)
        elif inputs is not None:
            return super().forward(**inputs)
        return super().forward(**kwargs)


class SuperNetLlamaModelMEDIUM(
    LlamaModel, LlamaSuperNetMixinMEDIUMSpace
):
    def forward(self, inputs=None, **kwargs):
        if isinstance(inputs, dict):
            return super().forward(**inputs)
        elif inputs is not None:
            return super().forward(**inputs)
        return super().forward(**kwargs)


class SuperNetLlamaForSequenceClassificationLARGE(
    LlamaForSequenceClassification, LlamaSuperNetMixinLARGESpace
):
    def forward(self, inputs=None, **kwargs):
        if isinstance(inputs, dict):
            return super().forward(**inputs)
        elif inputs is not None:
            return super().forward(**inputs)
        return super().forward(**kwargs)


class SuperNetLlamaModelLARGE(
    LlamaModel, LlamaSuperNetMixinLARGESpace
):
    def forward(self, inputs=None, **kwargs):
        if isinstance(inputs, dict):
            return super().forward(**inputs)
        elif inputs is not None:
            return super().forward(**inputs)
        return super().forward(**kwargs)


# Adicionar classe SuperNetLlamaForSequenceClassification para compatibilidade com importação
class SuperNetLlamaForSequenceClassification(SuperNetLlamaForSequenceClassificationSMALL):
    """Classe de compatibilidade para importação"""
    pass