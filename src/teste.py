# Criar um script temporário para verificar search_spaces.py
import importlib.util
import sys
import torch
from transformers import AutoConfig

# Importar search_spaces e verificar sua implementação
try:
    from search_spaces import SmallSearchSpace
    
    # Carregar config do Llama
    config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    # Instanciar search space
    search_space = SmallSearchSpace(config)
    
    # Testar geração de máscara
    test_config = {
        "num_layers": config.num_hidden_layers // 2,
        "num_heads": 4,
        "num_units": 1024
    }
    
    print(f"Testando geração de máscara para config: {test_config}")
    head_mask, ffn_mask = search_space.config_to_mask(test_config)
    
    print(f"head_mask shape: {head_mask.shape}, dtype: {head_mask.dtype}")
    print(f"ffn_mask shape: {ffn_mask.shape}, dtype: {ffn_mask.dtype}")
    
    # Verificar valores
    print(f"head_mask valores únicos: {torch.unique(head_mask)}")
    print(f"ffn_mask valores únicos: {torch.unique(ffn_mask)}")
    
    # Verificar dimensões para Llama
    print(f"\nConfiguração Llama:")
    print(f"Número de camadas: {config.num_hidden_layers}")
    print(f"Número de cabeças: {config.num_attention_heads}")
    print(f"Tamanho intermediário: {config.intermediate_size}")
    
    # Verificar compatibilidade
    print("\nVerificando compatibilidade:")
    if head_mask.shape[0] != config.num_hidden_layers:
        print(f"AVISO: head_mask tem {head_mask.shape[0]} camadas, mas Llama tem {config.num_hidden_layers}")
        
    if head_mask.shape[1] < config.num_attention_heads:
        print(f"AVISO: head_mask tem {head_mask.shape[1]} cabeças, mas Llama tem {config.num_attention_heads}")
    
    if ffn_mask.shape[1] < config.intermediate_size:
        print(f"AVISO: ffn_mask tem {ffn_mask.shape[1]} unidades, mas Llama tem intermediate_size={config.intermediate_size}")
        
except Exception as e:
    print(f"Erro ao verificar search_spaces: {e}")
    import traceback
    traceback.print_exc()