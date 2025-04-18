from transformers.models.llama.configuration_llama import LlamaConfig as OriginalLlamaConfig

class CustomLlamaConfig(OriginalLlamaConfig):
    """
    Custom LlamaConfig class that extends the original to handle Llama 3 rope_scaling format.
    """
    
    def _rope_scaling_validation(self):
        """
        Override the validation method to handle both the original and Llama 3 format.
        """
        if self.rope_scaling is None:
            return
            
        # Check for Llama 3 format (has rope_type field)
        if isinstance(self.rope_scaling, dict) and "rope_type" in self.rope_scaling and self.rope_scaling["rope_type"] == "llama3":
            # This is a Llama 3 style rope_scaling config, so we accept it
            expected_fields = ["factor", "high_freq_factor", "low_freq_factor", "original_max_position_embeddings", "rope_type"]
            for field in expected_fields:
                if field not in self.rope_scaling:
                    print(f"Warning: Expected field '{field}' not found in rope_scaling for Llama 3")
            return
        
        # Otherwise, apply the original validation for standard LlamaConfig
        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be an float > 1, got {rope_scaling_factor}")