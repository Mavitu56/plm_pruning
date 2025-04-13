from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaForSequenceClassification,
)

from model_wrapper.mask import mask_llama
from search_spaces import (
    SmallSearchSpace,
    LayerSearchSpace,
    FullSearchSpace,
    MediumSearchSpace,
)


class LLAMASuperNetMixin:
    search_space = None
    handles = None

    def select_sub_network(self, sub_network_config):
        head_mask, ffn_mask = self.search_space.config_to_mask(sub_network_config)
        head_mask = head_mask.to(device="cuda", dtype=self.dtype)
        ffn_mask = ffn_mask.to(device="cuda", dtype=self.dtype)
        
        # Access the main model structure - LlamaForCausalLM has a 'model' attribute
        if hasattr(self, "model"):
            target_model = self.model
        else:
            target_model = self
            
        self.handles = mask_llama(target_model, ffn_mask, head_mask)

    def reset_super_network(self):
        for handle in self.handles:
            handle.remove()


class LLAMASuperNetMixinLAYERSpace(LLAMASuperNetMixin):
    @property
    def search_space(self):
        return LayerSearchSpace(self.config)


class LLAMASuperNetMixinMEDIUMSpace(LLAMASuperNetMixin):
    @property
    def search_space(self):
        return MediumSearchSpace(self.config)


class LLAMASuperNetMixinLARGESpace(LLAMASuperNetMixin):
    @property
    def search_space(self):
        return FullSearchSpace(self.config)


class LLAMASuperNetMixinSMALLSpace(LLAMASuperNetMixin):
    @property
    def search_space(self):
        return SmallSearchSpace(self.config)


# CausalLM variants (for text generation)
class SuperNetLlamaForCausalLMSMALL(
    LlamaForCausalLM, LLAMASuperNetMixinSMALLSpace
):
    def forward(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            return super().forward(**inputs)
        return super().forward(**kwargs)


class SuperNetLlamaForCausalLMLAYER(
    LlamaForCausalLM, LLAMASuperNetMixinLAYERSpace
):
    def forward(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            return super().forward(**inputs)
        return super().forward(**kwargs)


class SuperNetLlamaForCausalLMMEDIUM(
    LlamaForCausalLM, LLAMASuperNetMixinMEDIUMSpace
):
    def forward(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            return super().forward(**inputs)
        return super().forward(**kwargs)


class SuperNetLlamaForCausalLMLARGE(
    LlamaForCausalLM, LLAMASuperNetMixinLARGESpace
):
    def forward(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            return super().forward(**inputs)
        return super().forward(**kwargs)


# SequenceClassification variants (for classification tasks)
class SuperNetLlamaForSequenceClassificationSMALL(
    LlamaForSequenceClassification, LLAMASuperNetMixinSMALLSpace
):
    def forward(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            return super().forward(**inputs)
        return super().forward(**kwargs)


class SuperNetLlamaForSequenceClassificationLAYER(
    LlamaForSequenceClassification, LLAMASuperNetMixinLAYERSpace
):
    def forward(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            return super().forward(**inputs)
        return super().forward(**kwargs)


class SuperNetLlamaForSequenceClassificationMEDIUM(
    LlamaForSequenceClassification, LLAMASuperNetMixinMEDIUMSpace
):
    def forward(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            return super().forward(**inputs)
        return super().forward(**kwargs)


class SuperNetLlamaForSequenceClassificationLARGE(
    LlamaForSequenceClassification, LLAMASuperNetMixinLARGESpace
):
    def forward(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            return super().forward(**inputs)
        return super().forward(**kwargs)