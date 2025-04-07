from transformers.models.llama.modeling_llama import (
    LlamaForSequenceClassification,
    LlamaModel,
)

from model_wrapper.mask import mask_llama
from search_spaces import (
    SmallSearchSpace,
    LayerSearchSpace,
    FullSearchSpace,
    MediumSearchSpace,
)


class LlamaSuperNetMixin:
    search_space = None
    handles = None

    def select_sub_network(self, sub_network_config):
        head_mask, ffn_mask = self.search_space.config_to_mask(sub_network_config)
        head_mask = head_mask.to(device="cuda", dtype=self.dtype)
        ffn_mask = ffn_mask.to(device="cuda", dtype=self.dtype)
        self.handles = mask_llama(self.model, ffn_mask, head_mask)

    def reset_super_network(self):
        for handle in self.handles:
            handle.remove()


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


class SuperNetLlamaForSequenceClassificationSMALL(
    LlamaForSequenceClassification, LlamaSuperNetMixinSMALLSpace
):
    def forward(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            return super().forward(**inputs)
        return super().forward(**inputs)


class SuperNetLlamaModelSMALL(
    LlamaModel, LlamaSuperNetMixinSMALLSpace
):
    def forward(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            return super().forward(**inputs)
        return super().forward(**inputs)


class SuperNetLlamaForSequenceClassificationLAYER(
    LlamaForSequenceClassification, LlamaSuperNetMixinLAYERSpace
):
    def forward(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            return super().forward(**inputs)
        return super().forward(**inputs)


class SuperNetLlamaModelLAYER(
    LlamaModel, LlamaSuperNetMixinLAYERSpace
):
    def forward(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            return super().forward(**inputs)
        return super().forward(**inputs)


class SuperNetLlamaForSequenceClassificationMEDIUM(
    LlamaForSequenceClassification, LlamaSuperNetMixinMEDIUMSpace
):
    def forward(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            return super().forward(**inputs)
        return super().forward(**inputs)


class SuperNetLlamaModelMEDIUM(
    LlamaModel, LlamaSuperNetMixinMEDIUMSpace
):
    def forward(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            return super().forward(**inputs)
        return super().forward(**inputs)


class SuperNetLlamaForSequenceClassificationLARGE(
    LlamaForSequenceClassification, LlamaSuperNetMixinLARGESpace
):
    def forward(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            return super().forward(**inputs)
        return super().forward(**inputs)


class SuperNetLlamaModelLARGE(
    LlamaModel, LlamaSuperNetMixinLARGESpace
):
    def forward(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            return super().forward(**inputs)
        return super().forward(**inputs)