"""
# Adapted from https://huggingface.co/MILVLG/imp-v1-3b/blob/main/vision_encoder.py
"""

from typing import Optional, Tuple, Union, Dict
from dataclasses import dataclass
from functools import partial, reduce
from PIL import Image
import torch
import torch.utils.checkpoint
from torch import nn
import os
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig
from transformers.utils import ModelOutput
# from llava.utils import rank0_print


class SigLipImageProcessor:
    def __init__(self, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5), size=(384, 384), crop_size: Dict[str, int] = None, resample=PILImageResampling.BICUBIC, rescale_factor=1 / 255, data_format=ChannelDimension.FIRST):
        crop_size = crop_size if crop_size is not None else {"height": 384, "width": 384}
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size

    def preprocess(self, images, return_tensors):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            # to adapt video data
            images = [to_numpy_array(image) for image in images]
            assert isinstance(images, list)

        transforms = [
            convert_to_rgb,
            to_numpy_array,
            partial(resize, size=self.size, resample=self.resample, data_format=self.data_format),
            partial(rescale, scale=self.rescale_factor, data_format=self.data_format),
            partial(normalize, mean=self.image_mean, std=self.image_std, data_format=self.data_format),
            partial(to_channel_dimension_format, channel_dim=self.data_format, input_channel_dim=self.data_format),
        ]

        images = reduce(lambda x, f: [*map(f, x)], transforms, images)
        data = {"pixel_values": images}

        return BatchFeature(data=data, tensor_type=return_tensors)


class SigLipVisionConfig(PretrainedConfig):
    model_type = "siglip_vision_model"

    def __init__(
        self,
        hidden_size=1152,
        image_mean=(0.5, 0.5, 0.5),
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_channels=3,
        image_size=384,
        patch_size=14,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.image_mean = image_mean
        print("INIT EXECUTED")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from SigLipConfig
        if config_dict.get("model_type") == "siglip":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            print(f"You are using a model of type {config_dict['model_type']} to instantiate a model of type " f"{cls.model_type}. This is not supported for all configurations of models and can yield errors.")

        return cls.from_dict(config_dict, **kwargs)


@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPVisionModelOutput with CLIP->SigLip
class SigLipVisionModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.

    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class SigLipVisionEmbeddings(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        embeddings = embeddings + self.position_embedding(self.position_ids.long())
        return embeddings


class SigLipAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Copied from transformers.models.clip.modeling_clip.CLIPAttention.__init__
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:" f" {self.num_heads}).")
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        k_v_seq_len = key_states.shape[-2]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
            raise ValueError(f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is" f" {attn_weights.size()}")

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, q_len, k_v_seq_len):
                raise ValueError(f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, but is {attention_mask.size()}")
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is" f" {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->SigLip
class SigLipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# Copied from transformers.models.clip.modeling_clip.CLIPEncoderLayer with CLIP->SigLip
class SigLipEncoderLayer(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SigLipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SigLipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class SigLipPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SigLipVisionConfig
    base_model_prefix = "siglip"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        pass


# Copied from transformers.models.clip.modeling_clip.CLIPEncoder with CLIP->SigLip
class SigLipEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`SigLipEncoderLayer`].

    Args:
        config: SigLipVisionConfig
    """

    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SigLipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # Ignore copy
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)


class SigLipVisionTransformer(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SigLipVisionEmbeddings(config)
        self.encoder = SigLipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.head = SigLipMultiheadAttentionPoolingHead(config)

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooled_output = self.head(last_hidden_state)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class SigLipMultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, config: SigLipVisionConfig):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SigLipMLP(config)

    def forward(self, hidden_state):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


class SigLipVisionModel(SigLipPreTrainedModel):
    config_class = SigLipVisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["SigLipEncoderLayer"]

    def __init__(self, config: SigLipVisionConfig):
        super().__init__(config)

        self.vision_model = SigLipVisionTransformer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, SigLipVisionModel

        >>> model = SigLipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled features
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class SigLipVisionTower(nn.Module):
    def __init__(self, vision_tower, vision_tower_cfg, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.config = SigLipVisionConfig()

        self.vision_tower_name = vision_tower

        self.image_processor = SigLipImageProcessor()
        
        # Attention map extraction configuration
        # When enabled, the forward method will also return attention maps
        self.with_attn_map = False
        self.attn_map_method = "rollout"  # Options: "rollout", "raw", "mean", "last_layer"
        self.attn_head_fusion = "mean"  # Options: "mean", "max", "min"
        self.attn_discard_ratio = 0.9  # For rollout method: ratio of lowest values to discard
        self.attn_return_all_layers = False  # Whether to return per-layer attention maps

        if not delay_load:
            # rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(vision_tower_cfg, "unfreeze_mm_vision_tower", False):
            # TODO: better detector is needed.
            # rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(vision_tower_cfg, "mm_tunable_parts") and "mm_vision_tower" in vision_tower_cfg.mm_tunable_parts:
            # rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()
        else:
            self.cfg_only = self.config

    def load_model(self, device_map=None):
        if self.is_loaded:
            # rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return

        self.vision_tower = SigLipVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)

        del self.vision_tower.vision_model.encoder.layers[-1:]
        self.vision_tower.vision_model.head = nn.Identity()
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def forward(self, images):
        """
        Forward pass through the vision tower.
        
        Args:
            images: Input images tensor of shape (batch_size, channels, height, width)
                   or list of individual image tensors.
        
        Returns:
            If self.with_attn_map is False (default):
                tuple: (image_features, image_features_middle)
                    - image_features: Final layer features (batch_size, 729, hidden_size)
                    - image_features_middle: List of intermediate layer features
            
            If self.with_attn_map is True:
                tuple: (image_features, image_features_middle, attention_maps)
                    - image_features: Final layer features
                    - image_features_middle: List of intermediate layer features
                    - attention_maps: Attention maps computed using the configured method
                                    Shape (batch_size, num_patches, num_patches) or dict if
                                    attn_return_all_layers is True
        """
        layers=[6, 12, 18, 24]
        
        if type(images) is list:
            # raise error here
            raise ValueError("Not implemented yet")
            image_features = []
            image_features_middle = []
            for image in images:
                temp_image_features_middle = []
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0), 
                    output_hidden_states=True,
                    output_attentions=self.with_attn_map  # Enable attentions if needed
                )
                image_feature = image_forward_out.hidden_states[-1].to(image.dtype)
                assert image_features.shape[-2] == 729
                image_features.append(image_feature)
                for i, d in enumerate(image_forward_out.hidden_states):
                    # print(f'Layer {i}: {d.shape}')
                    if i-1 in layers:
                        temp_image_features_middle.append(d.to(images.dtype))
                image_features_middle.append(temp_image_features_middle) # len(images) x len(layers) x (1, sequence_length, hidden_size)
        else:
            # Forward pass through vision tower
            # Enable attention output if attention maps are requested
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype), 
                output_hidden_states=True,
                output_attentions=self.with_attn_map  # Only compute attentions when needed
            )
            
            image_features = image_forward_outs.hidden_states[-1].to(images.dtype)
            assert image_features.shape[-2] == 729

            # Extract intermediate layer features at specified layers
            image_features_middle = []
            for i, d in enumerate(image_forward_outs.hidden_states):
                # print(f'Layer {i}: {d.shape}')
                if i-1 in layers:
                    image_features_middle.append(d.to(images.dtype)) # len(layers) x (batch_size, sequence_length, hidden_size)

            # If attention maps are requested, extract them using the configured method
            if self.with_attn_map:
                attention_maps = self._extract_attention_maps(image_forward_outs.attentions)
                return image_features, image_features_middle, attention_maps

        return image_features, image_features_middle

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size
        # return self.model_config["vision_cfg"]["image_size"] // self.model_config["vision_cfg"]["patch_size"]

    @property
    def image_size(self):
        return self.config.image_size

    def _extract_attention_maps(
        self,
        attentions: Tuple[torch.Tensor]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Internal method to extract attention maps from attention tensors using classical methods.
        
        This method is called internally by the forward() method when self.with_attn_map is True.
        It processes the raw attention tensors from all transformer layers and aggregates them
        according to the configured method (rollout, raw, mean, or last_layer).
        
        The method implements several classical attention visualization techniques:
        - Attention Rollout: Multiplies attention matrices across layers to track information
          flow through the network (Abnar & Zuidema, 2020)
        - Raw: Returns attention weights from the last layer only
        - Mean: Averages attention weights across all layers
        - Last Layer: Uses only the final transformer layer with residual connections
        
        Args:
            attentions (Tuple[torch.Tensor]): Tuple of attention tensors from each layer.
                                              Each tensor has shape (batch_size, num_heads, seq_len, seq_len).
                                              Obtained from the transformer encoder output.
        
        Returns:
            Union[torch.Tensor, Dict[str, torch.Tensor]]:
                If self.attn_return_all_layers is False:
                    - torch.Tensor of shape (batch_size, num_patches, num_patches) containing
                      the aggregated attention map where element [b,i,j] represents attention
                      from patch i to patch j in batch element b.
                
                If self.attn_return_all_layers is True:
                    - Dict with keys 'layer_0', 'layer_1', ..., 'layer_N' and 'final',
                      where each value is a tensor of shape (batch_size, num_patches, num_patches).
                      This allows inspection of how attention evolves through the network.
        
        Note:
            - Configuration is controlled by instance attributes:
              * self.attn_map_method: "rollout", "raw", "mean", or "last_layer"
              * self.attn_head_fusion: "mean", "max", or "min"
              * self.attn_discard_ratio: float in [0, 1] for noise reduction (rollout only)
              * self.attn_return_all_layers: bool to return per-layer maps
            - For SigLip with 384x384 images and patch_size=14, you get 729 patches (27x27)
            - All positions represent spatial patches (no CLS token in SigLip)
        """
        # Store individual layer attention maps if requested
        layer_attention_maps = {}
        
        # Process attention based on the configured method
        if self.attn_map_method == "raw":
            # Return raw attention from last layer, fused across heads
            # This shows the unprocessed attention weights from the final layer
            last_layer_attn = attentions[-1]  # (batch_size, num_heads, seq_len, seq_len)
            attention_map = self._fuse_heads(last_layer_attn, self.attn_head_fusion)
            
            if self.attn_return_all_layers:
                # Store attention maps for each layer
                for i, attn in enumerate(attentions):
                    layer_attention_maps[f"layer_{i}"] = self._fuse_heads(attn, self.attn_head_fusion)
                layer_attention_maps["final"] = attention_map
                return layer_attention_maps
            return attention_map
        
        elif self.attn_map_method == "last_layer":
            # Use only the last layer's attention, but add identity matrix
            # to account for residual connections in the transformer
            last_layer_attn = attentions[-1]  # (batch_size, num_heads, seq_len, seq_len)
            attention_map = self._fuse_heads(last_layer_attn, self.attn_head_fusion)
            
            # Add identity matrix to account for residual connections
            # Each token also "attends to itself" through skip connections
            batch_size, seq_len, _ = attention_map.shape
            eye = torch.eye(seq_len, device=attention_map.device, dtype=attention_map.dtype)
            eye = eye.unsqueeze(0).expand(batch_size, -1, -1)
            attention_map = attention_map + eye
            
            # Normalize so each row sums to 1 (valid probability distribution)
            attention_map = attention_map / attention_map.sum(dim=-1, keepdim=True)
            
            if self.attn_return_all_layers:
                for i, attn in enumerate(attentions):
                    layer_attn = self._fuse_heads(attn, self.attn_head_fusion)
                    layer_attn = layer_attn + eye
                    layer_attn = layer_attn / layer_attn.sum(dim=-1, keepdim=True)
                    layer_attention_maps[f"layer_{i}"] = layer_attn
                layer_attention_maps["final"] = attention_map
                return layer_attention_maps
            return attention_map
        
        elif self.attn_map_method == "mean":
            # Average attention across all layers
            # First fuse heads for each layer, then average across layers
            # This gives equal weight to all layers in the network
            fused_attentions = []
            for i, attn in enumerate(attentions):
                fused_attn = self._fuse_heads(attn, self.attn_head_fusion)
                fused_attentions.append(fused_attn)
                if self.attn_return_all_layers:
                    layer_attention_maps[f"layer_{i}"] = fused_attn
            
            # Stack and take mean across layer dimension
            attention_map = torch.stack(fused_attentions, dim=0).mean(dim=0)
            
            if self.attn_return_all_layers:
                layer_attention_maps["final"] = attention_map
                return layer_attention_maps
            return attention_map
        
        elif self.attn_map_method == "rollout":
            # Attention Rollout: multiply attention matrices layer by layer
            # This tracks how attention flows through the entire network
            # Reference: "Quantifying Attention Flow in Transformers" (Abnar & Zuidema, 2020)
            
            # Start with identity matrix representing initial "attention"
            # Initially, each token only attends to itself
            batch_size = attentions[0].shape[0]
            seq_len = attentions[0].shape[2]
            
            # Initialize rollout accumulator with identity matrix
            rollout = torch.eye(seq_len, device=self.device, dtype=self.dtype)
            rollout = rollout.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Iterate through each layer and accumulate attention
            for i, attn in enumerate(attentions):
                # Fuse attention heads using the configured method
                attn_heads_fused = self._fuse_heads(attn, self.attn_head_fusion)
                
                # Add identity matrix to account for residual connections
                # Residual connections allow information to bypass attention layers
                eye = torch.eye(seq_len, device=attn_heads_fused.device, dtype=attn_heads_fused.dtype)
                eye = eye.unsqueeze(0).expand(batch_size, -1, -1)
                attn_with_residual = attn_heads_fused + eye
                
                # Normalize so each row sums to 1 (valid probability distribution)
                attn_with_residual = attn_with_residual / attn_with_residual.sum(dim=-1, keepdim=True)
                
                # Apply discard ratio: zero out lowest attention values to reduce noise
                # This helps focus on the most salient connections in the network
                if self.attn_discard_ratio > 0:
                    # Flatten attention values to find threshold
                    flat_attn = attn_with_residual.view(batch_size, -1)
                    # Find the value at the discard_ratio percentile
                    threshold_idx = int(flat_attn.shape[1] * self.attn_discard_ratio)
                    sorted_attn, _ = torch.sort(flat_attn, dim=1)
                    threshold = sorted_attn[:, threshold_idx].unsqueeze(1).unsqueeze(2)
                    
                    # Zero out values below threshold (noise reduction)
                    attn_with_residual = torch.where(
                        attn_with_residual < threshold,
                        torch.zeros_like(attn_with_residual),
                        attn_with_residual
                    )
                    
                    # Re-normalize after discarding low values
                    attn_with_residual = attn_with_residual / attn_with_residual.sum(dim=-1, keepdim=True)
                
                # Matrix multiplication to accumulate attention across layers
                # rollout @ attn gives us how attention flows from input through current layer
                rollout = torch.matmul(attn_with_residual, rollout)
                
                # Store intermediate rollout if requested
                if self.attn_return_all_layers:
                    layer_attention_maps[f"layer_{i}"] = rollout.clone()
            
            attention_map = rollout
            
            if self.attn_return_all_layers:
                layer_attention_maps["final"] = attention_map
                return layer_attention_maps
            return attention_map
        
        else:
            raise ValueError(
                f"Unknown attention map method: {self.attn_map_method}. "
                f"Choose from: rollout, raw, mean, last_layer"
            )
    
    def _fuse_heads(self, attention: torch.Tensor, fusion: str = "mean") -> torch.Tensor:
        """
        Fuse attention from multiple heads into a single attention matrix.
        
        Multi-head attention allows the model to attend to different aspects of the input
        simultaneously (e.g., different semantic features, spatial relationships, etc.).
        This method combines these multiple attention heads into a single unified attention
        map for easier interpretation and visualization.
        
        Args:
            attention (torch.Tensor): Attention tensor of shape 
                                     (batch_size, num_heads, seq_len, seq_len).
                                     Each element [b, h, i, j] represents how much head h
                                     in batch b attends from position i to position j.
            
            fusion (str): Method to fuse attention heads:
                - "mean": Average attention across heads (most common approach).
                         Gives a balanced view of attention from all heads.
                - "max": Take maximum attention value across heads.
                        Highlights the strongest connections any head found.
                - "min": Take minimum attention value across heads.
                        Shows where all heads agree there is attention.
        
        Returns:
            torch.Tensor: Fused attention tensor of shape (batch_size, seq_len, seq_len).
                         Each element [b, i, j] represents the aggregated attention from
                         position i to position j for batch element b.
        
        Example:
            >>> # Example with 2 images, 16 attention heads, 729 patches
            >>> attention = torch.randn(2, 16, 729, 729)
            >>> fused = self._fuse_heads(attention, fusion="mean")
            >>> fused.shape  # torch.Size([2, 729, 729])
        
        Note:
            - The choice of fusion method can significantly affect the interpretation:
              * "mean" provides a smooth, averaged view
              * "max" emphasizes strong connections but may be noisy
              * "min" is conservative, showing only consensus attention
        """
        if fusion == "mean":
            # Average across the head dimension (dim=1)
            # This gives equal weight to all attention heads
            return attention.mean(dim=1)
        elif fusion == "max":
            # Take the maximum value across the head dimension
            # This highlights the strongest attention patterns
            return attention.max(dim=1)[0]
        elif fusion == "min":
            # Take the minimum value across the head dimension
            # This shows where all heads agree there is attention
            return attention.min(dim=1)[0]
        else:
            raise ValueError(
                f"Unknown fusion method: {fusion}. "
                f"Choose from: mean, max, min"
            )
