"""
LLaVA Model Module

This module provides the model components for Anomaly-OV.
Language model imports are optional and only needed for the full LLaVA pipeline.
For anomaly detection, only anomaly_expert and multimodal_encoder are required.
"""
import os

# Export core components for anomaly detection
from .anomaly_expert import AnomalyOV

# Language model imports are optional (used for full LLaVA pipeline)
AVAILABLE_MODELS = {
    "llava_llama": "LlavaLlamaForCausalLM, LlavaConfig",
    "llava_qwen": "LlavaQwenForCausalLM, LlavaQwenConfig",
    "llava_mistral": "LlavaMistralForCausalLM, LlavaMistralConfig",
    "llava_mixtral": "LlavaMixtralForCausalLM, LlavaMixtralConfig",
}

# Silently try to import language models (optional for detection-only usage)
for model_name, model_classes in AVAILABLE_MODELS.items():
    try:
        exec(f"from .language_model.{model_name} import {model_classes}")
    except ImportError:
        pass  # Language models are optional for detection-only mode
    except Exception:
        pass  # Silently ignore other import errors

