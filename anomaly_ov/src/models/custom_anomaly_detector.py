import os
from llava.model.anomaly_expert import AnomalyOV
from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
import torch.nn as nn
import torch
class OVAnomalyDetector(nn.Module):
    def __init__(self, anomaly_expert: AnomalyOV, vision_encoder: SigLipVisionTower, dtype=None):
        super(OVAnomalyDetector, self).__init__()
        self.anomaly_expert = anomaly_expert
        self.vision_encoder = vision_encoder
        self.dtype = dtype
        self.config = type('Config', (), {})()  # Simple config object for compatibility
    
    def save_checkpoint(self, save_path, optimizer=None, scheduler=None, epoch=None, 
                        best_eval_loss=None, best_eval_acc=None, epochs_without_improvement=None, history=None):
        """
        Save the complete model state to a single .pt file.
        
        Args:
            save_path: Path where to save the checkpoint (e.g., 'model_checkpoint.pt')
            optimizer: Optional optimizer to save state
            scheduler: Optional scheduler to save state
            epoch: Optional current epoch number
            best_eval_loss: Optional best evaluation loss so far
            best_eval_acc: Optional best evaluation accuracy so far
            epochs_without_improvement: Optional counter for early stopping
            history: Optional training history dict
        """
        checkpoint = {
            'vision_encoder_state_dict': self.vision_encoder.state_dict(),
            'anomaly_expert_state_dict': self.anomaly_expert.state_dict() if self.anomaly_expert else None,
            'dtype': str(self.dtype),
            'vision_tower_name': getattr(self.vision_encoder, 'vision_tower_name', 'google/siglip-so400m-patch14-384'),
        }
        
        # Save training state if provided
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if best_eval_loss is not None:
            checkpoint['best_eval_loss'] = best_eval_loss
        if best_eval_acc is not None:
            checkpoint['best_eval_acc'] = best_eval_acc
        if epochs_without_improvement is not None:
            checkpoint['epochs_without_improvement'] = epochs_without_improvement
        if history is not None:
            checkpoint['history'] = history
            
        torch.save(checkpoint, save_path)
        print(f"Model checkpoint saved to: {save_path}")
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, device='cuda', device_map='auto'):
        """
        Load the model state from a .pt checkpoint file.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            device: Device to load the model on ('cuda', 'cpu', etc.)
            device_map: Device mapping strategy (default: 'auto')
        
        Returns:
            OVAnomalyDetector: Loaded detector instance
            image_processor: Image processor from vision tower
        """
        print(f"Loading model checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Parse dtype
        dtype_str = checkpoint.get('dtype', 'torch.float32')
        dtype = getattr(torch, dtype_str.replace('torch.', '')) if isinstance(dtype_str, str) else torch.float32
        
        # Fallback on CPU if unsupported
        if str(device).startswith('cpu') and dtype in (torch.float16, torch.bfloat16):
            print(f"Warning: Requested dtype {dtype} on CPU; falling back to float32.")
            dtype = torch.float32
        
        # Build vision tower
        vision_tower_name = checkpoint.get('vision_tower_name', 'google/siglip-so400m-patch14-384')
        print(f"Loading vision tower: {vision_tower_name}")
        vision_tower = SigLipVisionTower(vision_tower_name, vision_tower_cfg={}, delay_load=False)
        vision_tower.load_state_dict(checkpoint['vision_encoder_state_dict'])
        vision_tower.to(device)
        vision_tower.to(dtype=dtype)
        vision_tower.requires_grad_(False)
        vision_tower.eval()
        
        # Build anomaly expert if present
        anomaly_expert = None
        if checkpoint.get('anomaly_expert_state_dict') is not None:
            print("Loading anomaly expert...")
            anomaly_expert = AnomalyOV()
            anomaly_expert.load_state_dict(checkpoint['anomaly_expert_state_dict'])
            anomaly_expert.to(dtype=dtype, device=device)
            anomaly_expert.requires_grad_(False)
            anomaly_expert.eval()
        
        # Create detector
        detector = cls(
            anomaly_expert=anomaly_expert,
            vision_encoder=vision_tower,
            dtype=dtype
        )
        detector.eval()
        
        image_processor = vision_tower.image_processor
        
        print("Model loaded successfully from checkpoint!")
        return detector, image_processor

    def set_anomaly_encoder(self, anomaly_encoder):
        """Set or update the anomaly encoder/expert."""
        self.anomaly_expert = anomaly_encoder

    def get_vision_tower(self):
        return self.vision_encoder

    def encode_images(self, images, with_attention_map=False):

        if with_attention_map:
            self.vision_encoder.with_attn_map = True 
            image_features, image_level_features, attn_maps = self.vision_encoder(images)
            return image_features, image_level_features, attn_maps
        
        image_features, image_level_features = self.vision_encoder(images)
        return image_features, image_level_features
    
    def get_anomaly_tokens(self, ov_image_features, sig_multi_level_features, split_sizes, return_anomaly_map=False, anomaly_map_size=(224, 224)):
        return self.anomaly_expert(
            ov_image_features, 
            sig_multi_level_features, 
            split_sizes, 
            return_anomaly_map=return_anomaly_map, 
            anomaly_map_size=anomaly_map_size
        )
    
    def get_anomaly_features(self, ov_image_features, sig_multi_level_features, split_sizes, return_anomaly_map=False, anomaly_map_size=(224, 224)):
        if return_anomaly_map:
            _, _, final_prediction, anomaly_map = self.get_anomaly_tokens(ov_image_features, sig_multi_level_features, split_sizes, return_anomaly_map=return_anomaly_map, anomaly_map_size=anomaly_map_size)
        else:
            _, _, final_prediction = self.get_anomaly_tokens(ov_image_features, sig_multi_level_features, split_sizes, return_anomaly_map=return_anomaly_map)

        if return_anomaly_map:
            return final_prediction, anomaly_map
        return final_prediction
    
    def get_anomaly_fetures_from_images(self, images, with_attention_map=False, anomaly_map_size=(224, 224)):

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]   

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))
            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]

            if with_attention_map:

                encoded_image_features, encoded_image_level_features, attn_maps = self.encode_images(concat_images, with_attention_map)
                final_preds, anomaly_map = self.get_anomaly_features(encoded_image_features, encoded_image_level_features, split_sizes, return_anomaly_map=with_attention_map, anomaly_map_size=anomaly_map_size)
                return final_preds, attn_maps, anomaly_map

            else:
                encoded_image_features, encoded_image_level_features = self.encode_images(concat_images, with_attention_map)
                final_preds = self.get_anomaly_features(encoded_image_features, encoded_image_level_features, split_sizes, return_anomaly_map=with_attention_map, anomaly_map_size=anomaly_map_size)

                return final_preds
        else:
            raise ValueError("Not implemented yet")
        
    
def build_ov_anomaly_detector(
    vision_tower_name="google/siglip-so400m-patch14-384",
    anomaly_expert_path="./pretrained_expert_7b.pth",
    device="cuda",
    dtype=None,
) -> OVAnomalyDetector:
    """
    Build and return an OVAnomalyDetector instance with the same components
    that would be inside a model created by load_pretrained_model, but without
    loading the full language model.
    
    Args:
        vision_tower_name: HuggingFace model ID for the vision encoder
                          (default: "google/siglip-so400m-patch14-384")
        anomaly_expert_path: Path to the pretrained anomaly expert weights
                           (use './pretrained_expert_05b.pth' for 0.5B models,
                            './pretrained_expert_7b.pth' for 7B models)
        device: Device to load the models on ('cuda', 'cpu', etc.)
        dtype: Data type for the models (torch.float16, torch.bfloat16, or None for float32)
        load_pretrained_projector: Path to pretrained projector weights (e.g., from 
                                   model_path/mm_projector.bin). If None, uses random init.
    
    Returns:
        OVAnomalyDetector: Configured detector instance
    """

    
    # Normalize dtype argument: allow strings like "bfloat16", "fp16", etc.
    if dtype is None:
        dtype = torch.float32
    else:
        if isinstance(dtype, str):
            dtype_map = {
                "float32": torch.float32, "fp32": torch.float32, "float": torch.float32,
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                "float16": torch.float16, "fp16": torch.float16, "half": torch.float16
            }
            dtype = dtype_map.get(dtype.lower(), torch.float32)
        # Fallback on CPU if unsupported low-precision requested
        if str(device).startswith("cpu") and dtype in (torch.float16, torch.bfloat16):
            print(f"[build_ov_anomaly_detector] Warning: Requested dtype {dtype} on CPU; falling back to float32.")
            dtype = torch.float32
    
    # 1. Build SigLip Vision Tower
    print(f"Loading vision tower: {vision_tower_name}")
   
    vision_tower = SigLipVisionTower(vision_tower_name, vision_tower_cfg={}, delay_load=False)
    vision_tower.to(device)
    vision_tower.to(dtype=dtype)
    vision_tower.requires_grad_(False)
    vision_tower.eval()


    print("vision tower dtype:", next(vision_tower.parameters()).dtype)

    
    # 3. Build Anomaly Expert
    if anomaly_expert_path is not None:
        print(f"Loading anomaly expert from: {anomaly_expert_path}")
        anomaly_expert = AnomalyOV()
        anomaly_expert.load_zero_shot_weights(path=anomaly_expert_path)
        anomaly_expert.freeze_layers()
        anomaly_expert.to(dtype=dtype, device=device)
        anomaly_expert.requires_grad_(False)
        anomaly_expert.eval()
        print("anomaly expert dtype:", next(anomaly_expert.parameters()).dtype)
    else:
        print("Skipping anomaly expert loading (path is None)")
        anomaly_expert = None


    # 4. Create the OVAnomalyDetector
    print("Building OVAnomalyDetector...")
    detector = OVAnomalyDetector(
        anomaly_expert=anomaly_expert,
        vision_encoder=vision_tower,
        dtype=dtype
    )

    
    image_processor = vision_tower.image_processor
    
    
    detector.eval()
    print("OVAnomalyDetector built successfully!")
    print(f"  - Vision tower: SigLipVisionTower")


    
    return detector, image_processor


if __name__ == "__main__":
    # Example 1: Build model and save checkpoint
    print("=" * 50)
    print("Building and saving model...")
    print("=" * 50)
    detector, image_processor = build_ov_anomaly_detector(
        vision_tower_name="google/siglip-so400m-patch14-384",
        anomaly_expert_path="./pretrained_expert_7b.pth",
        device="cpu",
        dtype=torch.float32,
    )
    
    # Save the checkpoint
    checkpoint_path = "ov_anomaly_detector_checkpoint.pt"
    detector.save_checkpoint(checkpoint_path)
    
    print("\n" + "=" * 50)
    print("Loading model from checkpoint...")
    print("=" * 50)
    
    # Example 2: Load model from checkpoint
    loaded_detector, loaded_image_processor = OVAnomalyDetector.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device="cpu"
    )
    
    print("\nDone! Model saved and loaded successfully.")
    