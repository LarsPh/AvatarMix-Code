"""
Identity Loss Implementation using CLIP for DiFix3D+ Training

This module provides CLIP-based identity preservation loss for maintaining
visual identity consistency during head refinement.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Union, Any
from loguru import logger

# Import open_clip for differentiable CLIP functionality
try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    logger.error("open_clip not available. Install with: pip install open_clip_torch")



class CLIPIdentityLoss:
    """
    CLIP-based identity loss for maintaining visual identity in head regions.
    
    Computes direct image-to-image similarity between refined heads and ground truth heads
    using CLIP visual embeddings without requiring text prompts.
    """
    
    def __init__(
        self,
        model_name_or_path: str = "ViT-B-32",
        pretrained: str = "openai",
        loss_weight: float = 0.1,
        similarity_target: float = 1.0,
        reduction: str = "mean",
        device: Optional[torch.device] = None
    ):
        """
        Initialize CLIP identity loss with open_clip.
        
        Args:
            model_name_or_path: open_clip model name (e.g., "ViT-B-32")
            pretrained: pretrained weights (e.g., "openai")
            loss_weight: Weight for identity loss in total loss
            similarity_target: Target similarity score (0-1, open_clip range)
            reduction: Reduction method ("mean", "sum", "none")
            device: Device for computation
        """
        self.model_name = model_name_or_path
        self.pretrained = pretrained
        self.loss_weight = loss_weight
        self.similarity_target = similarity_target
        self.reduction = reduction
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize open_clip model
        self.clip_model = self._initialize_clip_model()
        
        if self.clip_model is None:
            raise RuntimeError("Failed to initialize open_clip model for identity loss")
        
        # CLIP normalization constants (from open_clip preprocessing)
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)
        
        logger.info(f"CLIPIdentityLoss initialized with open_clip:")
        logger.info(f"  Model: {model_name_or_path}")
        logger.info(f"  Pretrained: {pretrained}")
        logger.info(f"  Method: Differentiable image-to-image similarity")
        logger.info(f"  Loss weight: {loss_weight}")
        logger.info(f"  Target similarity: {similarity_target}")
        logger.info(f"  Device: {self.device}")
    
    def _initialize_clip_model(self) -> Optional[Any]:
        """
        Initialize open_clip model for differentiable feature extraction.
            
        Returns:
            open_clip model instance or None if failed
        """
        if not OPEN_CLIP_AVAILABLE:
            logger.error("open_clip not available for identity loss")
            return None
            
        try:
            # Load open_clip model (differentiable!)
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name, 
                pretrained=self.pretrained
            )
            
            # Move to device and set to eval (but gradients still flow)
            model = model.to(self.device)
            model.eval()
            
            logger.info(f"Successfully loaded open_clip model: {self.model_name} ({self.pretrained})")
            return model
                
        except Exception as e:
            logger.error(f"Failed to initialize open_clip model: {e}")
            return None
    
    def compute_identity_loss(
        self,
        refined_heads: torch.Tensor,
        gt_heads: torch.Tensor,
        success_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute identity preservation loss between refined and ground truth heads.
        
        Args:
            refined_heads: Refined head crops (N, C, H, W) in [0, 1] range
            gt_heads: Ground truth head crops (N, C, H, W) in [0, 1] range
            success_mask: Boolean mask indicating valid head crops (N,)
            
        Returns:
            Dictionary with loss components:
            - identity_loss: Main identity loss tensor
            - clip_similarity: Raw CLIP similarity scores
            - valid_samples: Number of valid samples
        """
        batch_size = refined_heads.shape[0]
        
        # Apply success mask if provided
        if success_mask is not None:
            valid_indices = success_mask.nonzero(as_tuple=True)[0]
            if len(valid_indices) == 0:
                logger.warning("No valid head crops for identity loss computation")
                return self._create_zero_loss(batch_size)
            
            refined_heads = refined_heads[valid_indices]
            gt_heads = gt_heads[valid_indices]
            valid_samples = len(valid_indices)
        else:
            valid_samples = batch_size
        
        # Compute CLIP similarities (differentiable!)
        try:
            clip_similarities = self._compute_clip_similarities(refined_heads, gt_heads)
            
            # Convert similarity to loss
            # Loss = (target - similarity) clamped to positive values
            # For perfect similarity (1.0), loss = 0.0
            clip_loss = torch.clamp(self.similarity_target - clip_similarities, min=0.0)
            
            # Apply reduction
            if self.reduction == "mean":
                clip_loss = clip_loss.mean()
            elif self.reduction == "sum":
                clip_loss = clip_loss.sum()
            # "none" keeps individual losses
            
            # Apply loss weight
            weighted_clip_loss = clip_loss * self.loss_weight
            
            return {
                "clip_loss": weighted_clip_loss,
                "clip_similarity": clip_similarities.mean() if clip_similarities.dim() > 0 else clip_similarities,
                "valid_samples": torch.tensor(valid_samples, device=self.device)
            }
            
        except Exception as e:
            logger.error(f"Error computing identity loss: {e}")
            return self._create_zero_loss(batch_size)
    
    def _compute_clip_similarities(
        self,
        refined_heads: torch.Tensor,
        gt_heads: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute differentiable CLIP similarities using open_clip features.
        
        Args:
            refined_heads: Refined head tensors (N, C, H, W) in [0, 1] range
            gt_heads: GT head tensors (N, C, H, W) in [0, 1] range
            
        Returns:
            Cosine similarity scores tensor (N,) in [0, 1] range
        """
        if self.clip_model is None:
            raise RuntimeError("CLIP model not initialized")
        
        try:
            # Normalize inputs to CLIP preprocessing range
            refined_normalized = self._normalize_for_clip(refined_heads)
            gt_normalized = self._normalize_for_clip(gt_heads)
            
            # Extract features (differentiable!)
            with torch.set_grad_enabled(True):  # Ensure gradients are enabled
                refined_features = self.clip_model.encode_image(refined_normalized)
                gt_features = self.clip_model.encode_image(gt_normalized)
                
                # Normalize features for cosine similarity
                refined_features = refined_features / refined_features.norm(dim=-1, keepdim=True)
                gt_features = gt_features / gt_features.norm(dim=-1, keepdim=True)
                
                # Compute cosine similarity (differentiable)
                similarities = (refined_features * gt_features).sum(dim=-1)
                
                # Ensure similarities are in [0, 1] range (cosine can be [-1, 1])
                # For identity preservation, we want similarities close to 1.0
                similarities = torch.clamp(similarities, min=0.0, max=1.0)
                
            return similarities
                
        except Exception as e:
            logger.error(f"Error computing open_clip similarities: {e}")
            raise e
    
    def _normalize_for_clip(self, images: torch.Tensor) -> torch.Tensor:
        """
        Normalize [0,1] image tensors to CLIP preprocessing format.
        
        Args:
            images: Input images (N, C, H, W) in [0, 1] range
            
        Returns:
            CLIP-normalized images (N, C, H, W)
        """
        # Apply CLIP normalization: (x - mean) / std
        normalized = (images - self.clip_mean) / self.clip_std
        return normalized
    
    
    def _create_zero_loss(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Create zero loss dictionary for error cases."""
        return {
            "identity_loss": torch.zeros(1, device=self.device, requires_grad=True),
            "clip_similarity": torch.zeros(1, device=self.device),
            "valid_samples": torch.tensor(0, device=self.device)
        }
    
    def validate_inputs(
        self,
        refined_heads: torch.Tensor,
        gt_heads: torch.Tensor
    ) -> bool:
        """
        Validate input tensors for identity loss computation.
        
        Args:
            refined_heads: Refined head tensors
            gt_heads: GT head tensors
            
        Returns:
            True if inputs are valid
        """
        try:
            # Check tensor shapes
            if refined_heads.shape != gt_heads.shape:
                logger.warning(f"Shape mismatch: refined {refined_heads.shape} vs GT {gt_heads.shape}")
                return False
            
            # Check tensor ranges
            if refined_heads.min() < 0 or refined_heads.max() > 1:
                logger.warning(f"Refined heads out of [0,1] range: [{refined_heads.min():.3f}, {refined_heads.max():.3f}]")
            
            if gt_heads.min() < 0 or gt_heads.max() > 1:
                logger.warning(f"GT heads out of [0,1] range: [{gt_heads.min():.3f}, {gt_heads.max():.3f}]")
            
            # Check for NaN/Inf values
            if torch.isnan(refined_heads).any() or torch.isinf(refined_heads).any():
                logger.warning("NaN/Inf detected in refined heads")
                return False
            
            if torch.isnan(gt_heads).any() or torch.isinf(gt_heads).any():
                logger.warning("NaN/Inf detected in GT heads")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            return False
    
    def get_loss_info(self) -> Dict[str, Any]:
        """Get information about the identity loss configuration."""
        return {
            "method": "differentiable_open_clip",
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "loss_weight": self.loss_weight,
            "similarity_target": self.similarity_target,
            "reduction": self.reduction,
            "clip_model_available": self.clip_model is not None,
            "open_clip_available": OPEN_CLIP_AVAILABLE
        }

