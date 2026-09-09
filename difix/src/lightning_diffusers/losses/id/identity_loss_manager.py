"""
Identity Loss Manager for Multi-Loss Architecture

This module provides a unified manager for multiple identity loss types
with weight-based activation and automatic resolution handling.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Any
from loguru import logger

try:
    from .clip_loss import CLIPIdentityLoss
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logger.warning("CLIP not available - CLIP identity loss will be disabled")

try:
    from .arcface_loss import ArcFaceIDLoss
    ARCFACE_AVAILABLE = True
except ImportError:
    ARCFACE_AVAILABLE = False
    logger.warning("ArcFace ID loss not available - ArcFace identity loss will be disabled")

try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    TORCHMETRICS_LPIPS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_LPIPS_AVAILABLE = False
    logger.warning("Torchmetrics LPIPS not available - LPIPS identity loss will be disabled")


class IdentityLossManager(torch.nn.Module):
    """
    Unified manager for multiple identity loss types with weight-based activation.

    Supports CLIP, ArcFace, L1 pixel, and LPIPS pixel identity losses.
    Losses are only initialized and computed if their corresponding weights are > 0.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        
        # Extract loss weights
        self.clip_weight = config.get("clip_identity_loss_weight", 0.0)
        self.arcface_weight = config.get("arcface_identity_loss_weight", 0.0)
        self.l1_weight = config.get("l1_identity_loss_weight", 0.0)
        self.lpips_weight = config.get("lpips_identity_loss_weight", 0.0)

        # Initialize active losses based on weights
        self.clip_loss = None
        self.arcface_loss = None
        self.lpips_fn = None
        
        if self.clip_weight > 0:
            self._initialize_clip_loss()
            
        if self.arcface_weight > 0:
            self._initialize_arcface_loss()

        if self.lpips_weight > 0:
            self._initialize_lpips_loss()

        # Log active losses
        active_losses = []
        if self.clip_loss is not None:
            active_losses.append(f"CLIP (weight: {self.clip_weight})")
        if self.arcface_loss is not None:
            active_losses.append(f"ArcFace (weight: {self.arcface_weight})")
        if self.l1_weight > 0:
            active_losses.append(f"L1 (weight: {self.l1_weight})")
        if self.lpips_fn is not None:
            active_losses.append(f"LPIPS (weight: {self.lpips_weight})")

        if active_losses:
            logger.info(f"Identity loss manager initialized with: {', '.join(active_losses)}")
        else:
            logger.warning("No identity losses active (all weights are 0)")

    def _initialize_clip_loss(self) -> None:
        """
        Create CLIPIdentityLoss from configuration dictionary.
        
        Args:
            config: Configuration dictionary with identity training settings
            
        Returns:
            CLIPIdentityLoss instance if enabled, None otherwise
        """
        if not CLIP_AVAILABLE:
            logger.warning("CLIP loss weight > 0 but CLIP not available - skipping")
            return
        
        
        if not self.config.get("enabled", False):
            logger.info("Identity loss disabled in configuration")
            return
        
        try:
            self.clip_loss = CLIPIdentityLoss(
                model_name_or_path=self.config.get("clip_model_name_or_path", "ViT-B-32"),
                pretrained=self.config.get("clip_pretrained", "openai"),
                loss_weight=self.config.get("clip_identity_loss_weight", 0.1),
                similarity_target=self.config.get("clip_similarity_target", 1.0),
                reduction=self.config.get("clip_reduction", "mean"),
                device=self.device
            )
            
        except Exception as e:
            logger.error(f"Failed to create identity loss: {e}")
            return

    def _initialize_arcface_loss(self) -> None:
        """Initialize ArcFace identity loss if available and configured."""
        if not ARCFACE_AVAILABLE:
            logger.warning("ArcFace loss weight > 0 but ArcFace not available - skipping")
            return
        
        try:
            arcface_model_path = self.config.get("arcface_model_path", "pretrained/model_ir_se50.pth")
            self.arcface_loss = ArcFaceIDLoss(ir_se50_path=arcface_model_path).to(self.device)
            self.arcface_loss.eval()
            logger.info(f"ArcFace identity loss initialized successfully with model: {arcface_model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ArcFace identity loss: {e}")
            self.arcface_loss = None

    def _initialize_lpips_loss(self) -> None:
        """Initialize LPIPS loss using torchmetrics if available and configured."""
        if not TORCHMETRICS_LPIPS_AVAILABLE:
            logger.warning("LPIPS loss weight > 0 but torchmetrics LPIPS not available - skipping")
            return

        try:
            self.lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
            self.lpips_fn = self.lpips_fn.to(self.device)
            self.lpips_fn.eval()  # Set to eval mode but gradients still flow
            logger.info(f"LPIPS identity loss initialized successfully with VGG network")
        except Exception as e:
            logger.error(f"Failed to initialize LPIPS identity loss: {e}")
            self.lpips_fn = None
    
    def compute_identity_loss(
        self, 
        refined_heads: torch.Tensor, 
        gt_heads: torch.Tensor, 
        success_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted combination of all active identity losses.
        
        Args:
            refined_heads: Refined head crops (B, C, 224, 224)
            gt_heads: Ground truth head crops (B, C, 224, 224)  
            success_mask: Boolean mask for valid samples (B,)
            
        Returns:
            Dictionary with loss components:
            - identity_loss: Total weighted identity loss
            - clip_loss: CLIP identity loss (if active)
            - arcface_loss: ArcFace identity loss (if active)
            - clip_similarity: CLIP similarity score (if active)
            - l1_loss: L1 identity loss (if active)
            - lpips_loss: LPIPS identity loss (if active)
            - l1_error_map: L1 error map for visualization (if L1 active)
            - valid_samples: Number of valid samples
        """
        loss_dict = {
            "identity_loss": torch.zeros(1, device=self.device, requires_grad=True),
            "valid_samples": torch.tensor(0, device=self.device)
        }
        
        if refined_heads is None or gt_heads is None:
            logger.warning("Missing head crops for identity loss computation")
            return loss_dict
        
        # Apply success mask if provided
        if success_mask is not None:
            valid_indices = success_mask.bool()
            if valid_indices.sum() == 0:
                logger.debug("No valid samples for identity loss computation")
                return loss_dict
            
            refined_heads = refined_heads[valid_indices]
            gt_heads = gt_heads[valid_indices]
            loss_dict["valid_samples"] = valid_indices.sum()
        else:
            loss_dict["valid_samples"] = torch.tensor(refined_heads.shape[0], device=self.device)
        
        total_loss = torch.zeros(1, device=self.device, requires_grad=True)
        
        # Compute CLIP identity loss
        if self.clip_loss is not None and self.clip_weight > 0:
            try:
                # CLIP expects 224x224 (current head crop size)
                weighted_clip_loss = self.clip_loss.compute_identity_loss(refined_heads, gt_heads, success_mask)["clip_loss"]

                total_loss = total_loss + weighted_clip_loss
                
                loss_dict["clip_loss"] = weighted_clip_loss
                
            except Exception as e:
                logger.error(f"Error computing CLIP identity loss: {e}")
        
        # Compute ArcFace identity loss
        if self.arcface_loss is not None and self.arcface_weight > 0:
            try:
                # 0, 1 to 1, -1
                gt_heads_in = gt_heads * 2 - 1
                refined_heads_in = refined_heads * 2 - 1
                arcface_loss = self.arcface_loss(gt_heads_in, refined_heads_in)
                weighted_arcface_loss = self.arcface_weight * arcface_loss
                total_loss = total_loss + weighted_arcface_loss
                
                loss_dict["arcface_loss"] = weighted_arcface_loss
                
            except Exception as e:
                logger.error(f"Error computing ArcFace identity loss: {e}")

        # Compute L1 identity loss
        if self.l1_weight > 0:
            try:
                # Compute L1 loss with no reduction to get per-pixel errors
                l1_error_map = F.l1_loss(refined_heads, gt_heads, reduction='none')  # (B, C, H, W)

                # Use mean of error map as loss value (more efficient than separate computation)
                l1_loss = l1_error_map.mean()
                weighted_l1_loss = self.l1_weight * l1_loss
                total_loss = total_loss + weighted_l1_loss

                loss_dict["l1_loss"] = weighted_l1_loss
                loss_dict["l1_error_map"] = l1_error_map  # Store full error map for visualization

            except Exception as e:
                logger.error(f"Error computing L1 identity loss: {e}")

        # Compute LPIPS identity loss
        if self.lpips_fn is not None and self.lpips_weight > 0:
            try:
                # Convert [0,1] to [-1,1] range as required by torchmetrics LPIPS
                refined_lpips = (refined_heads * 2) - 1
                gt_lpips = (gt_heads * 2) - 1

                lpips_loss = self.lpips_fn(refined_lpips, gt_lpips)
                weighted_lpips_loss = self.lpips_weight * lpips_loss
                total_loss = total_loss + weighted_lpips_loss

                loss_dict["lpips_loss"] = weighted_lpips_loss

            except Exception as e:
                logger.error(f"Error computing LPIPS identity loss: {e}")

        loss_dict["identity_loss"] = total_loss
        return loss_dict
    
    def has_active_losses(self) -> bool:
        """Check if any identity losses are active."""
        return (self.clip_loss is not None and self.clip_weight > 0) or \
               (self.arcface_loss is not None and self.arcface_weight > 0) or \
               (self.l1_weight > 0) or \
               (self.lpips_fn is not None and self.lpips_weight > 0)

    def extract_error_map_data(self, loss_dict: Dict[str, torch.Tensor], max_samples: int = 4) -> Optional[list]:
        """
        Extract L1 error map data for wandb logging with proper heat map visualization.

        Args:
            loss_dict: Loss dictionary from compute_identity_loss()
            max_samples: Maximum number of samples to process for visualization

        Returns:
            List of colored heatmap numpy arrays for wandb, or None if no error map available
        """
        if "l1_error_map" not in loss_dict:
            return None

        try:
            import cv2
            import numpy as np

            error_map = loss_dict["l1_error_map"]  # (B, C, H, W)

            # Limit to max_samples for performance
            batch_size = min(error_map.shape[0], max_samples)
            error_map = error_map[:batch_size]

            # Convert to grayscale by averaging channels
            error_grayscale = error_map.mean(dim=1)  # (B, H, W)

            # Convert to colored heatmaps using cv2
            colored_heatmaps = []
            for i in range(batch_size):
                # Convert to numpy and normalize to [0, 255]
                error_np = error_grayscale[i].detach().cpu().numpy()
                error_normalized = (error_np * 255).astype(np.uint8)

                # Apply colormap (COLORMAP_JET gives good heat map visualization)
                colored_heatmap = cv2.applyColorMap(error_normalized, cv2.COLORMAP_JET)
                # Convert BGR to RGB for proper wandb display
                colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

                colored_heatmaps.append(colored_heatmap)

            return colored_heatmaps

        except Exception as e:
            logger.warning(f"Failed to extract error map data: {e}")
            return None