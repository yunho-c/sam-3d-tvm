"""Port DINOv2 encoder from PyTorch to TVM Relax.

This script exports the DINOv2 ViT-B/14 encoder used by SAM-3D-Objects to TVM.

Usage:
    pixi run python src/sam_3d_tvm/scripts/port_dino.py [--validate]
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import tvm
from tvm import relax


class DinoEncoder(nn.Module):
    """Standalone DINOv2 encoder for TVM export.
    
    This is a simplified version of the SAM-3D Dino class that's more
    amenable to torch.export (no dynamic hub loading).
    """

    def __init__(
        self,
        dino_model: nn.Module,
        input_size: int = 224,
        normalize_images: bool = True,
    ):
        super().__init__()
        self.backbone = dino_model
        self.input_size = input_size
        self.normalize_images = normalize_images
        
        # ImageNet normalization
        self.register_buffer(
            "mean", 
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), 
            persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DINOv2.
        
        Args:
            x: Input image tensor of shape (B, C, H, W), values in [0, 1]
            
        Returns:
            Tokens of shape (B, N+1, D) where N is number of patches, D is embed_dim
        """
        # Resize to expected input size
        x = F.interpolate(
            x,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        
        # Handle grayscale
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Normalize
        if self.normalize_images:
            x = (x - self.mean) / self.std
        
        # Forward through DINOv2
        output = self.backbone.forward_features(x)
        
        # Concatenate CLS token with patch tokens
        tokens = torch.cat(
            [
                output["x_norm_clstoken"].unsqueeze(1),
                output["x_norm_patchtokens"],
            ],
            dim=1,
        )
        
        return tokens


def load_dinov2_model(model_name: str = "dinov2_vitb14") -> nn.Module:
    """Load DINOv2 model from torch.hub."""
    print(f"Loading DINOv2 model: {model_name}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            pretrained=True,
            verbose=False,
        )
    model.eval()
    return model


def export_to_tvm(model: nn.Module, sample_input: torch.Tensor) -> tvm.IRModule:
    """Export PyTorch model to TVM Relax IR."""
    from tvm.relax.frontend.torch import from_exported_program
    
    print("Exporting with torch.export...")
    with torch.no_grad():
        exported = torch.export.export(model.eval(), (sample_input,))
    
    print("Converting to TVM Relax IR...")
    mod = from_exported_program(exported, keep_params_as_input=True)
    
    return mod, exported


def build_tvm_module(mod: tvm.IRModule, target: str = "llvm") -> relax.VirtualMachine:
    """Build TVM module for execution."""
    print(f"Building TVM module for target: {target}")
    ex = relax.build(mod, target=target)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    return vm


def validate_output(
    pytorch_model: nn.Module,
    tvm_vm: relax.VirtualMachine,
    exported_program,
    sample_input: torch.Tensor,
    atol: float = 1e-3,
) -> tuple[bool, float]:
    """Validate TVM output against PyTorch."""
    print("Validating outputs...")
    
    # PyTorch inference
    with torch.no_grad():
        pytorch_out = pytorch_model(sample_input).numpy()
    
    # TVM inference
    tvm_input = tvm.runtime.tensor(sample_input.numpy())
    params = [tvm.runtime.tensor(p.detach().numpy()) for p in exported_program.state_dict.values()]
    
    tvm_out = tvm_vm["main"](tvm_input, *params)
    
    # Handle tuple return
    if hasattr(tvm_out, "numpy"):
        tvm_out_np = tvm_out.numpy()
    else:
        tvm_out_np = tvm_out[0].numpy()
    
    # Compare
    max_error = np.abs(pytorch_out - tvm_out_np).max()
    mean_error = np.abs(pytorch_out - tvm_out_np).mean()
    
    print(f"  Max error:  {max_error:.2e}")
    print(f"  Mean error: {mean_error:.2e}")
    print(f"  Shape match: {pytorch_out.shape == tvm_out_np.shape}")
    
    passed = max_error < atol
    return passed, max_error


def main():
    parser = argparse.ArgumentParser(description="Port DINOv2 to TVM")
    parser.add_argument("--validate", action="store_true", help="Run validation")
    parser.add_argument("--save-ir", type=str, help="Save TVM IR to file")
    parser.add_argument("--model", default="dinov2_vitb14", help="DINOv2 model name")
    parser.add_argument("--input-size", type=int, default=224, help="Input image size")
    args = parser.parse_args()
    
    # Load DINOv2
    dino_backbone = load_dinov2_model(args.model)
    
    # Create encoder wrapper
    encoder = DinoEncoder(
        dino_model=dino_backbone,
        input_size=args.input_size,
        normalize_images=True,
    ).eval()
    
    print(f"Encoder embed_dim: {dino_backbone.embed_dim}")
    print(f"Encoder num_tokens: {(args.input_size // 14) ** 2 + 1}")  # patch_size=14 for DINOv2
    
    # Create sample input
    sample_input = torch.randn(1, 3, args.input_size, args.input_size)
    
    # Export to TVM
    mod, exported = export_to_tvm(encoder, sample_input)
    print("✓ Export to TVM successful!")
    
    # Save IR if requested
    if args.save_ir:
        ir_path = Path(args.save_ir)
        ir_path.write_text(mod.astext())
        print(f"Saved TVM IR to: {ir_path}")
    
    # Validate if requested
    if args.validate:
        vm = build_tvm_module(mod)
        passed, max_error = validate_output(encoder, vm, exported, sample_input)
        
        if passed:
            print(f"✓ Validation PASSED (max error: {max_error:.2e})")
        else:
            print(f"✗ Validation FAILED (max error: {max_error:.2e})")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
