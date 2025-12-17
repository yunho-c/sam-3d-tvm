"""Minimal TVM import test to validate torch.export → Relax workflow."""

import numpy as np
import torch
import tvm
from tvm.relax.frontend.torch import from_exported_program


def test_minimal_tvm_import():
    """Test basic PyTorch to TVM Relax conversion."""
    # Simple MLP model
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(64, 128)
            self.linear2 = torch.nn.Linear(128, 64)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.nn.functional.gelu(self.linear1(x))
            return self.linear2(x)

    # Create model and sample input
    model = SimpleModel().eval()
    sample_input = torch.randn(1, 64)

    # 1. Export from PyTorch
    exported = torch.export.export(model, (sample_input,))

    # 2. Convert to TVM Relax
    mod = from_exported_program(exported, keep_params_as_input=True)
    assert mod is not None, "Failed to convert to TVM IR"

    # 3. Build for LLVM target
    ex = tvm.relax.build(mod, target="llvm")
    vm = tvm.relax.VirtualMachine(ex, tvm.cpu())

    # 4. Run inference
    # Get params from exported program
    params = [p.detach().numpy() for p in exported.state_dict.values()]
    tvm_input = tvm.runtime.tensor(sample_input.numpy())
    tvm_params = [tvm.runtime.tensor(p) for p in params]

    tvm_output = vm["main"](tvm_input, *tvm_params)

    # 5. Compare with PyTorch output
    with torch.no_grad():
        torch_output = model(sample_input).numpy()

    # TVM returns a tuple/Array, extract first element
    if hasattr(tvm_output, "numpy"):
        tvm_output_np = tvm_output.numpy()
    else:
        tvm_output_np = tvm_output[0].numpy()
    max_error = np.abs(torch_output - tvm_output_np).max()

    assert max_error < 1e-4, f"Max error {max_error} exceeds threshold"
    print(f"✓ TVM import test passed! Max error: {max_error:.2e}")


if __name__ == "__main__":
    test_minimal_tvm_import()
