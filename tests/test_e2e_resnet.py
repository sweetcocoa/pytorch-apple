"""End-to-end tests: compile IR → load weights → run on Metal → compare with CPU."""

import os
import tempfile

import numpy as np
import numpy.testing as npt
import pytest
import torch
import torch.nn as nn
from torch_ir import extract_ir

import npu_compiler
from npu_runtime.buffer import NPUBuffer
from npu_runtime.executor import Executor
from npu_runtime.weight_loader import load_weights

# ─── Simple ConvNet (matches torch_to_ir example) ───


class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def _extract_ir_json(model: nn.Module, example_input: torch.Tensor, model_name: str) -> str:
    """Extract IR JSON using torch_to_ir and return temp file path."""
    with torch.device("meta"):
        meta_model = type(model)()
    meta_model.eval()

    meta_input = (torch.randn(*example_input.shape, device="meta"),)
    ir = extract_ir(meta_model, meta_input, model_name=model_name)

    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    ir.save(tmp.name)
    return tmp.name


def _model_state_dict_to_numpy(model: nn.Module) -> dict[str, np.ndarray]:
    """Convert model state dict to numpy arrays."""
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}


class TestSimpleConvNetE2E:
    """E2E test with the SimpleConvNet from torch_to_ir examples."""

    def test_prebuilt_ir(self, device):
        """Test using the pre-existing simple_convnet_ir.json."""
        ir_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "torch_to_ir", "simple_convnet_ir.json"
        )
        if not os.path.exists(ir_path):
            pytest.skip("simple_convnet_ir.json not found")

        # Compile
        program = npu_compiler.compile(ir_path)
        assert program.model_name == "SimpleConvNet"

        # Create model and get weights
        torch.manual_seed(42)
        model = SimpleConvNet()
        model.eval()
        state_dict_np = _model_state_dict_to_numpy(model)

        # Load weights to NPU
        weights = load_weights(state_dict_np, program, device)

        # Create executor
        executor = Executor(program, device)

        # Prepare input (read alloc_shape from compiler spec — single source of truth)
        torch.manual_seed(123)
        input_tensor = torch.randn(1, 3, 32, 32)
        input_spec = program.input_specs[0]
        input_buf = {input_spec.name: NPUBuffer.from_numpy(
            input_tensor.numpy(), device, spec=input_spec,
        )}

        # Run on NPU
        outputs = executor.run(input_buf, weights)

        # Run on CPU
        with torch.no_grad():
            cpu_output = model(input_tensor).numpy()

        # Compare
        output_spec = program.output_specs[0]
        npu_output = list(outputs.values())[0].to_numpy(spec=output_spec)
        npt.assert_allclose(npu_output, cpu_output, rtol=5e-2, atol=5e-2)

    def test_extracted_ir(self, device):
        """Test with freshly extracted IR."""
        torch.manual_seed(42)
        model = SimpleConvNet()
        model.eval()

        input_tensor = torch.randn(1, 3, 32, 32)

        # Extract IR
        ir_path = _extract_ir_json(model, input_tensor, "SimpleConvNet")

        try:
            # Compile
            program = npu_compiler.compile(ir_path)

            # Load weights
            state_dict_np = _model_state_dict_to_numpy(model)
            weights = load_weights(state_dict_np, program, device)

            # Execute
            executor = Executor(program, device)
            input_spec = program.input_specs[0]
            input_buf = {input_spec.name: NPUBuffer.from_numpy(
                input_tensor.numpy(), device, spec=input_spec,
            )}
            outputs = executor.run(input_buf, weights)

            # Compare with CPU
            with torch.no_grad():
                cpu_output = model(input_tensor).numpy()

            output_spec = program.output_specs[0]
            npu_output = list(outputs.values())[0].to_numpy(spec=output_spec)
            npt.assert_allclose(npu_output, cpu_output, rtol=5e-2, atol=5e-2)
        finally:
            os.unlink(ir_path)


def _extract_resnet_ir(model_fn, model_name, input_shape=(1, 3, 224, 224)):
    """Extract IR for a ResNet model."""
    with torch.device("meta"):
        meta_model = model_fn(weights=None)
    meta_model.eval()

    meta_input = (torch.randn(*input_shape, device="meta"),)
    ir = extract_ir(meta_model, meta_input, model_name=model_name)

    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    ir.save(tmp.name)
    return tmp.name


class TestResNet18E2E:
    """E2E test with ResNet-18 (requires torchvision)."""

    def _extract_resnet_ir(self, model_fn, model_name, input_shape=(1, 3, 224, 224)):
        """Extract IR for a ResNet model."""
        with torch.device("meta"):
            meta_model = model_fn(weights=None)
        meta_model.eval()

        meta_input = (torch.randn(*input_shape, device="meta"),)
        ir = extract_ir(meta_model, meta_input, model_name=model_name)

        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        ir.save(tmp.name)
        return tmp.name

    def test_resnet18(self, device):
        """Full ResNet-18 E2E: extract IR → compile → load weights → run → compare."""
        import torchvision.models as models

        torch.manual_seed(0)
        model = models.resnet18(weights=None)
        model.eval()

        # Extract IR
        ir_path = self._extract_resnet_ir(models.resnet18, "ResNet18")

        try:
            # Compile
            program = npu_compiler.compile(ir_path)

            # Load weights
            state_dict_np = _model_state_dict_to_numpy(model)
            weights = load_weights(state_dict_np, program, device)

            # Execute
            executor = Executor(program, device)
            input_tensor = torch.randn(1, 3, 224, 224)
            input_spec = program.input_specs[0]
            input_buf = {input_spec.name: NPUBuffer.from_numpy(
                input_tensor.numpy(), device, spec=input_spec,
            )}

            outputs = executor.run(input_buf, weights)

            # Compare with CPU
            with torch.no_grad():
                cpu_output = model(input_tensor).numpy()

            output_spec = program.output_specs[0]
            npu_output = list(outputs.values())[0].to_numpy(spec=output_spec)

            # ResNet-18 E2E with FP16: allow larger tolerance
            npt.assert_allclose(npu_output, cpu_output, rtol=1e-1, atol=1e-1)

            # Check argmax matches (classification agreement)
            assert np.argmax(npu_output) == np.argmax(cpu_output), (
                f"Classification mismatch: NPU={np.argmax(npu_output)}, CPU={np.argmax(cpu_output)}"
            )
        finally:
            os.unlink(ir_path)


class TestResNet50E2E:
    """E2E test with ResNet-50 (bottleneck blocks)."""

    def test_resnet50(self, device):
        import torchvision.models as models

        torch.manual_seed(0)
        model = models.resnet50(weights=None)
        model.eval()

        ir_path = _extract_resnet_ir(models.resnet50, "ResNet50")

        try:
            program = npu_compiler.compile(ir_path)

            state_dict_np = _model_state_dict_to_numpy(model)
            weights = load_weights(state_dict_np, program, device)

            executor = Executor(program, device)
            input_tensor = torch.randn(1, 3, 224, 224)
            input_spec = program.input_specs[0]
            input_buf = {input_spec.name: NPUBuffer.from_numpy(
                input_tensor.numpy(), device, spec=input_spec,
            )}

            outputs = executor.run(input_buf, weights)

            with torch.no_grad():
                cpu_output = model(input_tensor).numpy()

            output_spec = program.output_specs[0]
            npu_output = list(outputs.values())[0].to_numpy(spec=output_spec)

            npt.assert_allclose(npu_output, cpu_output, rtol=2e-1, atol=2e-1)

            assert np.argmax(npu_output) == np.argmax(cpu_output), (
                f"Classification mismatch: NPU={np.argmax(npu_output)}, CPU={np.argmax(cpu_output)}"
            )
        finally:
            os.unlink(ir_path)
