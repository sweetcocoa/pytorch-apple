"""CompiledProgram: serializable execution plan (.npubin format)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from npu_compiler.codegen import BufferAllocation, ExecutionPlan, KernelCall
from npu_compiler.graph_optimizer import WeightTransformRecipe
from npu_compiler.ir_reader import TensorSpec


@dataclass
class CompiledProgram:
    """A compiled NPU program ready for execution."""
    model_name: str
    execution_plan: ExecutionPlan
    weight_recipes: list[WeightTransformRecipe]

    @property
    def input_specs(self) -> list[TensorSpec]:
        return self.execution_plan.input_specs

    @property
    def output_specs(self) -> list[TensorSpec]:
        return self.execution_plan.output_specs

    @property
    def weight_specs(self) -> list[TensorSpec]:
        return self.execution_plan.weight_specs

    @property
    def weight_name_mapping(self) -> dict[str, str]:
        return self.execution_plan.weight_name_mapping

    @property
    def kernel_calls(self) -> list[KernelCall]:
        return self.execution_plan.kernel_calls

    @property
    def buffer_allocations(self) -> list[BufferAllocation]:
        return self.execution_plan.buffer_allocations

    @property
    def compute_dtype(self) -> str:
        return self.execution_plan.compute_dtype

    def save(self, path: str):
        """Serialize to .npubin (JSON-based for now)."""
        data = {
            "format": "npubin",
            "version": 1,
            "model_name": self.model_name,
            "execution_plan": _plan_to_dict(self.execution_plan),
            "weight_recipes": [asdict(r) for r in self.weight_recipes],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @staticmethod
    def load(path: str) -> CompiledProgram:
        """Deserialize from .npubin."""
        data = json.loads(Path(path).read_text())
        assert data["format"] == "npubin"
        assert data["version"] == 1

        plan = _dict_to_plan(data["execution_plan"])
        recipes = [
            WeightTransformRecipe(**r) for r in data["weight_recipes"]
        ]
        return CompiledProgram(
            model_name=data["model_name"],
            execution_plan=plan,
            weight_recipes=recipes,
        )


def _plan_to_dict(plan: ExecutionPlan) -> dict:
    return {
        "kernel_calls": [asdict(k) for k in plan.kernel_calls],
        "buffer_allocations": [asdict(b) for b in plan.buffer_allocations],
        "input_specs": [_tensor_to_dict(s) for s in plan.input_specs],
        "output_specs": [_tensor_to_dict(s) for s in plan.output_specs],
        "weight_specs": [_tensor_to_dict(s) for s in plan.weight_specs],
        "weight_name_mapping": plan.weight_name_mapping,
        "compute_dtype": plan.compute_dtype,
    }


def _tensor_to_dict(t: TensorSpec) -> dict:
    d = {"name": t.name, "shape": t.shape, "dtype": t.dtype,
         "producer_node": t.producer_node, "producer_output_idx": t.producer_output_idx}
    if t.alloc_shape is not None:
        d["alloc_shape"] = t.alloc_shape
    if t.transform_steps is not None:
        d["transform_steps"] = t.transform_steps
    return d


def _migrate_kernel_call(k: dict) -> dict:
    """Backward-compatible migration: rename legacy 'metal_file' → 'kernel_source'."""
    if "metal_file" in k and "kernel_source" not in k:
        k = dict(k)
        k["kernel_source"] = k.pop("metal_file")
    return k


def _dict_to_plan(d: dict) -> ExecutionPlan:
    return ExecutionPlan(
        kernel_calls=[KernelCall(**_migrate_kernel_call(k)) for k in d["kernel_calls"]],
        buffer_allocations=[BufferAllocation(**b) for b in d["buffer_allocations"]],
        input_specs=[TensorSpec(**s) for s in d["input_specs"]],
        output_specs=[TensorSpec(**s) for s in d["output_specs"]],
        weight_specs=[TensorSpec(**s) for s in d["weight_specs"]],
        weight_name_mapping=d["weight_name_mapping"],
        compute_dtype=d["compute_dtype"],
    )
