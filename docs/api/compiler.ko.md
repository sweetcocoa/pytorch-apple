# 컴파일러 API

## IR Reader

::: npu_compiler.ir_reader
    options:
      members:
        - IRGraph
        - OpNode
        - TensorSpec
        - load_ir

## Constraint Checker

::: npu_compiler.constraint_checker
    options:
      members:
        - check_constraints
        - pad_channels
        - padded_shape_4d

## Graph Optimizer

::: npu_compiler.graph_optimizer
    options:
      members:
        - optimize_graph

## Fusion Patterns

::: npu_compiler.fusion_patterns
    options:
      members:
        - FusedGroup
        - find_fusion_groups

## Code Generator

::: npu_compiler.codegen
    options:
      members:
        - KernelCall
        - BufferAllocation
        - ExecutionPlan
        - generate_execution_plan

## Compiled Program

::: npu_compiler.compiled_program
    options:
      members:
        - CompiledProgram
