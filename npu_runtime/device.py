"""Metal device management for NPU simulation."""

from __future__ import annotations

import Metal  # pyobjc-framework-Metal


class Device:
    """Wraps Metal device, command queue, and shader library compilation."""

    def __init__(self):
        self._device = Metal.MTLCreateSystemDefaultDevice()
        if self._device is None:
            raise RuntimeError("No Metal device found")
        self._command_queue = self._device.newCommandQueue()
        self._pipeline_cache: dict[str, Metal.MTLComputePipelineState] = {}
        self._library_cache: dict[tuple, Metal.MTLLibrary] = {}

    @property
    def name(self) -> str:
        return self._device.name()

    @property
    def mtl_device(self):
        return self._device

    @property
    def command_queue(self):
        return self._command_queue

    def compile_metal_file(self, path: str, macros: dict[str, object] | None = None) -> Metal.MTLLibrary:
        """Compile a .metal file and return the library.

        Args:
            path: Path to the .metal source file.
            macros: Optional preprocessor macros (e.g. {"USE_BFLOAT": 1}).
        """
        cache_key = (path, frozenset(macros.items()) if macros else frozenset())
        if cache_key in self._library_cache:
            return self._library_cache[cache_key]

        with open(path) as f:
            source = f.read()

        options = None
        if macros:
            options = Metal.MTLCompileOptions.alloc().init()
            # Convert macro values to NSString for Metal preprocessor
            ns_macros = {k: str(v) for k, v in macros.items()}
            options.setPreprocessorMacros_(ns_macros)

        library, error = self._device.newLibraryWithSource_options_error_(source, options, None)
        if library is None:
            raise RuntimeError(f"Metal file compilation failed ({path}): {error}")
        self._library_cache[cache_key] = library
        return library

    def get_pipeline(self, library: Metal.MTLLibrary, function_name: str):
        """Get a compute pipeline from a library."""
        cache_key = f"{id(library)}:{function_name}"
        if cache_key in self._pipeline_cache:
            return self._pipeline_cache[cache_key]

        function = library.newFunctionWithName_(function_name)
        if function is None:
            raise RuntimeError(f"Function '{function_name}' not found")

        pipeline, error = self._device.newComputePipelineStateWithFunction_error_(function, None)
        if pipeline is None:
            raise RuntimeError(f"Pipeline creation failed: {error}")

        self._pipeline_cache[cache_key] = pipeline
        return pipeline

    def new_command_buffer(self):
        return self._command_queue.commandBuffer()
