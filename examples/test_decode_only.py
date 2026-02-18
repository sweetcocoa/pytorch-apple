"""Test decode-only path: verifies index_copy KV cache update and multi-step decode.

Skips the slow prefill by providing random KV cache values.
Focuses on verifying:
1. Decode compilation and execution works
2. index_copy preserves KV cache shape across steps
3. KV mapping correctly connects outputs to inputs
"""
import sys, time, json
sys.path.insert(0, '.')
import numpy as np
import npu_compiler
from npu_runtime.buffer import NPUBuffer
from npu_runtime.device import Device
from npu_runtime.executor import Executor

def p(msg):
    print(msg, flush=True)

p('=== Decode-only test (index_copy + KV mapping) ===')

p('Step 1: Compile decode')
dp = npu_compiler.compile('qwen2_decode_ir.json')
p(f'  {len(dp.kernel_calls)} calls, {len(dp.buffer_allocations)} allocs')

# Count kernel types
from collections import Counter
ktype = Counter(c.kernel_name for c in dp.kernel_calls)
p(f'  index_copy_kernel: {ktype.get("index_copy_kernel", 0)}')
p(f'  slice_kernel: {ktype.get("slice_kernel", 0)}')

p('Step 2: Load weights')
from hf_utils import load_weights_from_hf
device = Device()
dw = load_weights_from_hf('Qwen/Qwen2.5-1.5B-Instruct', dp, device)
p(f'  {len(dw)} weight buffers')

p('Step 3: KV mapping')
with open('qwen2_kv_mapping.json') as f:
    kv_map = json.load(f)
p(f'  {kv_map["num_layers"]} layers')

p('Step 4: Create executor')
de = Executor(dp, device)

p('Step 5: Prepare dummy KV cache')
# Find KV input specs
fixed_inputs = {'input_ids', 'attention_mask', 'position_ids', 'cache_position'}
kv_input_specs = [s for s in dp.input_specs if s.name not in fixed_inputs]
max_cache_len = kv_map['max_cache_len']
p(f'  KV input specs: {len(kv_input_specs)}')
p(f'  First KV shape: {kv_input_specs[0].shape}')
p(f'  max_cache_len: {max_cache_len}')

# Create random KV cache (simulating prefill output)
kv_buffers = {}
for spec in kv_input_specs:
    data = np.random.randn(*spec.shape).astype(np.float16) * 0.01
    kv_buffers[spec.name] = NPUBuffer.from_numpy(data, device, spec=spec)

p('Step 6: Multi-step decode (3 steps)')
_NEG_INF = np.float16(-np.inf)

for step in range(3):
    cur_pos = 1 + step  # start from position 1 (after 1 prefill token)
    token_id = 9707  # "Hello" token

    token_arr = np.array([[token_id]], dtype=np.int64)
    decode_mask = np.full((1, 1, 1, max_cache_len), _NEG_INF, dtype=np.float16)
    decode_mask[0, 0, 0, :cur_pos + 1] = 0.0
    decode_pos_ids = np.array([[cur_pos]], dtype=np.int64)
    decode_cache_pos = np.array([cur_pos], dtype=np.int64)

    dnm = {
        'input_ids': token_arr,
        'attention_mask': decode_mask,
        'position_ids': decode_pos_ids,
        'cache_position': decode_cache_pos,
    }

    decode_inputs = {}
    for spec in dp.input_specs:
        if spec.name in dnm:
            decode_inputs[spec.name] = NPUBuffer.from_numpy(dnm[spec.name], device, spec=spec)
        elif spec.name in kv_buffers:
            decode_inputs[spec.name] = kv_buffers[spec.name]

    missing = [s.name for s in dp.input_specs if s.name not in decode_inputs]
    if missing:
        p(f'  WARNING: missing inputs: {missing}')

    t0 = time.time()
    decode_outputs = de.run(inputs=decode_inputs, weights=dw)
    dt = time.time() - t0

    # Check output shapes
    logits_spec = dp.output_specs[0]
    logits_buf = decode_outputs[logits_spec.name]
    logits = logits_buf.to_numpy(spec=logits_spec)
    next_token = int(np.argmax(logits[0, -1, :]))

    # Check KV output shapes match input shapes
    kv_out_ok = True
    for layer in kv_map['layers']:
        k_out = decode_outputs.get(layer['decode_key_output'])
        v_out = decode_outputs.get(layer['decode_value_output'])
        if k_out is None or v_out is None:
            p(f'  ERROR: missing KV output for layer {layer["layer"]}')
            kv_out_ok = False
            break
        # Check shape matches input
        k_in = kv_buffers[layer['decode_key_input']]
        if k_out.shape != k_in.shape:
            p(f'  ERROR: shape mismatch layer {layer["layer"]}: '
              f'input={k_in.shape}, output={k_out.shape}')
            kv_out_ok = False

    p(f'  Step {step}: token={next_token}, time={dt:.3f}s, '
      f'logits_shape={logits.shape}, KV_shape_ok={kv_out_ok}')

    if not kv_out_ok:
        break

    # Update KV for next step (using mapping)
    for layer in kv_map['layers']:
        kv_buffers[layer['decode_key_input']] = decode_outputs[layer['decode_key_output']]
        kv_buffers[layer['decode_value_input']] = decode_outputs[layer['decode_value_output']]

p('\n=== Done! ===')
