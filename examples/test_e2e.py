"""Quick E2E test for Qwen decode with index_copy."""
import sys, time, json
sys.path.insert(0, '.')
import numpy as np
from hf_utils import load_weights_from_hf
import npu_compiler
from npu_runtime.buffer import NPUBuffer
from npu_runtime.device import Device
from npu_runtime.executor import Executor

def p(msg):
    print(msg, flush=True)

p('Step 1: Compile')
pp = npu_compiler.compile('qwen2_prefill_ir.json')
dp = npu_compiler.compile('qwen2_decode_ir.json')
device = Device()

p('Step 2: Load prefill weights')
pw = load_weights_from_hf('Qwen/Qwen2.5-1.5B-Instruct', pp, device)
p(f'  {len(pw)} buffers')

p('Step 3: Load decode weights')
dw = load_weights_from_hf('Qwen/Qwen2.5-1.5B-Instruct', dp, device)
p(f'  {len(dw)} buffers')

p('Step 4: KV mapping')
with open('qwen2_kv_mapping.json') as f:
    kv_map = json.load(f)

p('Step 5: Executors')
pe = Executor(pp, device)
de = Executor(dp, device)

p('Step 6: Tokenize')
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
input_ids = tokenizer.encode('Hello', return_tensors='np')[0]
p(f'  tokens: {input_ids}')

p('Step 7: Prefill')
prefill_seq_len = pp.input_specs[0].shape[1]
actual_len = min(len(input_ids), prefill_seq_len)
padded_ids = np.zeros((1, prefill_seq_len), dtype=np.int64)
padded_ids[0, :actual_len] = input_ids[:actual_len]
causal_mask = np.triu(np.full((1, 1, prefill_seq_len, prefill_seq_len), -65504.0, dtype=np.float16), k=1)
position_ids = np.arange(prefill_seq_len, dtype=np.int64).reshape(1, -1)
cache_position = np.arange(prefill_seq_len, dtype=np.int64)
np_map = {'input_ids': padded_ids, 'attention_mask': causal_mask, 'position_ids': position_ids, 'cache_position': cache_position}
prefill_inputs = {spec.name: NPUBuffer.from_numpy(np_map[spec.name], device, spec=spec) for spec in pp.input_specs}

t0 = time.time()
prefill_outputs = pe.run(inputs=prefill_inputs, weights=pw)
p(f'  Prefill: {time.time()-t0:.3f}s')

logits = prefill_outputs[pp.output_specs[0].name].to_numpy(spec=pp.output_specs[0])
next_token = int(np.argmax(logits[0, actual_len - 1, :]))
p(f'  First token: {next_token} ({tokenizer.decode([next_token])})')

max_cache_len = kv_map['max_cache_len']

p('Step 8: KV transfer (pad to max_cache_len)')
_FIXED_INPUTS = {'input_ids', 'attention_mask', 'position_ids', 'cache_position'}
decode_kv_specs = {s.name: s for s in dp.input_specs if s.name not in _FIXED_INPUTS}
kv_buffers = {}
for layer in kv_map['layers']:
    for pkey, dkey in [(layer['prefill_key_output'], layer['decode_key_input']),
                       (layer['prefill_value_output'], layer['decode_value_input'])]:
        prefill_kv = prefill_outputs[pkey]
        decode_spec = decode_kv_specs[dkey]
        prefill_spec = next(s for s in pp.output_specs if s.name == pkey)
        kv_np = prefill_kv.to_numpy(spec=prefill_spec)
        padded_kv = np.zeros(decode_spec.shape, dtype=kv_np.dtype)
        padded_kv[:, :, :prefill_seq_len, :] = kv_np
        kv_buffers[dkey] = NPUBuffer.from_numpy(padded_kv, device, spec=decode_spec)
p(f'  {len(kv_buffers)} KV buffers')

_NEG_INF = np.float16(-np.inf)
p('Step 9: Decode loop (7 tokens)')
generated = [next_token]
for step in range(7):
    cur_pos = actual_len + step
    token_arr = np.array([[next_token]], dtype=np.int64)
    decode_mask = np.full((1, 1, 1, max_cache_len), _NEG_INF, dtype=np.float16)
    decode_mask[0, 0, 0, :cur_pos + 1] = 0.0
    decode_pos_ids = np.array([[cur_pos]], dtype=np.int64)
    decode_cache_pos = np.array([cur_pos], dtype=np.int64)
    dnm = {'input_ids': token_arr, 'attention_mask': decode_mask, 'position_ids': decode_pos_ids, 'cache_position': decode_cache_pos}

    decode_inputs = {}
    for spec in dp.input_specs:
        if spec.name in dnm:
            decode_inputs[spec.name] = NPUBuffer.from_numpy(dnm[spec.name], device, spec=spec)
        elif spec.name in kv_buffers:
            decode_inputs[spec.name] = kv_buffers[spec.name]

    t0 = time.time()
    decode_outputs = de.run(inputs=decode_inputs, weights=dw)
    dt = time.time() - t0

    dl = decode_outputs[dp.output_specs[0].name].to_numpy(spec=dp.output_specs[0])
    next_token = int(np.argmax(dl[0, -1, :]))
    generated.append(next_token)
    p(f'  Step {step}: token={next_token} ({tokenizer.decode([next_token])}) time={dt:.3f}s')

    if next_token == tokenizer.eos_token_id:
        break

    # Update KV
    for layer in kv_map['layers']:
        kv_buffers[layer['decode_key_input']] = decode_outputs[layer['decode_key_output']]
        kv_buffers[layer['decode_value_input']] = decode_outputs[layer['decode_value_output']]

output = tokenizer.decode(generated)
p(f'\nGenerated: Hello{output}')
p('Done!')
