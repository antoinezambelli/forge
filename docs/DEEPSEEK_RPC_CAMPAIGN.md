# DeepSeek V4 RPC campaign

Forge can own the proven two-rig DeepSeek-V4-Flash-0731 launch directly from
`batch_eval`; the old external smoke launcher is not part of this path.

## Select the reasoning effort

The sole effort control is
`MODEL_SAMPLING_DEFAULTS["DeepSeek-V4-Flash-0731-UD-Q4_K_XL"]` in
`src/forge/clients/sampling_defaults.py`. Its
`chat_template_kwargs.reasoning_effort` must be one of `low`, `high`, or
`max`. Change only that value between isolated effort campaigns; the DeepSeek
batch config reads it into `reasoning_level` so result provenance and resume
identity match the effort actually run. The dry-run prints the selected value
for confirmation. The config has no sampling override.

## Supply the machine-local topology

Copy [`examples/deepseek-v4-rpc-topology.json`](examples/deepseek-v4-rpc-topology.json)
outside the repository and replace the worker host, SSH target, executable
paths, and log directory. The example preserves the tested `Vulkan0,RPC0`
layer split, 1:1 tensor split, worker tensor cache, RADV environment, and a
30-minute startup/readiness timeout. The file contains no reasoning-effort
setting.

The immutable model recipe supplies batch 2048, microbatch 128, Q8 K/V,
`--fit off`, full offload, no mmap, flash attention, reasoning budget 32768,
automatic reasoning formatting, and no-prefill-assistant. `ServerManager`
owns `-ngl 999` and native `--jinja`; the invocation owns the 256K context.

## Dry-run

Set the campaign-time choices explicitly, then inspect the complete normalized
worker/coordinator commands and the 26 selected scenario names:

```bash
cd /absolute/path/to/forge

TOPOLOGY=/absolute/path/to/deepseek-v4-rpc-topology.json
MODELS_DIR=/absolute/path/to/models
OUTPUT=/absolute/path/to/eval_results.jsonl
ABLATION=reforged
REPLAY=none
GENERATION=3
REQUEST_TIMEOUT=7200
RUN_TIMEOUT=7200

.venv/bin/python -u -m tests.eval.batch_eval \
  --config deepseek-v4-rpc \
  --rpc-topology "$TOPOLOGY" \
  --tags plumbing model_quality advanced_reasoning \
  --runs 50 \
  --budget-mode manual \
  --num-ctx 262144 \
  --request-timeout "$REQUEST_TIMEOUT" \
  --run-timeout "$RUN_TIMEOUT" \
  --ablation "$ABLATION" \
  --reasoning-replay "$REPLAY" \
  --generation "$GENERATION" \
  --models-dir "$MODELS_DIR" \
  --output "$OUTPUT" \
  --dry-run
```

With the first GGUF shard present, this reports one config, 26 scenarios, and
1,300 maximum rows at N=50. Dry-run reads existing result/static artifact
state but does not start SSH/RPC, load the model, build an inference client,
send a request, or write the output.

## Live campaign

After reviewing the dry-run, execute the same command without `--dry-run`.
Forge starts the worker first, starts the coordinator, keeps both loaded across
the selected scenarios and repetitions, and stops both at the configuration
boundary. Ablation, replay, generation, output filename, and request/run
timeouts remain deliberate campaign choices in the shell variables above.
The 256K context remains `--budget-mode manual --num-ctx 262144` rather than a
server-recipe constant.

The deterministic ticket gate does not launch this real two-rig campaign. The
human live gate should confirm the ports are initially clear, the 50/50 Q8
load, basic inference, managed `data_gap_recovery_extended_stateful` N=5,
restart replay, and a clean two-rig stop.
