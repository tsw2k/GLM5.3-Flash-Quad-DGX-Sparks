# GLM-5.3-Flash on four DGX Sparks — what it took to actually keep it up

Serving [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (320B MoE)
at TP=4 across four **ASUS Ascent GX10** (NVIDIA GB10, DGX Spark class), on a 100G RoCEv2
fabric.

This is a field report, not another quickstart. The recipe itself is
[tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark)
and it is excellent — everything here is what we hit *around* it: three failed launches,
a fabric that overheated itself, and a silent NCCL hang that looks exactly like hard work.

## Results

Fabric: Arista 7060CX-32S, **100G**, single rail, passive DAC. Model:
`LibertAIDAI/GLM-5.3-Flash-NVFP4`, TP=4, fp8 KV-cache, MTP-4 speculative decoding,
`--enforce-eager`, `--max-model-len 1048576`, KV 24 GiB/rank.

| Content regime | TTFT | Decode |
|---|---|---|
| structured (counting, lists) | 0.202 s | **64.0 tok/s** |
| code generation | 0.198 s | **46.9 tok/s** |
| freeform prose | 0.195 s | **33.4 tok/s** |

| Concurrency | Aggregate | Per stream |
|---|---|---|
| 1 | 44.2 tok/s | 44.2 |
| 2 | 79.7 tok/s | 39.8 |
| 8 | **114.0 tok/s** | 14.3 |

KV pool 3,774,873 tokens (3.6× concurrent full-1M-context requests). Speculative decode
acceptance 87–95.6 %.

**These match the reference build's numbers, which were measured on a 200G fabric.** On
this model, decode is not fabric-bound — worth knowing before anyone buys faster
switching to make tokens come out quicker. Where more fabric would help is prefill of
long contexts and heavy multi-tenant load.

## The five things that cost us a day

### 1. `NCCL_IB_GID_INDEX` is per node, and getting it wrong hangs silently

The RoCEv2/IPv4 GID does **not** live at the same index on every node. Ours:

| Node | rail A | rail B |
|---|---|---|
| spark-01 | 3 | 4 |
| spark-02 | 3 | 3 |
| spark-03 | 3 | 4 |
| spark-04 | **4** | 4 |

An index becomes a hole when an address is removed and re-added — the new GID takes the
next free slot instead of the old one. Check it, never assume:

```bash
for i in 0 1 2 3 4 5; do
  echo -n "idx$i: "; cat /sys/class/infiniband/rocep1s0f0/ports/1/gid_attrs/types/$i 2>/dev/null
  cat /sys/class/infiniband/rocep1s0f0/ports/1/gids/$i 2>/dev/null
done
```

Two failure modes, and the second is the nasty one:

- **wrong explicit index** → `ibv_modify_qp failed ... local GID ::` — a clean, obvious error
- **no index set at all (auto-selection)** → the collective **hangs**. Twenty minutes of
  no log output, GPUs pinned at 96 % utilisation.

**96 % utilisation at 22–26 W is not compute — it is a spin-wait.** Real weight loading on
a GB10 draws far more. Watching `nvidia-smi --query-gpu=utilization.gpu,power.draw` beats
watching the log: high utilisation with low power means a hung collective, every time.

Fix: set it per rank ([`launch-glm53-tp4.sh`](launch-glm53-tp4.sh)).

**Update, after a day of reboots: a per-rank pin goes stale too.** The table above was
true when we wrote it. We rebooted the fleet a few times for unrelated reasons, and
spark-04's RoCEv2 GID moved from index 4 back to 3 — so the pin that had been correct
became a pin to an empty slot. The symptom is not the clean `local GID ::` error you get
from a wrong index at cabling time; it is `NCCL error: unhandled system error` at init,
on rank 3 only, with the head reporting nothing more useful than "WorkerProc
initialization failed". Two relaunches and a full fleet reboot did not help, because
none of them changed the pin.

The launcher now reads the index out of sysfs before it starts the container, and keeps
the pinned value only as a fallback. It is six lines, and it prints what it picked:

```
using NCCL_IB_GID_INDEX=3
```

### 2. The `persistent_topk` patch is mandatory, and short tests will not catch it

The kernel wants ≥128 KB shared memory per block; GB10 has 101 376 B. Past roughly
**24 000 tokens of context**, the engine dies mid-request:

```
RuntimeError: launch_persistent_topk ... FilteredTopK fallback requires >=128KB smem
per block (have 101376). total_ctas=124 > num_sms*occupancy=48
```

The fix is a bind-mount, not an image rebuild — but only one of the two launchers in the
upstream repo carries it. We used the other one, and every short test passed. The first
real user request killed the server.

```
-v $HOME/patches/sparse_attn_indexer_kpool.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py:ro
```

**Test with a long prompt *and* a long answer.** A 40 K prompt with a 15-token reply
proves nothing — the crash happens during decode steps, not prefill. Ours:
29 442 tokens in / 218 out, then 39 625 / 219. Both survived; the engine stayed healthy.

### 3. Building the image: the patch chain is not v1→v8

The published `radixark/vllm-glm53-flash:sm121-v8` image is private, so we rebuilt it. The
Dockerfiles are numbered v1–v9, and the README says "apply v1→v8 in order" — but
**v2 is a debug build (a NaN localiser) and is not part of the chain.** Follow the `FROM`
lines instead:

```
base -> v1 -> v3 -> v4 -> v5 -> v6 -> v7 -> v8
        │      │     │     │     │     │     └ fp8 KV for the FA2 NoPE path
        │      │     │     │     │     └ uninitialised top-k memory
        │      │     │     │     └ PDL race
        │      │     │     └ restore cutlass-dsl 4.6.2
        │      │     └ restore NCCL 2.30.7 (the FlashInfer nightly downgrades it)
        │      └ FlashInfer 0.6.18 (0.6.17 produces NaN on 64–256-row batches)
        └ extend the SM90 NoPE-MLA backend to capability 12
```

[`scripts/build-sm121.sh`](scripts/build-sm121.sh) does this.

### 4. AOC transceivers cook themselves inside the GX10 cages

This one is hardware, and it is the reason two launches died with the engine loaded.

The GX10 pulls air in through the **bottom** and exhausts it out the **back** — where the
QSFP cages are. A 100G AOC dissipates 3.5 W, and with a cable in **both** cages the
modules sit at 68–72 °C idle, against a 70 °C warning and 75 °C alarm. Under load they
cross it and the driver protects the optics:

```
mlx5_core: Port module event[error]: module 0, Cable error, High Temperature
mlx5_core enp1s0f0np0: Link down
```

A worker's rail vanishing mid-inference kills the engine — you get
`TimeoutError: RPC call to sample_tokens timed out`, which looks like a software problem
and is not.

Nothing helps from software. We measured it: `ip link set down` on the idle cage bought
**1 °C** (the module stays powered), and `ethtool --set-module power-mode-policy` is not
supported on this device.

**Use passive DAC.** 1.5 W instead of 3.5, no laser, no DOM sensor, and the failure mode
disappears by construction. After swapping: links stable, 98.01 Gbps per link, 2.58 µs
latency, and the temperature reading is simply gone.

If you must run AOC, watch it — `ethtool -m <iface> | grep "Module temperature"` — and
give the units vertical spacing. Ours were stacked; the two in the middle of the row ran
81–84 °C on the board versus 70–73 at the edges.

### 5. Cables plugged into a running node link up throttled

Known in the community but worth restating: inserting a QSFP cable into a **running**
GX10 leaves the CX7 firmware in a throttled state — ~13.4 Gbps regardless of protocol,
QP count or message size, while **latency stays perfect (2.4 µs)** and every error
counter reads zero. That combination is the signature: a rate limit, not congestion.

**Reboot the node with the cable already inserted.** 13.4 → 98 Gbps, no config change.
Our rule now: recabled → reboot.

## Bonus: how not to measure decode speed

Our first benchmark reported a flat 13.7 tok/s across every content type. The engine's own
metrics said 25–50. The benchmark was wrong: it counted **SSE chunks**, and with
speculative decoding one chunk carries several tokens. Ask for usage instead:

```json
{"stream": true, "stream_options": {"include_usage": true}}
```

and divide `completion_tokens` by the time from first token to last.
[`scripts/bench.py`](scripts/bench.py) does it correctly.

## Running it as a service

Two systemd units, both in [`ops/`](ops/):

- `glm53-flusher.service` runs the unconditional page-cache flusher. GB10's NVRM
  allocator needs it during weight load, and leaving it running costs nothing.
- `glm53-fleet.service` runs the watchdog: it probes `/health` every 60 s and, after
  three consecutive failures, tears every rank down, runs the memory ritual, and
  relaunches worker-first. vLLM v1 cannot revive a dead engine core, and Docker restart
  policies make it worse — a headless worker exits 0 when the head dies, so
  `on-failure` never fires, and the dead head often does not exit at all.

Recovery takes about 15 minutes, so raise `FAIL_THRESHOLD` before pointing it at
anything latency-sensitive.

An enabled supervisor also comes back on its own after a reboot, which matters if you
ever run something else on these nodes. Ours re-armed in the middle of an unrelated
deployment and spent several boots fighting it for the master port and the memory, and
the failures looked like they belonged to the other workload. Before you trust a long
debugging session on shared hardware:

```bash
systemctl list-units | grep -i <anything model-shaped>
```

## What is in here

| Path | |
|---|---|
| [`launch-glm53-tp4.sh`](launch-glm53-tp4.sh) | launcher with per-node GID index and the topk bind-mount |
| [`scripts/build-sm121.sh`](scripts/build-sm121.sh) | rebuild the image chain (skipping v2) |
| [`scripts/bench.py`](scripts/bench.py) | single-stream benchmark that counts real tokens |
| [`scripts/bench-concurrency.py`](scripts/bench-concurrency.py) | concurrency sweep |
| [`scripts/fabric-bench.sh`](scripts/fabric-bench.sh) | RDMA acceptance matrix before you blame the model |
| [`ops/fleet_watchdog.sh`](ops/fleet_watchdog.sh) | health probe plus orchestrated worker-first relaunch |
| [`ops/flusher-unconditional.sh`](ops/flusher-unconditional.sh) | the page-cache flusher, unconditional by design |
| [`ops/glm53-fleet.service`](ops/glm53-fleet.service), [`ops/glm53-flusher.service`](ops/glm53-flusher.service) | systemd units for both |
| [`docs/hardware-notes.md`](docs/hardware-notes.md) | GX10 socket-direct layout, thermals, bandwidth ceilings |

## Credits

The recipe, the patch stack and the hard debugging are
[tonyd2wild](https://github.com/tonyd2wild)'s — this repo only adds the operational
scar tissue from reproducing it on different hardware and a different fabric. Quant by
[LibertAIDAI](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4), model by
[zai-org](https://huggingface.co/zai-org/GLM-5.3-Flash).
