# GLM-5.3-Flash on four DGX Sparks: what it took to actually keep it up

Serving [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (320B MoE)
at TP=4 across four **ASUS Ascent GX10** (NVIDIA GB10, DGX Spark class), on a 100G RoCEv2
fabric.

This is a field report, not another quickstart. The recipe itself is
[tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark)
and it is excellent. Everything here is what we hit *around* it: three failed launches, a
fabric that overheated itself, and a silent NCCL hang that looks exactly like hard work.

## Results

Fabric: Arista 7060CX-32S, 100G, single rail, passive DAC. Model:
`RedHatAI/GLM-5.3-Flash-NVFP4`, TP=4, fp8 KV-cache, DFlash2 speculative decoding at k=7,
`--enforce-eager`, `--max-model-len 1048576`, KV 24 GiB/rank.

| Content regime | TTFT | Decode |
|---|---|---|
| structured (counting, lists) | 0.254 s | **76.0 tok/s** |
| code generation | 0.187 s | **71.3 tok/s** |
| freeform prose | 0.185 s | **31.6 tok/s** |

| Concurrency | Aggregate | Per stream |
|---|---|---|
| 1 | 65.4 tok/s | 65.4 |
| 2 | 104.2 tok/s | 52.1 |
| 8 | 99.9 tok/s | 12.5 |

KV pool 3,895,606 tokens, 3.72x a full 1M-token context. DFlash2 accepts 5.43 of 7 draft
tokens on average.

The first run of this stack used the ModelOpt checkpoint and MTP-4 drafting: 64.0
structured, 46.9 on code, 33.4 on prose. Changing the checkpoint and the drafter moved
code generation by half. Prose did not move at all, which is what you would expect, since
a drafter only wins where the next tokens are predictable. The C=8 number went the wrong
way and we have not worked out why.

Both sets of numbers match the reference build, which was measured on a 200G fabric. So
decode here is not fabric-bound. Worth knowing before anyone buys faster switching to make
tokens come out quicker. More fabric would help with prefill of long contexts and with
heavy multi-tenant load, not with this.

## The things that cost us a day

### 0. The obvious checkpoint is the one that corrupts tokens

We served `LibertAIDAI/GLM-5.3-Flash-NVFP4` for a day before upstream flagged the
problem. ModelOpt-quantized NVFP4 builds emit occasional corrupted token IDs
([vLLM #54150](https://github.com/vllm-project/vllm/issues/54150)). In English you barely
notice. Land one inside a tool-call block and the parser loses the thread, after which
generation can lock into a repetition loop.

`RedHatAI/GLM-5.3-Flash-NVFP4` is a compressed-tensors quant of the same architecture and
drops straight in: same flags, same launcher, only the path changes. It also loads faster,
11 large shards instead of 120 small ones.

Upstream measured 4, 9 and 8 replacement characters over three runs on ModelOpt. We ran
the same probe after switching and got 0, 0, 0:

```python
# Korean prompt, temperature 0, three passes, count U+FFFD in the output.
# Latin script hides this; scripts with multi-byte codepoints do not.
prompt = "한국어로 인공지능의 미래에 대해 200자 정도로 설명해 주세요."
bad = response["choices"][0]["message"]["content"].count("\ufffd")
```

The trade is real. RedHatAI quantizes activations too (W4A4 against W4A16), so hard
reasoning loses a little. We took it. Correct output beats slightly better reasoning.

Check what you are actually running:

```bash
python3 -c "import json;print(json.load(open('config.json'))['quantization_config']['quant_method'])"
# compressed-tensors  good
# modelopt            swap it
```

### 1. `NCCL_IB_GID_INDEX` is per node, and getting it wrong hangs silently

The RoCEv2/IPv4 GID does not live at the same index on every node. Ours:

| Node | rail A | rail B |
|---|---|---|
| spark-01 | 3 | 4 |
| spark-02 | 3 | 3 |
| spark-03 | 3 | 4 |
| spark-04 | **4** | 4 |

An index becomes a hole when an address is removed and re-added, because the new GID
takes the next free slot instead of the old one. Check it, never assume:

```bash
for i in 0 1 2 3 4 5; do
  echo -n "idx$i: "; cat /sys/class/infiniband/rocep1s0f0/ports/1/gid_attrs/types/$i 2>/dev/null
  cat /sys/class/infiniband/rocep1s0f0/ports/1/gids/$i 2>/dev/null
done
```

There are two failure modes, and the second is the nasty one. A wrong explicit index
gives you `ibv_modify_qp failed ... local GID ::`, which is a clean and obvious error.
Leaving the index unset so NCCL auto-selects makes the collective hang instead: twenty
minutes of no log output, GPUs pinned at 96% utilisation.

That 96% at 22 to 26 W is not compute, it is a spin-wait. Real weight loading on a GB10
draws far more. Watching `nvidia-smi --query-gpu=utilization.gpu,power.draw` beats
watching the log, because high utilisation with low power means a hung collective every
time.

Fix: set it per rank ([`launch-glm53-tp4.sh`](launch-glm53-tp4.sh)).

**Update, after a day of reboots: a per-rank pin goes stale too.** The table above was
true when we wrote it. We rebooted the fleet a few times for unrelated reasons, and
spark-04's RoCEv2 GID moved from index 4 back to 3, so the pin that had been correct
became a pin to an empty slot. The symptom is not the clean `local GID ::` error you get
from a wrong index at cabling time. It is `NCCL error: unhandled system error` at init,
on rank 3 only, with the head reporting nothing more useful than "WorkerProc
initialization failed". Two relaunches and a full fleet reboot did not help, because
none of them changed the pin.

The launcher now reads the index out of sysfs before it starts the container, and keeps
the pinned value only as a fallback. It is six lines, and it prints what it picked:

```
using NCCL_IB_GID_INDEX=3
```

### 2. The `persistent_topk` patch is mandatory, and short tests will not catch it

The kernel wants at least 128 KB of shared memory per block; GB10 has 101,376 B. Past
roughly 24,000 tokens of context, the engine dies mid-request:

```
RuntimeError: launch_persistent_topk ... FilteredTopK fallback requires >=128KB smem
per block (have 101376). total_ctas=124 > num_sms*occupancy=48
```

The fix is a bind-mount rather than an image rebuild, but only one of the two launchers
in the upstream repo carries it. We used the other one, and every short test passed. The first
real user request killed the server.

```
-v $HOME/patches/sparse_attn_indexer_kpool.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py:ro
```

**Test with a long prompt *and* a long answer.** A 40 K prompt with a 15-token reply
proves nothing, because the crash happens during decode steps rather than prefill. Ours:
29,442 tokens in and 218 out, then 39,625 and 219. Both survived and the engine stayed
healthy.

### 3. Building the image, which you no longer have to do

Upstream now publishes the image, anonymously pullable:

```bash
docker pull ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2
```

Pull it on one node. Four nodes pulling 31 GB at once will hit GHCR rate limits, and
Docker Hub will refuse anonymous pulls of the base image long before that.

The rest of this section is history, and still useful if you ever need to rebuild. The
Dockerfiles are numbered v1 to v9, and the README says to apply v1 through v8 in order,
but v2 is a debug build, a NaN localiser, and is not in the chain. Follow the `FROM` lines:

```
base -> v1 -> v3 -> v4 -> v5 -> v6 -> v7 -> v8
        |      |     |     |     |     |     ` fp8 KV for the FA2 NoPE path
        |      |     |     |     |     ` uninitialised top-k memory
        |      |     |     |     ` PDL race
        |      |     |     ` restore cutlass-dsl 4.6.2
        |      |     ` restore NCCL 2.30.7, the FlashInfer nightly downgrades it
        |      ` FlashInfer 0.6.18, since 0.6.17 produces NaN on 64 to 256-row batches
        ` extend the SM90 NoPE-MLA backend to capability 12
```

[`scripts/build-sm121.sh`](scripts/build-sm121.sh) does this.

### 4. AOC transceivers cook themselves inside the GX10 cages

This one is hardware, and it is the reason two launches died with the engine loaded.

The GX10 pulls air in through the bottom and exhausts it out the back, which is where
the QSFP cages are. A 100G AOC dissipates 3.5 W, and with a cable in both cages the
modules sit at 68 to 72 °C idle, against a 70 °C warning and a 75 °C alarm. Under load they
cross it and the driver protects the optics:

```
mlx5_core: Port module event[error]: module 0, Cable error, High Temperature
mlx5_core enp1s0f0np0: Link down
```

A worker's rail vanishing mid-inference kills the engine. You get
`TimeoutError: RPC call to sample_tokens timed out`, which looks like a software problem
and is not.

Nothing helps from software. We measured it: `ip link set down` on the idle cage bought
1 °C, because the module stays powered, and `ethtool --set-module power-mode-policy` is
not supported on this device.

**Use passive DAC.** 1.5 W instead of 3.5, no laser, no DOM sensor, and the failure mode
disappears by construction. After swapping: links stable, 98.01 Gbps per link, 2.58 µs
latency, and the temperature reading is simply gone.

If you must run AOC, watch it with `ethtool -m <iface> | grep "Module temperature"` and
give the units vertical spacing. Ours were stacked, and the two in the middle of the row
ran 81 to 84 °C on the board against 70 to 73 at the edges.

### 5. Cables plugged into a running node link up throttled

Known in the community but worth restating: inserting a QSFP cable into a **running**
GX10 leaves the CX7 firmware in a throttled state at about 13.4 Gbps regardless of
protocol, QP count or message size, while latency stays perfect at 2.4 µs and every error
counter reads zero. That combination is the signature of a rate limit rather than
congestion.

**Reboot the node with the cable already inserted.** That takes it from 13.4 to 98 Gbps
with no config change. Our rule now: recabled means rebooted.

## Moving 180 GB around without wasting the fabric

Both the weights and the image have to reach every node, and the obvious ways are slow
for the same reason. rsync over SSH and `docker save | ssh` both encrypt, and AES on the
GB10's ARM cores tops out near 1 Gbps. On a 100G fabric that is one percent of the wire.

The rails are an isolated L2 segment with no routing, so there is nothing to protect the
traffic from. We run an rsync daemon instead, read-only, restricted to the rail subnets:

```ini
# /etc/rsyncd.conf
uid = mtxc
gid = mtxc
use chroot = no
read only = yes
hosts allow = 10.77.1.0/24 10.77.2.0/24
hosts deny = *

[models]
path = /var/tmp
```

Measured on one 4 GiB shard between two nodes: 1977 MB/s, against roughly 1 Gbps over
SSH. The image goes the same way, `docker save` to a tar under the module root, then each
node pulls it and runs `docker load`. It costs 31 GB of scratch space and one extra
write-read cycle, and it is still far quicker than encrypting the same bytes.

## The supervisor will fight you during maintenance

We run a watchdog that probes `/health` and rebuilds the fleet when the engine dies,
because vLLM cannot recover a dead engine core and Docker restart policies do not help:
headless workers exit 0 when the head dies, so `on-failure` never fires, and the dead head
often does not exit at all.

It is the right tool and it will still ruin your afternoon. Halfway through swapping the
image and the checkpoint, it saw `/health` fail, decided the fleet was down, tore down the
containers we had just started by hand, and relaunched from its own configuration. That
configuration still named the previous launcher, which an upstream pull had deleted on the
head node. We ended up with three workers on the old stack and no head at all, which is
worse than either state we were moving between.

Stop it before you touch anything:

```bash
sudo systemctl stop glm53-fleet
# do the work, verify a real launch
sudo systemctl start glm53-fleet
```

For shorter work, `touch ~/.fleet_watchdog.pause` pauses it without stopping the unit;
`rm` it to resume.

And after every upstream pull, re-check the four things the watchdog keeps its own copy
of: the launcher path, the node map, the SSH key, and the health URL. Ours pointed at a
file that no longer existed and nothing warned us.

## Bonus: how not to measure decode speed

Our first benchmark reported a flat 13.7 tok/s across every content type. The engine's own
metrics said 25 to 50. The benchmark was wrong: it counted SSE chunks, and with
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
- `glm53-fleet.service` runs the watchdog. Every 60 s it checks that every rank's
  container is running, probes `/health`, and sends a one-token canary request. After
  three consecutive failures it tears every rank down, runs the memory ritual, and
  relaunches worker-first; after three failed relaunches in a row it gives up and
  writes `~/.fleet_watchdog.gaveup` instead of looping. vLLM v1 cannot revive a dead engine core, and Docker restart
  policies make it worse, since a headless worker exits 0 when the head dies so
  `on-failure` never fires, and the dead head often does not exit at all.

Recovery takes about 15 minutes, so raise `FAIL_THRESHOLD` before pointing it at
anything latency-sensitive.

`/health` alone is not enough. In a failover test on DeepSeek-V4.1-Flash (same vLLM
multi-node executor, same four nodes), killing one worker left the head answering
`/health` with 200 and logging nothing, while client requests hung without a reply until
an NCCL timeout took the head down about six minutes later. The head is blocked in a
collective. The per-rank container check catches a dead worker within one probe, and the
canary catches a stuck engine that still looks healthy. This was measured on V4.1, not
on GLM-5.3; the mechanism is the same.

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
| [`ops/fleet_watchdog.sh`](ops/fleet_watchdog.sh) | per-rank, health and canary probes plus orchestrated worker-first relaunch |
| [`ops/flusher-unconditional.sh`](ops/flusher-unconditional.sh) | the page-cache flusher, unconditional by design |
| [`ops/glm53-fleet.service`](ops/glm53-fleet.service), [`ops/glm53-flusher.service`](ops/glm53-flusher.service) | systemd units for both |
| [`docs/hardware-notes.md`](docs/hardware-notes.md) | GX10 socket-direct layout, thermals, bandwidth ceilings |

## Credits

The recipe, the patch stack and the hard debugging are
[tonyd2wild](https://github.com/tonyd2wild)'s. This repo only adds the operational scar
tissue from reproducing it on different hardware and a different fabric. Quant by
[LibertAIDAI](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4), model by
[zai-org](https://huggingface.co/zai-org/GLM-5.3-Flash).
