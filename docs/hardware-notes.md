# GX10 / GB10 hardware notes

Measured on four ASUS Ascent GX10 units, not taken from spec sheets.

## Networking is not "two 200G ports"

Each unit has **two QSFP cages**, and each cage is reachable from **both** of the node's
PCIe Gen5 x4 links (socket-direct). The OS therefore shows four interfaces for two
physical ports:

| Cage | via PCIe domain 0000 | via PCIe domain 0002 |
|---|---|---|
| A | `enp1s0f0np0` / `rocep1s0f0` | `enP2p1s0f0np0` / `roceP2p1s0f0` |
| B | `enp1s0f1np1` / `rocep1s0f1` | `enP2p1s0f1np1` / `roceP2p1s0f1` |

Confirmed by reading the transceiver serial through both interfaces of a pair — same
cable — and by the switch seeing both MACs on one port.

**The node's ceiling is PCIe, not the wire: 2 × x4 ≈ 220 Gbps.**

| Cabling | Wire | Delivered | Notes |
|---|---|---|---|
| 1 × 100G | 98 | **98 (measured)** | one PF saturates it |
| 2 × 100G, one per cage | 196 | ~196 | one PF per cage, from different domains |
| 1 × 200G | 196 | ~190 | needs BOTH PFs of that cage striped in NCCL |
| 2 × 200G | 392 | ~200–220 | PCIe-bound; the wires idle |

Two consequences people get wrong:

- Two subnets on **one** cable is not dual-rail. We measured 49 + 49 Gbps — the PFs
  simply split the wire.
- At 200G a single PF caps near 110 Gbps (PCIe x4), so reaching ~190 on one cable
  *requires* both PFs of that cage in `NCCL_IB_HCA`. That is why reference configs list
  two HCAs.

For real dual-rail, use one PF per cage and take them from **different** PCIe domains, so
both x4 links carry traffic.

## Airflow: bottom intake, rear exhaust — and the optics sit in the exhaust

The GX10 draws air through vents in the **bottom** of the chassis and exhausts out the
**back**. The QSFP cages are at the back, so transceivers sit in the node's own hot
airflow.

Measured with two 3.5 W AOCs installed (idle, chassis otherwise cool at GPU 51–56 °C):

| Node | cage A | cage B |
|---|---|---|
| 1 | 68.2 °C | 67.9 °C |
| 2 | **71.7 °C** | 70.4 °C |
| 3 | 69.4 °C | 68.2 °C |
| 4 | 69.3 °C | 68.5 °C |

Module warning is **70 °C**, alarm **75 °C**. Under load they cross it and the driver
drops the link. With passive DAC (1.5 W, no optics) the problem does not exist.

Placement matters too: stacked units in the middle of a row ran **81–84 °C** on the board
versus 70–73 at the edges. Bottom intake means each unit needs clearance underneath —
stacking them flat starves it.

## Cable inserted into a running node → throttled link

Hot-plugging a QSFP cable leaves the CX7 firmware limited to ~13.4 Gbps: identical for
RDMA and TCP, unchanged by QP count or message size, and — the tell — **latency stays
perfect at 2.4 µs with every error counter at zero**. That is a rate limit, not
congestion.

Reboot the node with the cable already in: 13.4 → 98 Gbps.

## Verified fabric numbers (100G DAC, Arista 7060CX-32S)

| Measurement | Value |
|---|---|
| RDMA write bandwidth per link | 98.01 Gbps |
| Both rails in parallel, one node | 196 Gbps |
| RDMA write latency | 2.58 µs |
| Jumbo (9000 B) across all pairs | pass |

Switch side: MTU 9214, PFC on priority 3 no-drop, isolated L2 per rail (no SVI).
Node side: MTU 9000, DSCP 26 → TC3.
