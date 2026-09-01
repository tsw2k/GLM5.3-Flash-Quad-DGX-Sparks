#!/usr/bin/env bash
# Fabric acceptance benchmark: ib_write_bw matrix over the 100G rails.
# Run from the workstation (VPN up). Orchestrates via mgmt SSH; traffic goes over rails.
# Portable to bash 3.2 (macOS) — no associative arrays.
#
#   ./scripts/fabric-bench.sh            # full matrix (singles -> 2x -> 4x)
#   ./scripts/fabric-bench.sh singles    # only individual links
#
# Expected @100G: ~90-97 Gbps per link. Quad test: each node carries 2 streams
# (one per rail) -> per-stream drop toward ~90-95 is fine; node aggregate ~190 Gbps
# proves the GB10 SoC cap, not a fabric problem.
set -uo pipefail

KEY=~/.ssh/mtxc_spark
HCA_A=rocep1s0f0
HCA_B=roceP2p1s0f1
DUR=10
OUT=$(mktemp -d)

# Management addresses: edit for your fleet. Orchestration goes over mgmt on purpose,
# so a flapping compute rail cannot cut the benchmark off from the nodes.
mgmt()  { case $1 in spark-01) echo 192.0.2.10;; spark-02) echo 192.0.2.11;; spark-03) echo 192.0.2.12;; spark-04) echo 192.0.2.13;; esac; }
raila() { case $1 in spark-01) echo 10.77.1.11;;  spark-02) echo 10.77.1.12;;  spark-03) echo 10.77.1.13;;  spark-04) echo 10.77.1.14;;  esac; }
railb() { case $1 in spark-01) echo 10.77.2.11;;  spark-02) echo 10.77.2.12;;  spark-03) echo 10.77.2.13;;  spark-04) echo 10.77.2.14;;  esac; }

sshn() { h=$1; shift; ssh -i "$KEY" -o ConnectTimeout=8 -o BatchMode=yes "mtxc@$h" "$@"; }

# run_bw <label> <src-node> <dst-node> <rail A|B> <port>
run_bw() {
  local label=$1 src=$2 dst=$3 rail=$4 port=$5 hca dstip
  if [ "$rail" = A ]; then hca=$HCA_A; dstip=$(raila "$dst"); else hca=$HCA_B; dstip=$(railb "$dst"); fi
  sshn "$(mgmt "$dst")" "nohup ib_write_bw -d $hca -x 3 -F --report_gbits -D $DUR -p $port >/dev/null 2>&1 &"
  sleep 2
  local res
  res=$(sshn "$(mgmt "$src")" "ib_write_bw -d $hca -x 3 -F --report_gbits -D $DUR -p $port $dstip 2>&1 | awk '/^ 65536/{print \$4}'")
  echo "${label}: ${res:-FAILED} Gbps (avg)" | tee "$OUT/$label"
}

cleanup() { for n in spark-01 spark-02 spark-03 spark-04; do sshn "$(mgmt $n)" "pkill -f ib_write_bw" >/dev/null 2>&1; done; }
trap cleanup EXIT
cleanup

echo "=== Stage 1: individual links ==="
run_bw "s1_01-02_railA" spark-01 spark-02 A 18515
run_bw "s1_01-02_railB" spark-01 spark-02 B 18515
run_bw "s1_03-04_railA" spark-03 spark-04 A 18515
run_bw "s1_03-04_railB" spark-03 spark-04 B 18515
run_bw "s1_01-03_railA" spark-01 spark-03 A 18515
run_bw "s1_02-04_railB" spark-02 spark-04 B 18515

[ "${1:-}" = "singles" ] && exit 0

echo "=== Stage 2: 2 parallel sessions, disjoint pairs, rail A ==="
run_bw "s2_01-02_railA" spark-01 spark-02 A 18601 &
run_bw "s2_03-04_railA" spark-03 spark-04 A 18602 &
wait

echo "=== Stage 3: 4 parallel sessions — both pairs x both rails (SoC cap test) ==="
run_bw "s3_01-02_railA" spark-01 spark-02 A 18701 &
run_bw "s3_01-02_railB" spark-01 spark-02 B 18702 &
run_bw "s3_03-04_railA" spark-03 spark-04 A 18703 &
run_bw "s3_03-04_railB" spark-03 spark-04 B 18704 &
wait

echo
echo "=== Summary ==="
cat "$OUT"/* 2>/dev/null | sort
echo
echo "Pass: singles >= ~90 | stage3 per-stream >= ~85 with node aggregate ~190"
