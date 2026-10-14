#!/bin/bash
# TAP throughput test via BlackHole loopback.
#
# Launches two jackrabbit instances (A and B) sharing BlackHole 2ch,
# then floods pings between the TUN endpoints to measure throughput.
#
# Usage:  sudo ./run_iperf_test.sh [protocol] [--wired]
#   e.g.  sudo ./run_iperf_test.sh 11
#         sudo ./run_iperf_test.sh 13 --wired

set -euo pipefail

PROTO="${1:-11}"
WIRED_FLAG=""
if [[ "${2:-}" == "--wired" ]]; then
    WIRED_FLAG="--wired"
fi

BIN="./target/release/jackrabbit"
BH="BlackHole 2ch"
IP_A="10.73.0.1"
IP_B="10.73.0.2"
DURATION=30
# Optional env overrides:
#   TARGET_FRAME_MS=400  MTU_OVERRIDE=...  PING_SIZE=...
TARGET_FRAME_MS="${TARGET_FRAME_MS:-400}"
MTU_OVERRIDE="${MTU_OVERRIDE:-}"
# Ping payload size (+ 28 bytes IP+ICMP header = total packet)
PING_SIZE="${PING_SIZE:-200}"

MTU_FLAGS=()
if [ -n "$MTU_OVERRIDE" ]; then
    MTU_DESC="$MTU_OVERRIDE"
    MTU_FLAGS+=(--mtu "$MTU_OVERRIDE")
else
    MTU_DESC="auto/${TARGET_FRAME_MS}ms"
    MTU_FLAGS+=(--target-frame-ms "$TARGET_FRAME_MS")
fi

if [ ! -f "$BIN" ]; then
    echo "Building..."
    cargo build --release --features cli --bin jackrabbit
fi

cleanup() {
    echo ""
    echo "Cleaning up..."
    kill "$PID_A" "$PID_B" 2>/dev/null || true
    wait "$PID_A" "$PID_B" 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT

echo "=== JackRabbit TAP Throughput Test ==="
echo "Protocol: $PROTO ${WIRED_FLAG:+(wired)}"
echo "Audio: $BH (loopback)"
echo "TUN: $IP_A <-> $IP_B (MTU $MTU_DESC)"
echo "Ping: ${PING_SIZE}+28 bytes for ${DURATION}s"
echo ""

# Kill any leftover instances
pkill -f "jackrabbit.*BlackHole" 2>/dev/null || true
sleep 1

# Launch instance A
echo "Starting instance A ($IP_A)..."
$BIN --protocol "$PROTO" $WIRED_FLAG \
    --output "$BH" --input "$BH" \
    --ip "$IP_A" --peer "$IP_B" "${MTU_FLAGS[@]}" \
    2>/tmp/jackrabbit_a.log &
PID_A=$!

sleep 2

# Launch instance B
echo "Starting instance B ($IP_B)..."
$BIN --protocol "$PROTO" $WIRED_FLAG \
    --output "$BH" --input "$BH" \
    --ip "$IP_B" --peer "$IP_A" "${MTU_FLAGS[@]}" \
    2>/tmp/jackrabbit_b.log &
PID_B=$!

# Wait for TUN interfaces to come up
echo "Waiting for TUN interfaces..."
sleep 3

# Verify connectivity with a single ping (long timeout for slow link)
# macOS ping -W is in milliseconds; Linux ping -W is in seconds.
PING_WAIT=15
PING_WAIT_UNIT="s"
if [[ "$(uname -s)" == "Darwin" ]]; then
    PING_WAIT=15000
    PING_WAIT_UNIT="ms"
fi
echo -n "Ping test: "
if ping -c 1 -W "$PING_WAIT" -s 64 "$IP_B" >/dev/null 2>&1; then
    echo "OK"
else
    echo "FAIL (no response from $IP_B after ${PING_WAIT}${PING_WAIT_UNIT})"
    echo ""
    echo "--- Instance A log ---"
    tail -20 /tmp/jackrabbit_a.log
    echo "--- Instance B log ---"
    tail -20 /tmp/jackrabbit_b.log
    exit 1
fi

# Let link settle
sleep 2

# Flood ping: send as fast as the link allows.
# -f = flood (send next as soon as reply arrives, or after 10ms)
# -s = payload size
# -c = count (enough to fill the duration)
echo ""
echo "Flood ping ($IP_A -> $IP_B, ${PING_SIZE} byte payload, ${DURATION}s)..."
echo ""
# macOS ping -W is in milliseconds; Linux ping -W is in seconds.
FLOOD_WAIT=2
FLOOD_WAIT_UNIT="s"
if [[ "$(uname -s)" == "Darwin" ]]; then
    FLOOD_WAIT=2000
    FLOOD_WAIT_UNIT="ms"
fi
# Use timeout to cap duration; flood ping sends as fast as possible
timeout "$DURATION" ping -f -W "$FLOOD_WAIT" -s "$PING_SIZE" "$IP_B" 2>&1 || true

echo ""
echo ""
echo "=== Instance A stats (sender) ==="
grep -E '^\[' /tmp/jackrabbit_a.log | tail -8
echo ""
echo "=== Instance B stats (receiver) ==="
grep -E '^\[' /tmp/jackrabbit_b.log | tail -8
echo ""
echo "Full logs: /tmp/jackrabbit_{a,b}.log"
