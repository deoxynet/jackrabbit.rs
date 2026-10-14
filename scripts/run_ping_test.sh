#!/bin/bash
# Simple ping test via BlackHole loopback.
#
# Launches two jackrabbit instances sharing BlackHole 2ch,
# then sends individual pings to verify the link works.
#
# Usage:  sudo ./run_ping_test.sh [protocol] [--wired]
#   e.g.  sudo ./run_ping_test.sh 11
#         sudo ./run_ping_test.sh 13 --wired

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
TARGET_FRAME_MS="${TARGET_FRAME_MS:-400}"
NUM_PINGS="${NUM_PINGS:-5}"
PING_SIZE="${PING_SIZE:-56}"

# Build if needed
if [ ! -f "$BIN" ] || [ "$(find src -newer "$BIN" -print -quit 2>/dev/null)" ]; then
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

echo "=== JackRabbit Ping Test ==="
echo "Protocol: $PROTO ${WIRED_FLAG:+(wired)}"
echo "Audio:    $BH (loopback)"
echo "TUN:      $IP_A <-> $IP_B"
echo "Pings:    $NUM_PINGS x ${PING_SIZE} bytes"
echo ""

# Kill any leftover instances
pkill -f "jackrabbit.*BlackHole" 2>/dev/null || true
sleep 1

# Launch instance A (with debug)
echo "Starting instance A ($IP_A)..."
JACKRABBIT_DEBUG=1 $BIN --protocol "$PROTO" $WIRED_FLAG \
    --output "$BH" --input "$BH" \
    --ip "$IP_A" --peer "$IP_B" --target-frame-ms "$TARGET_FRAME_MS" \
    2>/tmp/jackrabbit_a.log &
PID_A=$!

sleep 2

# Launch instance B (with debug)
echo "Starting instance B ($IP_B)..."
JACKRABBIT_DEBUG=1 $BIN --protocol "$PROTO" $WIRED_FLAG \
    --output "$BH" --input "$BH" \
    --ip "$IP_B" --peer "$IP_A" --target-frame-ms "$TARGET_FRAME_MS" \
    2>/tmp/jackrabbit_b.log &
PID_B=$!

echo "Waiting for TUN interfaces..."
sleep 3

echo ""
echo "--- Sending $NUM_PINGS pings ($IP_A -> $IP_B) ---"
echo ""

# macOS ping -W is in milliseconds
PING_WAIT=10000
if [[ "$(uname -s)" != "Darwin" ]]; then
    PING_WAIT=10
fi

PASS=0
FAIL=0
for i in $(seq 1 "$NUM_PINGS"); do
    echo -n "Ping $i/$NUM_PINGS: "
    if ping -c 1 -W "$PING_WAIT" -s "$PING_SIZE" "$IP_B" 2>&1 | grep -E 'bytes from|time='; then
        PASS=$((PASS + 1))
    else
        echo "TIMEOUT (no response in ${PING_WAIT}ms)"
        FAIL=$((FAIL + 1))
    fi
    # Brief pause between pings to let link settle
    sleep 2
done

echo ""
echo "=== Results: $PASS/$NUM_PINGS passed, $FAIL failed ==="
echo ""
echo "--- Instance A stats ---"
grep -E '^\[' /tmp/jackrabbit_a.log | tail -12
echo ""
echo "--- Instance B stats ---"
grep -E '^\[' /tmp/jackrabbit_b.log | tail -12
echo ""

# Show startup banner for both
echo "--- Instance A config ---"
head -12 /tmp/jackrabbit_a.log
echo ""
echo "--- Instance B config ---"
head -12 /tmp/jackrabbit_b.log
echo ""
echo "Full logs: /tmp/jackrabbit_{a,b}.log"
