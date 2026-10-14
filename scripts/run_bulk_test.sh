#!/bin/bash
# Bulk file transfer test via BlackHole loopback.
#
# Launches two jackrabbit instances (A and B) sharing BlackHole 2ch,
# then sends a file from A to B using netcat to measure one-way
# bulk transfer throughput with burst pipelining.
#
# Usage:  sudo ./run_bulk_test.sh [protocol] [--wired]
#   e.g.  sudo ./run_bulk_test.sh 11
#         sudo ./run_bulk_test.sh 8 --wired
#
# Environment variables:
#   BURST=4         Frames per TX burst (default: 4)
#   SIZE_KB=10      Transfer size in KB (default: 10)
#   TARGET_FRAME_MS=400  Frame duration target

set -euo pipefail

PROTO="${1:-8}"
WIRED_FLAG=""
if [[ "${2:-}" == "--wired" ]]; then
    WIRED_FLAG="--wired"
fi

BIN="./target/release/jackrabbit"
BH="BlackHole 2ch"
IP_A="10.73.0.1"
IP_B="10.73.0.2"
NC_PORT=9000
BURST="${BURST:-4}"
SIZE_KB="${SIZE_KB:-10}"
TARGET_FRAME_MS="${TARGET_FRAME_MS:-400}"
TIMEOUT_SEC="${TIMEOUT_SEC:-120}"

SEND_FILE="/tmp/jackrabbit_send.bin"
RECV_FILE="/tmp/jackrabbit_recv.bin"

# Build if source newer than binary
if [ ! -f "$BIN" ] || [ "$(find src -newer "$BIN" -print -quit 2>/dev/null)" ]; then
    echo "Building..."
    cargo build --release --features cli --bin jackrabbit
fi

cleanup() {
    echo ""
    echo "Cleaning up..."
    # Kill netcat processes
    kill "$NC_RECV_PID" 2>/dev/null || true
    # Kill jackrabbit instances
    kill "$PID_A" "$PID_B" 2>/dev/null || true
    wait "$PID_A" "$PID_B" 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT

echo "=== JackRabbit Bulk Transfer Test ==="
echo "Protocol: $PROTO ${WIRED_FLAG:+(wired)}"
echo "Audio:    $BH (loopback)"
echo "TUN:      $IP_A -> $IP_B"
echo "Burst:    $BURST frames"
echo "Transfer: ${SIZE_KB} KB"
echo ""

# Kill any leftover instances
pkill -f "jackrabbit.*BlackHole" 2>/dev/null || true
sleep 1

# Generate test file
echo "Generating ${SIZE_KB} KB test file..."
dd if=/dev/urandom bs=1024 count="$SIZE_KB" of="$SEND_FILE" 2>/dev/null
SEND_SIZE=$(wc -c < "$SEND_FILE" | tr -d ' ')
echo "  $SEND_FILE ($SEND_SIZE bytes)"
rm -f "$RECV_FILE"

# Launch instance A (sender side)
echo ""
echo "Starting instance A ($IP_A)..."
JACKRABBIT_DEBUG=1 $BIN --protocol "$PROTO" $WIRED_FLAG \
    --output "$BH" --input "$BH" \
    --ip "$IP_A" --peer "$IP_B" \
    --burst "$BURST" --target-frame-ms "$TARGET_FRAME_MS" \
    2>/tmp/jackrabbit_a.log &
PID_A=$!

sleep 2

# Launch instance B (receiver side)
echo "Starting instance B ($IP_B)..."
JACKRABBIT_DEBUG=1 $BIN --protocol "$PROTO" $WIRED_FLAG \
    --output "$BH" --input "$BH" \
    --ip "$IP_B" --peer "$IP_A" \
    --burst "$BURST" --target-frame-ms "$TARGET_FRAME_MS" \
    2>/tmp/jackrabbit_b.log &
PID_B=$!

echo "Waiting for TUN interfaces..."
sleep 3

# Verify connectivity with a single ping
PING_WAIT=15000
if [[ "$(uname -s)" != "Darwin" ]]; then
    PING_WAIT=15
fi
echo -n "Ping check: "
if ping -c 1 -W "$PING_WAIT" -s 64 "$IP_B" >/dev/null 2>&1; then
    echo "OK"
else
    echo "FAIL (no response from $IP_B)"
    echo ""
    echo "--- Instance A log ---"
    tail -20 /tmp/jackrabbit_a.log
    echo "--- Instance B log ---"
    tail -20 /tmp/jackrabbit_b.log
    exit 1
fi

sleep 2

# Start receiver (listen on B's IP)
echo ""
echo "Starting receiver on $IP_B:$NC_PORT..."
nc -l "$NC_PORT" > "$RECV_FILE" &
NC_RECV_PID=$!
sleep 1

# Send file and measure time
echo "Sending ${SIZE_KB} KB ($SEND_SIZE bytes)..."
echo ""
T_START=$(python3 -c "import time; print(time.time())")

cat "$SEND_FILE" | nc -w 5 "$IP_B" "$NC_PORT" &
SEND_PID=$!

# Wait for receiver to finish or until timeout; avoid truncating slow links.
wait_start=$SECONDS
timed_out=0
while true; do
    if [ -f "$RECV_FILE" ]; then
        cur_size=$(wc -c < "$RECV_FILE" | tr -d ' ')
        if [ "$cur_size" -ge "$SEND_SIZE" ]; then
            break
        fi
    fi
    if [ $((SECONDS - wait_start)) -ge "$TIMEOUT_SEC" ]; then
        timed_out=1
        break
    fi
    sleep 0.2
done

T_END=$(python3 -c "import time; print(time.time())")

kill "$SEND_PID" 2>/dev/null || true
kill "$NC_RECV_PID" 2>/dev/null || true
wait "$SEND_PID" 2>/dev/null || true
wait "$NC_RECV_PID" 2>/dev/null || true
NC_RECV_PID=""

RECV_SIZE=$(wc -c < "$RECV_FILE" | tr -d ' ')

# Calculate results
ELAPSED=$(python3 -c "print(f'{$T_END - $T_START:.2f}')")
if [ "$RECV_SIZE" -gt 0 ]; then
    THROUGHPUT=$(python3 -c "
elapsed = $T_END - $T_START
if elapsed > 0:
    bps = $RECV_SIZE * 8 / elapsed
    if bps >= 1000:
        print(f'{bps/1000:.1f} kbps')
    else:
        print(f'{bps:.0f} bps')
else:
    print('N/A')
")
else
    THROUGHPUT="N/A"
fi

# Verify integrity
SEND_MD5=$(md5 -q "$SEND_FILE" 2>/dev/null || md5sum "$SEND_FILE" | cut -d' ' -f1)
RECV_MD5=$(md5 -q "$RECV_FILE" 2>/dev/null || md5sum "$RECV_FILE" | cut -d' ' -f1)

echo "=== Results ==="
echo "Sent:       $SEND_SIZE bytes"
echo "Received:   $RECV_SIZE bytes"
echo "Time:       ${ELAPSED}s"
echo "Throughput: $THROUGHPUT"
echo ""
if [ "$SEND_MD5" = "$RECV_MD5" ]; then
    echo "Integrity:  PASS (md5 match)"
else
    echo "Integrity:  FAIL"
    echo "  Send md5: $SEND_MD5"
    echo "  Recv md5: $RECV_MD5"
fi

echo ""
echo "=== Instance A stats (sender) ==="
grep -E '^\[' /tmp/jackrabbit_a.log | tail -8
echo ""
echo "=== Instance B stats (receiver) ==="
grep -E '^\[' /tmp/jackrabbit_b.log | tail -8
echo ""
echo "Full logs: /tmp/jackrabbit_{a,b}.log"
