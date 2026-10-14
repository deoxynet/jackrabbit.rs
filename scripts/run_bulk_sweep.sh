#!/bin/bash
# Sweep burst sizes and protocols for bulk transfer throughput.
#
# Launches fresh jackrabbit instances for each configuration,
# transfers a file via netcat, and prints a summary table.
#
# Usage:  sudo ./run_bulk_sweep.sh
#         sudo ./run_bulk_sweep.sh --wired
#         sudo ./run_bulk_sweep.sh --cable
#
# Environment variables:
#   SIZE_KB=10              Transfer size in KB (default: 10)
#   BURSTS="1 2 4 8"        Burst sizes to sweep (default: 4 wired/cable, 1 2 4 8 OTA)
#   PROTOS="11"             Protocols to sweep (default: 8 wired, 13 cable, 11 OTA)
#   TARGET_FRAME_MS=400     Frame duration target
#   TIMEOUT_SEC=60          Per-transfer timeout

set -uo pipefail

MODE_FLAG=""
MODE_LABEL=""
if [[ "${1:-}" == "--cable" ]]; then
    MODE_FLAG="--cable"
    MODE_LABEL="cable"
elif [[ "${1:-}" == "--wired" ]]; then
    MODE_FLAG="--wired"
    MODE_LABEL="wired"
fi

BIN="./target/release/jackrabbit"
BH="BlackHole 2ch"
IP_A="10.73.0.1"
IP_B="10.73.0.2"
NC_PORT=9000
SIZE_KB="${SIZE_KB:-10}"
TARGET_FRAME_MS="${TARGET_FRAME_MS:-400}"
TIMEOUT_SEC="${TIMEOUT_SEC:-60}"

if [ "$MODE_FLAG" = "--cable" ]; then
    BURSTS="${BURSTS:-4}"
    PROTOS="${PROTOS:-13}"
elif [ "$MODE_FLAG" = "--wired" ]; then
    BURSTS="${BURSTS:-4}"
    PROTOS="${PROTOS:-8}"
else
    BURSTS="${BURSTS:-1 2 4 8}"
    PROTOS="${PROTOS:-11}"
fi

SEND_FILE="/tmp/jackrabbit_send.bin"
RECV_FILE="/tmp/jackrabbit_recv.bin"

# Build if source newer than binary
if [ ! -f "$BIN" ] || [ "$(find src -newer "$BIN" -print -quit 2>/dev/null)" ]; then
    echo "Building..."
    cargo build --release --features cli --bin jackrabbit
fi

# Generate test file once
dd if=/dev/urandom bs=1024 count="$SIZE_KB" of="$SEND_FILE" 2>/dev/null
SEND_SIZE=$(wc -c < "$SEND_FILE" | tr -d ' ')
SEND_MD5=$(md5 -q "$SEND_FILE" 2>/dev/null || md5sum "$SEND_FILE" | cut -d' ' -f1)

# macOS vs Linux ping timeout flag
PING_WAIT=15000
if [[ "$(uname -s)" != "Darwin" ]]; then
    PING_WAIT=15
fi

cleanup_all() {
    echo ""
    echo "Cleaning up..."
    pkill -f "jackrabbit.*BlackHole" 2>/dev/null || true
    lsof -ti:"$NC_PORT" | xargs kill 2>/dev/null || true
    echo "Done."
}
trap cleanup_all EXIT

echo "=== JackRabbit Bulk Transfer Sweep ==="
echo "Audio:    $BH (loopback)"
echo "Transfer: ${SIZE_KB} KB (${SEND_SIZE} bytes)"
echo "Protocols: $PROTOS ${MODE_LABEL:+($MODE_LABEL)}"
echo "Bursts:   $BURSTS"
echo "Timeout:  ${TIMEOUT_SEC}s per transfer"
echo ""

# Result arrays
declare -a R_PROTO=()
declare -a R_BURST=()
declare -a R_TIME=()
declare -a R_KBPS=()
declare -a R_STATUS=()

run_transfer() {
    local proto=$1
    local burst=$2

    echo "--- Protocol $proto, burst $burst ${MODE_LABEL:+($MODE_LABEL)} ---"

    # Kill any leftover instances and nc listeners
    pkill -f "jackrabbit.*BlackHole" 2>/dev/null || true
    lsof -ti:"$NC_PORT" | xargs kill 2>/dev/null || true
    sleep 1

    # Launch instance A
    JACKRABBIT_DEBUG=1 $BIN --protocol "$proto" $MODE_FLAG \
        --output "$BH" --input "$BH" \
        --ip "$IP_A" --peer "$IP_B" \
        --burst "$burst" --target-frame-ms "$TARGET_FRAME_MS" \
        2>/tmp/jackrabbit_a.log &
    local pid_a=$!

    sleep 2

    # Launch instance B
    JACKRABBIT_DEBUG=1 $BIN --protocol "$proto" $MODE_FLAG \
        --output "$BH" --input "$BH" \
        --ip "$IP_B" --peer "$IP_A" \
        --burst "$burst" --target-frame-ms "$TARGET_FRAME_MS" \
        2>/tmp/jackrabbit_b.log &
    local pid_b=$!

    sleep 3

    # Verify connectivity
    if ! ping -c 1 -W "$PING_WAIT" -s 64 "$IP_B" >/dev/null 2>&1; then
        echo "  Ping FAIL — skipping"
        kill "$pid_a" "$pid_b" 2>/dev/null || true
        wait "$pid_a" "$pid_b" 2>/dev/null || true
        R_PROTO+=("$proto")
        R_BURST+=("$burst")
        R_TIME+=("-")
        R_KBPS+=("-")
        R_STATUS+=("NO LINK")
        return
    fi
    echo -n "  Ping OK, transferring ${SIZE_KB} KB... "

    sleep 1
    rm -f "$RECV_FILE"

    # Start receiver
    nc -l "$NC_PORT" > "$RECV_FILE" &
    local nc_pid=$!
    sleep 1

    # Send in background and time until receiver completes or timeout.
    # Use a short idle timeout on nc to avoid hanging on close.
    local t_start
    t_start=$(python3 -c "import time; print(time.time())")
    cat "$SEND_FILE" | nc -w 5 "$IP_B" "$NC_PORT" &
    local send_pid=$!

    local timed_out=0
    local wait_start=$SECONDS
    while true; do
        if [ -f "$RECV_FILE" ]; then
            local cur_size
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

    local t_end
    t_end=$(python3 -c "import time; print(time.time())")

    kill "$send_pid" 2>/dev/null || true
    kill "$nc_pid" 2>/dev/null || true
    wait "$send_pid" 2>/dev/null || true
    wait "$nc_pid" 2>/dev/null || true

    if [ "$timed_out" -eq 1 ]; then
        echo "TIMEOUT"
        kill "$pid_a" "$pid_b" 2>/dev/null || true
        wait "$pid_a" "$pid_b" 2>/dev/null || true
        R_PROTO+=("$proto")
        R_BURST+=("$burst")
        R_TIME+=(">${TIMEOUT_SEC}s")
        R_KBPS+=("-")
        R_STATUS+=("TIMEOUT")
        return
    fi

    local recv_size
    recv_size=$(wc -c < "$RECV_FILE" | tr -d ' ')

    # Check integrity
    local recv_md5
    recv_md5=$(md5 -q "$RECV_FILE" 2>/dev/null || md5sum "$RECV_FILE" | cut -d' ' -f1)

    local elapsed kbps status
    elapsed=$(python3 -c "print(f'{$t_end - $t_start:.1f}')")

    if [ "$recv_size" -eq "$SEND_SIZE" ] && [ "$SEND_MD5" = "$recv_md5" ]; then
        status="PASS"
        kbps=$(python3 -c "
e = $t_end - $t_start
print(f'{$recv_size * 8 / e / 1000:.2f}' if e > 0 else '0')
")
        echo "${elapsed}s, ${kbps} kbps, md5 OK"
    elif [ "$recv_size" -ne "$SEND_SIZE" ]; then
        status="INCOMPLETE"
        kbps="-"
        echo "${elapsed}s, got ${recv_size}/${SEND_SIZE} bytes"
    else
        status="BAD MD5"
        kbps="-"
        echo "${elapsed}s, md5 MISMATCH"
    fi

    # Tear down instances
    kill "$pid_a" "$pid_b" 2>/dev/null || true
    wait "$pid_a" "$pid_b" 2>/dev/null || true

    R_PROTO+=("$proto")
    R_BURST+=("$burst")
    R_TIME+=("${elapsed}s")
    R_KBPS+=("$kbps")
    R_STATUS+=("$status")
}

# Run the sweep
for proto in $PROTOS; do
    for burst in $BURSTS; do
        run_transfer "$proto" "$burst"
        echo ""
    done
done

# Summary table
echo ""
echo "=============================================="
echo "  BULK TRANSFER SWEEP RESULTS"
echo "  ${SIZE_KB} KB payload | ${BH} loopback"
echo "=============================================="
echo ""
printf "%-8s  %-6s  %8s  %10s  %-10s\n" "Proto" "Burst" "Time" "Throughput" "Status"
printf "%-8s  %-6s  %8s  %10s  %-10s\n" "--------" "------" "--------" "----------" "----------"

BEST_IDX=""
BEST_VAL=0

for i in "${!R_PROTO[@]}"; do
    local_kbps="${R_KBPS[$i]}"
    if [ "$local_kbps" != "-" ]; then
        thr="${local_kbps} kbps"
        cmp=$(python3 -c "print(1 if $local_kbps > $BEST_VAL else 0)")
        if [ "$cmp" = "1" ]; then
            BEST_VAL="$local_kbps"
            BEST_IDX="$i"
        fi
    else
        thr="  -"
    fi
    printf "%-8s  %-6s  %8s  %10s  %-10s\n" \
        "${R_PROTO[$i]}" "${R_BURST[$i]}" "${R_TIME[$i]}" "$thr" "${R_STATUS[$i]}"
done

echo ""
if [ -n "$BEST_IDX" ]; then
    echo "Best: protocol ${R_PROTO[$BEST_IDX]}, burst ${R_BURST[$BEST_IDX]} -> ${R_KBPS[$BEST_IDX]} kbps"
fi
echo ""
