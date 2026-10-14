#!/bin/bash
# Throughput measurement through hardware loopback.
#
# Sends a fixed payload through each protocol via the audio interface,
# records success/failure, and computes effective throughput.
#
# Usage:
#   ./run_throughput.sh              # test all protocols (0–15)
#   ./run_throughput.sh 8 13 14      # test specific protocols

set -uo pipefail

# --- Configuration ---
PAYLOAD_LENGTH=4000
OUTPUT_DEVICE="Scarlett"
INPUT_DEVICE="Scarlett"
PAUSE_BETWEEN=2  # seconds between tests to let audio device settle
# ---------------------

if [ $# -gt 0 ]; then
    PROTOCOLS=("$@")
else
    PROTOCOLS=($(seq 0 15))
fi

PAYLOAD=$(python3 -c "print('a'*${PAYLOAD_LENGTH}, end='')")

echo "Building release binary..."
cargo build --release --example ota_loopback 2>&1 | tail -3
echo ""

# Result arrays
declare -a R_ID=()
declare -a R_NAME=()
declare -a R_DUR=()
declare -a R_STATUS=()
declare -a R_KBPS=()
declare -a R_BPS=()

echo "Throughput test: ${PAYLOAD_LENGTH}-byte payload, ${#PROTOCOLS[@]} protocols"
echo "Devices: ${OUTPUT_DEVICE} / ${INPUT_DEVICE}"
echo ""

for PROTO in "${PROTOCOLS[@]}"; do
    echo -n "Protocol ${PROTO}... "

    OUTPUT=$(cargo run --release --example ota_loopback -- \
        --protocol "${PROTO}" \
        --payload "${PAYLOAD}" \
        --output-device "${OUTPUT_DEVICE}" \
        --input-device "${INPUT_DEVICE}" 2>&1) || true

    # Parse protocol name: "Protocol: 13 (WIRED_TURBO)"
    NAME=$(echo "$OUTPUT" | grep -m1 "^Protocol:" | sed -E 's/.*\(([^)]+)\)/\1/' || echo "?")

    # Parse waveform duration: "Encoded: 123456 samples (2.57s)"
    DUR=$(echo "$OUTPUT" | grep -m1 "^Encoded:" | sed -E 's/.*\(([0-9.]+)s\)/\1/' || echo "")

    if echo "$OUTPUT" | grep -q "^SUCCESS"; then
        STATUS="PASS"
    else
        STATUS="FAIL"
    fi

    if [ "$STATUS" = "PASS" ] && [ -n "$DUR" ] && [ "$DUR" != "0" ]; then
        BPS=$(echo "scale=0; ${PAYLOAD_LENGTH} * 8 / ${DUR}" | bc 2>/dev/null || echo "0")
        KBPS=$(echo "scale=1; ${BPS} / 1000" | bc 2>/dev/null || echo "0")
    else
        BPS="0"
        KBPS="-"
    fi

    R_ID+=("$PROTO")
    R_NAME+=("$NAME")
    R_DUR+=("${DUR:-?}")
    R_STATUS+=("$STATUS")
    R_KBPS+=("$KBPS")
    R_BPS+=("$BPS")

    if [ "$STATUS" = "PASS" ]; then
        echo "${NAME} ${DUR}s -> ${KBPS} kbps"
    else
        echo "${NAME} ${DUR:-?}s -> FAIL"
    fi

    # Pause between tests
    LAST_IDX=$((${#PROTOCOLS[@]} - 1))
    if [ "$PROTO" != "${PROTOCOLS[$LAST_IDX]}" ]; then
        sleep "$PAUSE_BETWEEN"
    fi
done

# --- Summary Table ---
echo ""
echo "==== THROUGHPUT RESULTS ===="
echo "Payload: ${PAYLOAD_LENGTH} bytes | Device: ${OUTPUT_DEVICE}"
echo ""
printf "%-4s  %-16s  %8s  %-6s  %10s\n" "ID" "Protocol" "Duration" "Status" "Throughput"
printf "%-4s  %-16s  %8s  %-6s  %10s\n" "----" "----------------" "--------" "------" "----------"

BEST_ID=""
BEST_KBPS=0

for i in "${!R_ID[@]}"; do
    if [ "${R_STATUS[$i]}" = "PASS" ]; then
        THR="${R_KBPS[$i]} kbps"
        # Track best
        CMP=$(echo "${R_BPS[$i]} > ${BEST_KBPS}" | bc 2>/dev/null || echo "0")
        if [ "$CMP" = "1" ]; then
            BEST_KBPS="${R_BPS[$i]}"
            BEST_ID="$i"
        fi
    else
        THR="  FAIL"
    fi
    printf "%-4s  %-16s  %6ss  %-6s  %10s\n" \
        "${R_ID[$i]}" "${R_NAME[$i]}" "${R_DUR[$i]}" "${R_STATUS[$i]}" "$THR"
done

echo ""
if [ -n "$BEST_ID" ]; then
    echo "Best: Protocol ${R_ID[$BEST_ID]} (${R_NAME[$BEST_ID]}) at ${R_KBPS[$BEST_ID]} kbps"
fi
echo "Done."
