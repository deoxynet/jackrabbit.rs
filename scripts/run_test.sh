#!/bin/bash

# --- Test Configuration ---
# Protocol ID: Choose from 0 to 15.
# Protocol 14 (WIRED_MAX) is the fastest reliable protocol found on your Focusrite 2i2.
PROTOCOL_ID=13

# Payload: The data to transmit. You can adjust the length of the payload.
# Using 4000 'a' characters as a default.
PAYLOAD_LENGTH=4000
PAYLOAD=$(printf 'a%.0s' {1..$PAYLOAD_LENGTH})

# Audio Devices: Replace "Scarlett" with the full or partial name of your input/output device if needed.
# For Focusrite Scarlett 2i2, "Scarlett" usually works for both.
OUTPUT_DEVICE="Scarlett"
INPUT_DEVICE="Scarlett"
# --------------------------

echo "Running OTA loopback test with Protocol ${PROTOCOL_ID} (${PAYLOAD_LENGTH} bytes payload) on ${OUTPUT_DEVICE}/${INPUT_DEVICE}..."

cargo run --release --example ota_loopback -- \
  --protocol "${PROTOCOL_ID}" \
  --payload "${PAYLOAD}" \
  --output-device "${OUTPUT_DEVICE}" \
  --input-device "${INPUT_DEVICE}"

echo "Test finished."
