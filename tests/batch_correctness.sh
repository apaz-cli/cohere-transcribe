#!/bin/bash
set -e
cd "$(dirname "$0")/.."
AUDIO=test_files/jeff.webm
OUT=test_files

echo "=== Running GPU batch-size 1 ==="
./transcribe_cuda --chunk --batch-size 1 "$AUDIO" 2>/dev/null > "$OUT/jeff_gpu.txt"
echo "=== Running GPU batch-size 64 ==="
./transcribe_cuda --chunk --batch-size 64 "$AUDIO" 2>/dev/null > "$OUT/jeff_batched_gpu.txt"
echo "=== Running CPU batch-size 1 ==="
./transcribe --chunk --batch-size 1 "$AUDIO" 2>/dev/null > "$OUT/jeff_cpu.txt"
echo "=== Running CPU batch-size 64 ==="
./transcribe --chunk --batch-size 64 "$AUDIO" 2>/dev/null > "$OUT/jeff_batched_cpu.txt"

echo ""
echo "=== Diffs ==="
PASS=1
for pair in \
    "jeff_gpu.txt jeff_batched_gpu.txt" \
    "jeff_gpu.txt jeff_cpu.txt" \
    "jeff_gpu.txt jeff_batched_cpu.txt"; do
    a=$(echo $pair | cut -d' ' -f1)
    b=$(echo $pair | cut -d' ' -f2)
    if diff -q "$OUT/$a" "$OUT/$b" > /dev/null 2>&1; then
        echo "PASS: $a == $b"
    else
        echo "FAIL: $a != $b"
        diff "$OUT/$a" "$OUT/$b" | head -10
        PASS=0
    fi
done

echo ""
if [ $PASS -eq 1 ]; then
    echo "ALL PASS"
    exit 0
else
    echo "FAILURES DETECTED"
    exit 1
fi
