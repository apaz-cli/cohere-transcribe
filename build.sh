#!/bin/sh
set -e
cd "$(dirname "$0")"

# ---- Python venv ----
if [ ! -d .venv ]; then
    echo "Creating .venv ..."
    python3 -m venv .venv
fi

.venv/bin/python3 -c "import huggingface_hub, torch, transformers, sentencepiece, google.protobuf" 2>/dev/null || {
    echo "Installing Python dependencies ..."
    .venv/bin/pip install -q huggingface_hub torch transformers sentencepiece protobuf
}

# ---- Download model + generate support files ----
if [ ! -f vocab.txt ] || [ ! -f model_files/model.safetensors ] || [ ! -f model_offsets.h ]; then
    echo "Running convert.py ..."
    .venv/bin/python3 convert.py
fi

# ---- Embed support files as C headers ----
[ ! -f vocab_txt.h ] || [ vocab.txt -nt vocab_txt.h ] && python3 -c "
d = open('vocab.txt', 'rb').read().replace(b'\n', b'\x00')
print('unsigned char vocab_txt[] = {')
for i in range(0, len(d), 12):
    print('  ' + ', '.join('0x{:02x}'.format(b) for b in d[i:i+12]) + ',')
print('};')
print('unsigned int vocab_txt_len = {};'.format(len(d)))
" > vocab_txt.h

# ---- Detect available backends ----
HAS_CUDA=0
HAS_IRON=0
command -v nvcc   >/dev/null 2>&1 && HAS_CUDA=1
CUDA_PREFIX="$(dirname "$(dirname "$(command -v nvcc)")")"
[ -e /dev/amdxdna ]            && HAS_IRON=1

COMMON_SRCS="transcribe.c model.h model_offsets.h vocab_txt.h"

# build_backend <name> <flags> <obj> <src> <compiler_cmd> <bin> <libs>
# Rebuilds .o if source or shared headers are newer, rebuilds binary if .o or common sources are newer.
build_backend() {
    name="$1" flags="$2" obj="$3" src="$4" compiler_cmd="$5" bin="$6" libs="$7"

    src_newer=0
    [ ! -f "$obj" ] && src_newer=1
    for f in "$src" backends/backend.h backends/model_types.h; do
        [ "$f" -nt "$obj" ] 2>/dev/null && src_newer=1
    done

    if [ "$src_newer" -eq 1 ]; then
        echo "Compiling $name backend ..."
        eval "$compiler_cmd"
    fi

    rebuild=0
    [ ! -f "$bin" ]                         && rebuild=1
    [ transcribe.c   -nt "$bin" ] 2>/dev/null && rebuild=1
    [ model.h        -nt "$bin" ] 2>/dev/null && rebuild=1
    [ model_offsets.h -nt "$bin" ] 2>/dev/null && rebuild=1
    [ vocab_txt.h    -nt "$bin" ] 2>/dev/null && rebuild=1
    [ "$obj"         -nt "$bin" ] 2>/dev/null && rebuild=1

    if [ "$rebuild" -eq 1 ]; then
        echo "Linking $bin ..."
        gcc -O3 -march=native $flags -I. -o "$bin" transcribe.c "$obj" $libs
        echo "Built: ./$bin"
    fi
}

# ---- CPU (always) ----
build_backend cpu "" \
    backends/cpu/backend.o \
    backends/cpu/backend.c \
    "gcc -O3 -march=native -I. -c backends/cpu/backend.c -o backends/cpu/backend.o" \
    transcribe \
    "-lm -lpthread"

# ---- CUDA (if nvcc present) ----
if [ "$HAS_CUDA" -eq 1 ]; then
    # -arch=native picks the installed GPU's SM version (requires CUDA ≥ 11.6)
    build_backend cuda "-DBACKEND_CUDA" \
        backends/cuda/backend.o \
        backends/cuda/backend.cu \
        "nvcc -O3 -arch=native -DBACKEND_CUDA -I. -c backends/cuda/backend.cu -o backends/cuda/backend.o" \
        transcribe_cuda \
        "-L${CUDA_PREFIX}/lib64 -lcudart -lstdc++ -lm -lpthread"
fi

# ---- IRON / AMD XDNA (if /dev/amdxdna present) ----
if [ "$HAS_IRON" -eq 1 ]; then
    build_backend iron "-DBACKEND_IRON" \
        backends/iron/backend.o \
        backends/iron/backend.c \
        "gcc -O3 -march=native -DBACKEND_IRON -I. -c backends/iron/backend.c -o backends/iron/backend.o" \
        transcribe_iron \
        "-lm -lpthread"
fi

# ---- Test: run all binaries that were built ----
AUDIO="${1:-test_files/start_the_game_already.mp3}"
if [ -f "$AUDIO" ]; then
    for bin in ./transcribe ./transcribe_cuda ./transcribe_iron; do
        [ -f "$bin" ] || continue
        echo "--- $bin ---"
        "$bin" -v "$AUDIO"
    done
fi
