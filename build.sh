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

# ---- Compile ----
if [ ! -f transcribe ] || [ transcribe.c -nt transcribe ] || \
   [ model.h -nt transcribe ] || \
   [ model_offsets.h -nt transcribe ] || \
   [ vocab_txt.h -nt transcribe ]; then
    echo "Compiling ..."
    gcc -O3 -march=native -o transcribe transcribe.c -lm -lpthread # -fsanitize=address -g
    echo "Built: ./transcribe"
fi

# ---- Test ----
AUDIO="${1:-test_files/start_the_game_already.mp3}"
if [ -f "$AUDIO" ]; then
    ./transcribe -v "$AUDIO"
fi
