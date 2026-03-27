#!/usr/bin/env python3
"""
Download cohere-transcribe-03-2026 weights and save support files:
  vocab.txt          — one SentencePiece token per line
  decode_config.json — prompt token ids and EOS id
  model_offsets.h    — precomputed tensor byte offsets for EMBEDDED_MODEL builds

The model weights stay in model_files/model.safetensors (BF16) and are
read directly at inference time by transcribe.c.

Run from the project root inside the .venv.
"""
import json, os, struct, textwrap

from huggingface_hub import hf_hub_download

MODEL_DIR = "model_files"
REPO      = "CohereLabs/cohere-transcribe-03-2026"

# ---- Download model files if needed ----
sf_path = os.path.join(MODEL_DIR, "model.safetensors")
for fname, desc in [
    ("model.safetensors",            "model weights (~4.1 GB)"),
    ("tokenizer.model",              "tokenizer model"),
    ("special_tokens_map.json",      "special tokens map"),
    ("tokenization_cohere_asr.py",   "tokenizer code"),
]:
    if not os.path.exists(os.path.join(MODEL_DIR, fname)):
        print(f"Downloading {desc} ...")
        hf_hub_download(REPO, fname, local_dir=MODEL_DIR)

# ---- Load tokenizer ----
print("Loading tokenizer ...")
from model_files.tokenization_cohere_asr import CohereAsrTokenizer

tok = CohereAsrTokenizer.from_pretrained(MODEL_DIR)

# ---- Save vocab ----
print("Saving vocab.txt ...")
spm_n = tok.sp_model.get_piece_size()
added = tok.added_tokens_decoder   # {id: AddedToken}
max_id = max(list(added.keys()) + [spm_n - 1])

vocab = [""] * (max_id + 1)
for i in range(spm_n):
    vocab[i] = tok.sp_model.id_to_piece(i)
for tid, t in added.items():
    vocab[tid] = t.content

with open("vocab.txt", "w", encoding="utf-8") as f:
    for p in vocab:
        f.write(p.replace("\n", "\\n") + "\n")
print(f"  vocab.txt: {len(vocab)} tokens")

# ---- Generate model_offsets.h ----
print("Generating model_offsets.h ...")

ENC_N = 48
DEC_N = 8

ENC_FIELDS = [
    ("ff1_lnw",  "norm_feed_forward1.weight"),
    ("ff1_lnb",  "norm_feed_forward1.bias"),
    ("ff1_w1",   "feed_forward1.linear1.weight"),
    ("ff1_b1",   "feed_forward1.linear1.bias"),
    ("ff1_w2",   "feed_forward1.linear2.weight"),
    ("ff1_b2",   "feed_forward1.linear2.bias"),
    ("sa_lnw",   "norm_self_att.weight"),
    ("sa_lnb",   "norm_self_att.bias"),
    ("sa_qw",    "self_attn.linear_q.weight"),
    ("sa_qb",    "self_attn.linear_q.bias"),
    ("sa_kw",    "self_attn.linear_k.weight"),
    ("sa_kb",    "self_attn.linear_k.bias"),
    ("sa_vw",    "self_attn.linear_v.weight"),
    ("sa_vb",    "self_attn.linear_v.bias"),
    ("sa_pw",    "self_attn.linear_pos.weight"),
    ("sa_pbu",   "self_attn.pos_bias_u"),
    ("sa_pbv",   "self_attn.pos_bias_v"),
    ("sa_ow",    "self_attn.linear_out.weight"),
    ("sa_ob",    "self_attn.linear_out.bias"),
    ("cv_lnw",   "norm_conv.weight"),
    ("cv_lnb",   "norm_conv.bias"),
    ("cv_pw1w",  "conv.pointwise_conv1.weight"),
    ("cv_pw1b",  "conv.pointwise_conv1.bias"),
    ("cv_dwb",   "conv.depthwise_conv.bias"),
    ("cv_dww",   "conv.depthwise_conv.weight"),
    ("cv_bnmean","conv.batch_norm.running_mean"),
    ("cv_bnvar", "conv.batch_norm.running_var"),
    ("cv_bnw",   "conv.batch_norm.weight"),
    ("cv_bnb",   "conv.batch_norm.bias"),
    ("cv_pw2w",  "conv.pointwise_conv2.weight"),
    ("cv_pw2b",  "conv.pointwise_conv2.bias"),
    ("ff2_lnw",  "norm_feed_forward2.weight"),
    ("ff2_lnb",  "norm_feed_forward2.bias"),
    ("ff2_w1",   "feed_forward2.linear1.weight"),
    ("ff2_b1",   "feed_forward2.linear1.bias"),
    ("ff2_w2",   "feed_forward2.linear2.weight"),
    ("ff2_b2",   "feed_forward2.linear2.bias"),
    ("out_lnw",  "norm_out.weight"),
    ("out_lnb",  "norm_out.bias"),
]

DEC_FIELDS = [
    ("sa_lnw",  "layer_norm_1.weight"),
    ("sa_lnb",  "layer_norm_1.bias"),
    ("sa_qw",   "first_sub_layer.query_net.weight"),
    ("sa_qb",   "first_sub_layer.query_net.bias"),
    ("sa_kw",   "first_sub_layer.key_net.weight"),
    ("sa_kb",   "first_sub_layer.key_net.bias"),
    ("sa_vw",   "first_sub_layer.value_net.weight"),
    ("sa_vb",   "first_sub_layer.value_net.bias"),
    ("sa_ow",   "first_sub_layer.out_projection.weight"),
    ("sa_ob",   "first_sub_layer.out_projection.bias"),
    ("ca_lnw",  "layer_norm_2.weight"),
    ("ca_lnb",  "layer_norm_2.bias"),
    ("ca_qw",   "second_sub_layer.query_net.weight"),
    ("ca_qb",   "second_sub_layer.query_net.bias"),
    ("ca_kw",   "second_sub_layer.key_net.weight"),
    ("ca_kb",   "second_sub_layer.key_net.bias"),
    ("ca_vw",   "second_sub_layer.value_net.weight"),
    ("ca_vb",   "second_sub_layer.value_net.bias"),
    ("ca_ow",   "second_sub_layer.out_projection.weight"),
    ("ca_ob",   "second_sub_layer.out_projection.bias"),
    ("ffn_lnw", "layer_norm_3.weight"),
    ("ffn_lnb", "layer_norm_3.bias"),
    ("ffn_w1",  "third_sub_layer.dense_in.weight"),
    ("ffn_b1",  "third_sub_layer.dense_in.bias"),
    ("ffn_w2",  "third_sub_layer.dense_out.weight"),
    ("ffn_b2",  "third_sub_layer.dense_out.bias"),
]

FIXED = [
    ("s0w",  "encoder.pre_encode.conv.0.weight"),
    ("s0b",  "encoder.pre_encode.conv.0.bias"),
    ("s2w",  "encoder.pre_encode.conv.2.weight"),
    ("s2b",  "encoder.pre_encode.conv.2.bias"),
    ("s3w",  "encoder.pre_encode.conv.3.weight"),
    ("s3b",  "encoder.pre_encode.conv.3.bias"),
    ("s5w",  "encoder.pre_encode.conv.5.weight"),
    ("s5b",  "encoder.pre_encode.conv.5.bias"),
    ("s6w",  "encoder.pre_encode.conv.6.weight"),
    ("s6b",  "encoder.pre_encode.conv.6.bias"),
    ("sow",  "encoder.pre_encode.out.weight"),
    ("sob",  "encoder.pre_encode.out.bias"),
    ("prw",  "encoder_decoder_proj.weight"),
    ("prb",  "encoder_decoder_proj.bias"),
    ("etw",  "transf_decoder._embedding.token_embedding.weight"),
    ("epw",  "transf_decoder._embedding.position_embedding.pos_enc"),
    ("elnw", "transf_decoder._embedding.layer_norm.weight"),
    ("elnb", "transf_decoder._embedding.layer_norm.bias"),
    ("dlnw", "transf_decoder._decoder.final_layer_norm.weight"),
    ("dlnb", "transf_decoder._decoder.final_layer_norm.bias"),
    ("hdb",  "log_softmax.mlp.layer0.bias"),
]

# ---- Reorder safetensors to match inference access order ----
# The file as downloaded stores tensors in Python dict (alphabetical) key order,
# which scatters e.g. encoder.layers.10 before encoder.layers.2.
# Rewriting once here ensures sequential reads during inference.

def build_access_order():
    order = []
    # Subsampling (before encoder layers)
    for _, name in FIXED[:12]:   # s0w..sob
        order.append(name)
    # Encoder layers 0..ENC_N-1, each field in forward-pass order
    for i in range(ENC_N):
        for _, suffix in ENC_FIELDS:
            order.append(f"encoder.layers.{i}.{suffix}")
    # enc→dec projection
    for _, name in FIXED[12:14]:  # prw, prb
        order.append(name)
    # precompute_ca_kv runs before any decoder step: group ca_kw/kb/vw/vb first
    dec_prefix = "transf_decoder._decoder.layers"
    ca_kv = [s for f, s in DEC_FIELDS if f in ("ca_kw", "ca_kb", "ca_vw", "ca_vb")]
    for i in range(DEC_N):
        for suffix in ca_kv:
            order.append(f"{dec_prefix}.{i}.{suffix}")
    # Decoder embedding + layer norm
    for _, name in FIXED[14:18]:  # etw, epw, elnw, elnb
        order.append(name)
    # Per-layer decoder step weights (sa_*, ca_ln/q/o, ffn_*)
    dec_step = [s for f, s in DEC_FIELDS if f not in ("ca_kw", "ca_kb", "ca_vw", "ca_vb")]
    for i in range(DEC_N):
        for suffix in dec_step:
            order.append(f"{dec_prefix}.{i}.{suffix}")
    # Final layer norm + head bias
    for _, name in FIXED[18:]:    # dlnw, dlnb, hdb
        order.append(name)
    return order

def reorder_safetensors(path):
    with open(path, "rb") as f:
        hs = struct.unpack_from("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hs))

    data_start_old = 8 + hs
    access_order = build_access_order()
    known = set(access_order)
    # Append any tensors in the file not covered by the access order
    for name in hdr:
        if name != "__metadata__" and name not in known:
            access_order.append(name)

    # Compute new data_offsets
    new_hdr = {}
    if "__metadata__" in hdr:
        new_hdr["__metadata__"] = hdr["__metadata__"]
    off = 0
    for name in access_order:
        info = hdr[name]
        size = info["data_offsets"][1] - info["data_offsets"][0]
        new_hdr[name] = {"dtype": info["dtype"], "shape": info["shape"],
                         "data_offsets": [off, off + size]}
        off += size

    new_hdr_bytes = json.dumps(new_hdr, separators=(',', ':')).encode()
    # safetensors requires header length to be a multiple of 8
    pad = (8 - len(new_hdr_bytes) % 8) % 8
    new_hdr_bytes += b' ' * pad

    tmp = path + ".tmp"
    CHUNK = 64 << 20  # 64 MB
    with open(path, "rb") as src, open(tmp, "wb") as dst:
        dst.write(struct.pack("<Q", len(new_hdr_bytes)))
        dst.write(new_hdr_bytes)
        for name in access_order:
            s, e = hdr[name]["data_offsets"]
            src.seek(data_start_old + s)
            remaining = e - s
            while remaining > 0:
                n = min(CHUNK, remaining)
                dst.write(src.read(n))
                remaining -= n
    os.replace(tmp, path)
    print(f"  Reordered {len(access_order)} tensors into inference access order")

print("Reordering model weights ...")
reorder_safetensors(sf_path)

with open(sf_path, "rb") as f:
    hs = struct.unpack_from("<Q", f.read(8))[0]
    sf_header = json.loads(f.read(hs))

data_start = 8 + hs

def sf_off(name):
    return data_start + sf_header[name]["data_offsets"][0]

def fmt_array(name, values, n):
    vals = ", ".join(f"{v}UL" for v in values)
    body = textwrap.fill(vals, width=100, subsequent_indent="    ")
    return f"static const size_t {name}[{n}] = {{\n    {body}\n}};"

out = [
    "/* Auto-generated by convert.py — do not edit */",
    f"/* safetensors data section starts at file byte {data_start} */",
    "",
    "/* Fixed tensors */",
]
for field, name in FIXED:
    out.append(f"#define SFOFF_{field} {sf_off(name)}UL")

out += ["", "/* Encoder layer offsets [ENC_N] */"]
for field, suffix in ENC_FIELDS:
    values = [sf_off(f"encoder.layers.{i}.{suffix}") for i in range(ENC_N)]
    out.append(fmt_array(f"SFOFF_enc_{field}", values, ENC_N))

out += ["", "/* Decoder layer offsets [DEC_N] */"]
for field, suffix in DEC_FIELDS:
    values = [sf_off(f"transf_decoder._decoder.layers.{i}.{suffix}") for i in range(DEC_N)]
    out.append(fmt_array(f"SFOFF_dec_{field}", values, DEC_N))

out.append("")
with open("model_offsets.h", "w") as f:
    f.write("\n".join(out))
print(f"  model_offsets.h: {len(ENC_FIELDS)*ENC_N + len(DEC_FIELDS)*DEC_N + len(FIXED)} offsets")

print("\nConversion complete.")
