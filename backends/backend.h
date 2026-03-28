/* backends/backend.h — interface contract for all compute backends.
 *
 * Include order: model.h includes this at the bottom, after ELayer/DLayer/Weights
 * are defined.  Backend .c/.cu files include model_types.h first, then this file.
 *
 * Weight pointers are always const uint16_t * (host or device BF16).
 * CUDA backend casts to __nv_bfloat16 * internally.
 */
#ifndef BACKENDS_BACKEND_H
#define BACKENDS_BACKEND_H

#include <stdint.h>
#include <stddef.h>

/* ============================================================ */
/* Lifecycle                                                     */
/* ============================================================ */

void  backend_init(int device_id);
void  backend_destroy(void);

/* Number of parallel execution units (GPUs for CUDA, CPU cores for CPU/IRON).
 * run_chunks_parallel uses this to cap the thread count. */
int   backend_num_devices(void);

/* Release all CUDA resources owned by the calling thread (stream, arena,
 * token buffer, graph cache).  No-op on CPU/IRON.  Must be called from each
 * worker thread before it exits; backend_destroy() covers the main thread. */
void  backend_thread_cleanup(void);

/* Upload the entire safetensors data blob to device memory.
 * No-op on CPU.  Must be called before patch_weights_to_device(). */
void  backend_weights_upload_blob(const void *host_base, size_t bytes);

/* Given a pointer inside the mmap'd host blob, return the corresponding
 * device pointer (or the same pointer unchanged on CPU). */
void *backend_device_ptr(const void *host_ptr);

/* Arena: device or host flat float region replacing malloc/free in transcribe_chunk. */
float *backend_arena_alloc(size_t floats);
void   backend_arena_free(float *arena);

/* Transfers: no-op on CPU. */
void backend_htod(float *dev_dst, const float *host_src, size_t floats);
void backend_dtoh(void *host_dst, const void *dev_src, size_t bytes);

/* ============================================================ */
/* Compute — natural kernel boundaries                           */
/* ============================================================ */

void backend_layernorm(float *dst, const float *src,
                       const uint16_t *w, const uint16_t *b, int T, int D);

/* Linear family: each is a GEMM + distinct epilogue. */
void backend_linear(float *y, const float *x,
                    const uint16_t *W, const uint16_t *b, int T, int in_d, int out_d);
void backend_linear_silu(float *y, const float *x,
                         const uint16_t *W, const uint16_t *b, int T, int in_d, int out_d);
void backend_linear_relu(float *y, const float *x,
                         const uint16_t *W, const uint16_t *b, int T, int in_d, int out_d);
/* acc[t] += scale * (W[t] @ x[t] + b) */
void backend_linear_fmadd(float *acc, float scale, const float *x,
                           const uint16_t *W, const uint16_t *b, int T, int in_d, int out_d);

/* Element-wise residual add: dst[i] += src[i] */
void backend_add_inplace(float *dst, const float *src, int n);

/* Encoder subsampling: mel (N_MELS, T_mel) → x_enc (T_enc, ENC_D). */
void backend_encoder_subsampling(const float *mel, int T_mel,
                                  float *x_enc, int T_enc, float *work,
                                  const Weights *W);

/* Relative sinusoidal positional encoding: writes (2T-1, ENC_D) floats to pe. */
void backend_make_rel_pos_emb(float *pe, int T);

/* Relative-position multi-head self-attention (encoder).
 * x (T, ENC_D) updated in-place. */
void backend_rel_pos_mhsa(float *x, const float *pos_emb,
                           int T, const ELayer *L, float *work);

/* Conformer conv module: x (T, ENC_D) updated in-place.
 * pw1 linear → GLU → depthwise conv1d → batchnorm → silu → pw2 linear. */
void backend_conformer_conv(float *x, int T, const ELayer *L, float *work);

/* Scaled dot-product attention (decoder self and cross).
 * mask is (T_q, T_k) additive bias, or NULL (no mask). */
void backend_sdp_attn(float *out, const float *q, const float *k, const float *v,
                      const float *mask, int T_q, int T_k, float *work);

/* Token embedding + positional embedding + layernorm → h (S, DEC_D).
 * tokens[0..S-1] are embedded at absolute positions [pos_start..pos_start+S-1].
 * tokens is a host array; backend copies it to device if needed. */
void backend_embed(float *h, const int *tokens, int S, int pos_start,
                   const uint16_t *etw, const uint16_t *epw,
                   const uint16_t *elnw, const uint16_t *elnb);

/* Fill S×S causal mask: mask[q][k] = -1e9 if k > q, else 0. */
void backend_fill_causal_mask(float *mask, int S);

/* LM head: logits_out[v] = hdb[v] + last[0..DEC_D-1] · etw[v*DEC_D..].
 * Writes VOCAB floats to logits_out (may be device memory). */
void backend_lm_head(float *logits_out, const float *last,
                     const uint16_t *etw, const uint16_t *hdb);

/* ============================================================ */
/* CUDA Graph capture/replay (no-ops on CPU/IRON)               */
/* ============================================================ */

/* Tags that identify which inference region is being captured. */
#define GRAPH_TAG_ENCODER    0
#define GRAPH_TAG_PRECOMPUTE 1
#define GRAPH_TAG_DECODER    2

/* Stage S token ids into the backend's persistent device token buffer.
 * Must be called before backend_graph_begin_capture / backend_graph_replay
 * for any region that calls backend_embed.
 * No-op on CPU/IRON (tokens are already host-accessible). */
void backend_upload_tokens(const int *host_tokens, int S);

/* Look up a cached graph by (tag, T_enc, S).
 * CUDA: if found, launches the graph on g_stream and returns 1.
 *       Returns 0 if not found — caller must run kernels and record them
 *       with backend_graph_begin_capture / backend_graph_end_capture.
 * CPU/IRON: always returns 0. */
int  backend_graph_replay(int tag, int T_enc, int S);

/* Begin stream capture.  Subsequent backend_* kernel calls are recorded but
 * not executed until backend_graph_end_capture launches the graph.
 * No-op on CPU/IRON. */
void backend_graph_begin_capture(int tag);

/* End capture, instantiate the graph, cache it under (tag, T_enc, S), and
 * launch it once so the first (recording) pass also produces results.
 * No-op on CPU/IRON. */
void backend_graph_end_capture(int tag, int T_enc, int S);

/* ============================================================ */
/* Device-side sequence counter for single-graph decode          */
/* ============================================================ */

/* Set the device-side decode counter to n.  Call once before the decode
 * loop, after the prefill step has populated the KV cache for n_prompt
 * positions.  On CPU/IRON this is a plain thread-local write. */
void backend_decode_set_S(int n);

/* Increment the device-side counter by 1.  Must be the first kernel
 * captured inside each decode graph, so all subsequent kernels see the
 * updated value (*g_dev_S == new sequence length including this token). */
void backend_decode_inc_S(void);

/* Linear projection (T_q=1) that writes its output row into a KV cache
 * buffer at the position indicated by the device-side counter:
 *   cache[(*g_dev_S - 1) * out_d .. (*g_dev_S - 1) * out_d + out_d - 1]
 * Equivalent to backend_linear with T=1 but avoids capturing a
 * per-step host-computed pointer inside the graph. */
void backend_linear_to_kvcache(float *cache, const float *x,
                                const uint16_t *W, const uint16_t *b,
                                int in_d, int out_d);

/* Scaled dot-product attention for T_q=1, T_k=*g_dev_S, no mask.
 * The grid is over-provisioned to MAX_SEQ; inactive lanes self-terminate.
 * work must hold at least MAX_SEQ floats for the score buffer. */
void backend_sdp_attn_devS(float *out, const float *q,
                            const float *k, const float *v, float *work);

/* Embed the single token staged in the backend token buffer at absolute
 * position (*g_dev_S - 1), writing one DEC_D-wide row into h.
 * The position is read from the device-side counter at kernel execution
 * time, so one captured graph serves all decode steps. */
void backend_embed_decode(float *h,
                          const uint16_t *etw, const uint16_t *epw,
                          const uint16_t *elnw, const uint16_t *elnb);

#endif /* BACKENDS_BACKEND_H */
