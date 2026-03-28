/* backends/cpu/backend.c — CPU backend implementation.
 *
 * All compute functions are the existing scalar C implementations from model.h,
 * moved here verbatim and renamed to backend_*.  CPU output is bit-identical to
 * the pre-refactor binary.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "backend.h"

/* ============================================================ */
/* Lifecycle                                                     */
/* ============================================================ */

void backend_init(int device_id) { (void)device_id; }
void backend_destroy(void) {}
void backend_thread_cleanup(void) {}
int  backend_num_devices(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? (int)n : 1;
}

/* Token buffer: backend_embed_decode reads from here. */
static _Thread_local int g_cpu_tokens[MAX_SEQ];
void backend_upload_tokens(const int *t, int S) { memcpy(g_cpu_tokens, t, (size_t)S * sizeof(int)); }

/* Per-item batch state mirrors (set by backend_embed_decode_batch / backend_upload_T_enc). */
#define MAX_BATCH 256
static _Thread_local int g_cpu_S_batch[MAX_BATCH];
static _Thread_local int g_cpu_T_enc_batch[MAX_BATCH];
int  backend_graph_replay(int tag, int T, int S, int B) { (void)tag; (void)T; (void)S; (void)B; return 0; }
void backend_graph_begin_capture(int tag) { (void)tag; }
void backend_graph_end_capture(int tag, int T, int S, int B) { (void)tag; (void)T; (void)S; (void)B; }

void backend_weights_upload_blob(const void *host_base, size_t bytes) {
    (void)host_base; (void)bytes;
}
void *backend_device_ptr(const void *host_ptr) { return (void *)host_ptr; }

float *backend_arena_alloc(size_t floats) {
    float *p = malloc(floats * sizeof(float));
    if (!p) { fputs("OOM\n", stderr); exit(1); }
    return p;
}
void backend_arena_free(float *arena) { free(arena); }

void backend_htod(float *dev_dst, const float *host_src, size_t floats) {
    memcpy(dev_dst, host_src, floats * sizeof(float));
}
void backend_dtoh(void *host_dst, const void *dev_src, size_t bytes) {
    memcpy(host_dst, dev_src, bytes);
}

/* ============================================================ */
/* Internal helpers (not in interface)                           */
/* ============================================================ */

static void silu(float *x, int n) {
    for (int i = 0; i < n; i++) x[i] = x[i] / (1.0f + expf(-x[i]));
}
static void relu(float *x, int n) {
    for (int i = 0; i < n; i++) if (x[i] < 0.0f) x[i] = 0.0f;
}
static void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

static void conv2d(float *out, const float *in, const uint16_t *Wf, const uint16_t *bf,
                   int in_c, int in_h, int in_w,
                   int out_c, int kH, int kW, int stride, int pad, int groups) {
    int out_h = (in_h + 2*pad - kH) / stride + 1;
    int out_w = (in_w + 2*pad - kW) / stride + 1;
    int ic_g  = in_c  / groups, oc_g = out_c / groups;
    for (int g = 0; g < groups; g++)
    for (int oc = 0; oc < oc_g; oc++) {
        int oc_gl = g*oc_g + oc;
        for (int oh = 0; oh < out_h; oh++)
        for (int ow = 0; ow < out_w; ow++) {
            float s = bf ? bf16(bf[oc_gl]) : 0.0f;
            for (int ic = 0; ic < ic_g; ic++) {
                int ic_gl = g*ic_g + ic;
                for (int kh = 0; kh < kH; kh++) {
                    int ih = oh*stride - pad + kh;
                    if (ih < 0 || ih >= in_h) continue;
                    for (int kw = 0; kw < kW; kw++) {
                        int iw = ow*stride - pad + kw;
                        if (iw < 0 || iw >= in_w) continue;
                        s += in[ic_gl*(in_h*in_w) + ih*in_w + iw]
                           * bf16(Wf[(oc_gl*ic_g + ic)*kH*kW + kh*kW + kw]);
                    }
                }
            }
            out[oc_gl*(out_h*out_w) + oh*out_w + ow] = s;
        }
    }
}

/* ============================================================ */
/* Kernel-boundary functions                                     */
/* ============================================================ */

void backend_layernorm(float *dst, const float *src,
                       const uint16_t *w, const uint16_t *b, int T, int D) {
    const float eps = 1e-5f;
    for (int t = 0; t < T; t++) {
        const float *sr = src + t * D;
        float       *dr = dst + t * D;
        float mean = 0, var = 0;
        for (int d = 0; d < D; d++) mean += sr[d];
        mean /= D;
        for (int d = 0; d < D; d++) { float v = sr[d] - mean; var += v * v; }
        float is = 1.0f / sqrtf(var / D + eps);
        for (int d = 0; d < D; d++) dr[d] = (sr[d]-mean)*is*bf16(w[d]) + bf16(b[d]);
    }
}

void backend_linear(float *y, const float *x,
                    const uint16_t *Wb, const uint16_t *bb,
                    int T, int in_d, int out_d) {
    for (int t = 0; t < T; t++) {
        const float    *xr = x  + t * in_d;
        float          *yr = y  + t * out_d;
        for (int j = 0; j < out_d; j++) {
            float s = bb ? bf16(bb[j]) : 0.0f;
            const uint16_t *wr = Wb + j * in_d;
            for (int k = 0; k < in_d; k++) s += xr[k] * bf16(wr[k]);
            yr[j] = s;
        }
    }
}

void backend_linear_silu(float *y, const float *x,
                         const uint16_t *Wb, const uint16_t *bb,
                         int T, int in_d, int out_d) {
    for (int t = 0; t < T; t++) {
        const float    *xr = x + t * in_d;
        float          *yr = y + t * out_d;
        for (int j = 0; j < out_d; j++) {
            float s = bf16(bb[j]);
            const uint16_t *wr = Wb + j * in_d;
            for (int k = 0; k < in_d; k++) s += xr[k] * bf16(wr[k]);
            yr[j] = s / (1.0f + expf(-s));
        }
    }
}

void backend_linear_relu(float *y, const float *x,
                         const uint16_t *Wb, const uint16_t *bb,
                         int T, int in_d, int out_d) {
    for (int t = 0; t < T; t++) {
        const float    *xr = x + t * in_d;
        float          *yr = y + t * out_d;
        for (int j = 0; j < out_d; j++) {
            float s = bf16(bb[j]);
            const uint16_t *wr = Wb + j * in_d;
            for (int k = 0; k < in_d; k++) s += xr[k] * bf16(wr[k]);
            yr[j] = s > 0.0f ? s : 0.0f;
        }
    }
}

void backend_linear_fmadd(float *acc, float scale, const float *x,
                           const uint16_t *Wb, const uint16_t *bb,
                           int T, int in_d, int out_d) {
    for (int t = 0; t < T; t++) {
        const float    *xr = x   + t * in_d;
        float          *ar = acc + t * out_d;
        for (int j = 0; j < out_d; j++) {
            float s = bb ? bf16(bb[j]) : 0.0f;
            const uint16_t *wr = Wb + j * in_d;
            for (int k = 0; k < in_d; k++) s += xr[k] * bf16(wr[k]);
            ar[j] += scale * s;
        }
    }
}

void backend_add_inplace(float *dst, const float *src, int n) {
    for (int i = 0; i < n; i++) dst[i] += src[i];
}

void backend_encoder_subsampling(const float *mel, int T_mel, int T_mel_stride,
                                  float *x_enc, int T_enc, float *work,
                                  const Weights *W) {
    const int T1 = (T_mel+1)/2, T2 = (T1+1)/2;
    const int F1 = 64, F2 = 32, F3 = 16;
    const int in_feat = SUB_CH * F3;
    const size_t B0 = (size_t)SUB_CH * T1 * F1;
    const size_t B1 = (size_t)SUB_CH * T2 * F2;
    const size_t B2 = (size_t)SUB_CH * T_enc * F3;

    float *buf0  = work;
    float *mel_t = work + B0;

    for (int t = 0; t < T_mel; t++)
        for (int m = 0; m < N_MELS; m++)
            mel_t[t * N_MELS + m] = mel[m * T_mel_stride + t];

    conv2d(buf0, mel_t, W->s0w, W->s0b, 1, T_mel, N_MELS, SUB_CH, 3, 3, 2, 1, 1);
    relu(buf0, SUB_CH * T1 * F1);

    float *buf1 = work + B0;
    conv2d(buf1, buf0, W->s2w, W->s2b, SUB_CH, T1, F1, SUB_CH, 3, 3, 2, 1, SUB_CH);

    float *buf1b = work;
    conv2d(buf1b, buf1, W->s3w, W->s3b, SUB_CH, T2, F2, SUB_CH, 1, 1, 1, 0, 1);
    relu(buf1b, SUB_CH * T2 * F2);

    float *buf2 = work + B1;
    conv2d(buf2, buf1b, W->s5w, W->s5b, SUB_CH, T2, F2, SUB_CH, 3, 3, 2, 1, SUB_CH);

    float *buf2b = work;
    conv2d(buf2b, buf2, W->s6w, W->s6b, SUB_CH, T_enc, F3, SUB_CH, 1, 1, 1, 0, 1);
    relu(buf2b, SUB_CH * T_enc * F3);

    float *flat = work + B2;
    for (int t = 0; t < T_enc; t++)
        for (int ch = 0; ch < SUB_CH; ch++)
            for (int f = 0; f < F3; f++)
                flat[t * in_feat + ch * F3 + f] = buf2b[ch * T_enc * F3 + t * F3 + f];

    backend_linear(x_enc, flat, W->sow, W->sob, T_enc, in_feat, ENC_D);
}

void backend_make_rel_pos_emb(float *pe, int T) {
    int L = 2 * T - 1;
    float log10k = logf(10000.0f);
    for (int p = 0; p < L; p++) {
        float pos = (float)(T - 1 - p);
        for (int k = 0; k < ENC_D / 2; k++) {
            float dt = expf(-(float)(2*k) * log10k / ENC_D);
            pe[p * ENC_D + 2*k]     = sinf(pos * dt);
            pe[p * ENC_D + 2*k + 1] = cosf(pos * dt);
        }
    }
}

void backend_rel_pos_mhsa(float *x, const float *pos_emb,
                           int T, const ELayer *L, float *work) {
    const int pos_len = 2 * T - 1;
    float *Q       = work;
    float *K       = Q + (size_t)T * ENC_D;
    float *V       = K + (size_t)T * ENC_D;
    float *P       = V + (size_t)T * ENC_D;
    float *shared  = P + (size_t)pos_len * ENC_D;
    float *attn_out= shared  + (size_t)T * ENC_D;

    float *scores    = shared;
    float *scores_bd = shared + (size_t)T * T;

    backend_linear(Q, x,       L->sa_qw, L->sa_qb, T,       ENC_D, ENC_D);
    backend_linear(K, x,       L->sa_kw, L->sa_kb, T,       ENC_D, ENC_D);
    backend_linear(V, x,       L->sa_vw, L->sa_vb, T,       ENC_D, ENC_D);
    backend_linear(P, pos_emb, L->sa_pw, NULL,     pos_len,  ENC_D, ENC_D);

    const float scale = 1.0f / sqrtf((float)ENC_DK);

    memset(attn_out, 0, (size_t)T * ENC_D * sizeof(float));

    for (int h = 0; h < ENC_H; h++) {
        for (int q = 0; q < T; q++)
        for (int k = 0; k < T; k++) {
            float s = 0;
            for (int d = 0; d < ENC_DK; d++)
                s += (Q[q*ENC_D + h*ENC_DK+d] + bf16(L->sa_pbu[h*ENC_DK+d]))
                   * K[k*ENC_D + h*ENC_DK+d];
            scores[q*T + k] = s * scale;
        }
        for (int q = 0; q < T; q++)
        for (int p = 0; p < pos_len; p++) {
            float s = 0;
            for (int d = 0; d < ENC_DK; d++)
                s += (Q[q*ENC_D + h*ENC_DK+d] + bf16(L->sa_pbv[h*ENC_DK+d]))
                   * P[p*ENC_D + h*ENC_DK+d];
            scores_bd[q*pos_len + p] = s;
        }
        for (int q = 0; q < T; q++)
        for (int k = 0; k < T; k++)
            scores[q*T + k] += scores_bd[q*pos_len + k + (T-1) - q] * scale;

        for (int q = 0; q < T; q++) softmax(scores + q*T, T);

        for (int q = 0; q < T; q++)
        for (int d = 0; d < ENC_DK; d++) {
            float s = 0;
            for (int k = 0; k < T; k++)
                s += scores[q*T + k] * V[k*ENC_D + h*ENC_DK+d];
            attn_out[q*ENC_D + h*ENC_DK+d] = s;
        }
    }
    backend_linear(x, attn_out, L->sa_ow, L->sa_ob, T, ENC_D, ENC_D);
}

void backend_conformer_conv(float *x, int T, const ELayer *L, float *work) {
    float *pw1_out = work;
    backend_linear(pw1_out, x, L->cv_pw1w, L->cv_pw1b, T, ENC_D, 2*ENC_D);

    float *glu_out = work + (size_t)T * 2 * ENC_D;
    for (int t = 0; t < T; t++)
    for (int d = 0; d < ENC_D; d++) {
        float a = pw1_out[t*2*ENC_D + d];
        float b = pw1_out[t*2*ENC_D + d + ENC_D];
        glu_out[t*ENC_D + d] = a / (1.0f + expf(-b));
    }

    const int pad_dw = (ENC_CK - 1) / 2;
    float *dw_out = work;
    for (int t = 0; t < T; t++)
    for (int c = 0; c < ENC_D; c++) {
        float s = bf16(L->cv_dwb[c]);
        for (int k = 0; k < ENC_CK; k++) {
            int ti = t - pad_dw + k;
            if (ti < 0 || ti >= T) continue;
            s += bf16(L->cv_dww[c * ENC_CK + k]) * glu_out[ti * ENC_D + c];
        }
        dw_out[t * ENC_D + c] = s;
    }

    const float bn_eps = 1e-5f;
    for (int t = 0; t < T; t++)
    for (int c = 0; c < ENC_D; c++) {
        float v = dw_out[t*ENC_D + c];
        v = (v - bf16(L->cv_bnmean[c])) / sqrtf(bf16(L->cv_bnvar[c]) + bn_eps)
            * bf16(L->cv_bnw[c]) + bf16(L->cv_bnb[c]);
        dw_out[t*ENC_D + c] = v;
    }

    silu(dw_out, T * ENC_D);

    backend_linear(x, dw_out, L->cv_pw2w, L->cv_pw2b, T, ENC_D, ENC_D);
}

void backend_sdp_attn(float *out, const float *q, const float *k, const float *v,
                      const float *mask, int T_q, int T_k, float *work) {
    float *scores = work;
    const float scale = 1.0f / sqrtf((float)DEC_DK);
    for (int h = 0; h < DEC_H; h++) {
        for (int qi = 0; qi < T_q; qi++)
        for (int ki = 0; ki < T_k; ki++) {
            float s = 0;
            for (int d = 0; d < DEC_DK; d++)
                s += q[qi*DEC_D + h*DEC_DK+d] * k[ki*DEC_D + h*DEC_DK+d];
            scores[qi*T_k + ki] = s * scale + (mask ? mask[qi*T_k + ki] : 0.0f);
        }
        for (int qi = 0; qi < T_q; qi++) softmax(scores + qi*T_k, T_k);
        for (int qi = 0; qi < T_q; qi++)
        for (int d = 0; d < DEC_DK; d++) {
            float s = 0;
            for (int ki = 0; ki < T_k; ki++)
                s += scores[qi*T_k + ki] * v[ki*DEC_D + h*DEC_DK+d];
            out[qi*DEC_D + h*DEC_DK+d] = s;
        }
    }
}

void backend_embed(float *h, const int *tokens, int S, int pos_start,
                   const uint16_t *etw, const uint16_t *epw,
                   const uint16_t *elnw, const uint16_t *elnb) {
    for (int t = 0; t < S; t++) {
        int id = tokens[t];
        for (int d = 0; d < DEC_D; d++)
            h[t*DEC_D + d] = bf16(etw[id*DEC_D + d]) + bf16(epw[(pos_start + t)*DEC_D + d]);
    }
    backend_layernorm(h, h, elnw, elnb, S, DEC_D);
}

void backend_fill_causal_mask(float *mask, int S) {
    for (int qi = 0; qi < S; qi++)
    for (int ki = 0; ki < S; ki++)
        mask[qi*S + ki] = (ki > qi) ? -1e9f : 0.0f;
}

void backend_lm_head(float *logits_out, const float *last,
                     const uint16_t *etw, const uint16_t *hdb) {
    for (int v = 0; v < VOCAB; v++) {
        float s = bf16(hdb[v]);
        const uint16_t *wr = etw + v * DEC_D;
        for (int d = 0; d < DEC_D; d++) s += last[d] * bf16(wr[d]);
        logits_out[v] = s;
    }
}

/* ============================================================ */
/* Device-side sequence counter (thread-local int on CPU)        */
/* ============================================================ */

static _Thread_local int g_cpu_S = 0;

void backend_decode_set_S(int n) { g_cpu_S = n; }
void backend_decode_inc_S(void)  { g_cpu_S++; }

void backend_linear_to_kvcache(float *cache, const float *x,
                                const uint16_t *W, const uint16_t *b,
                                int in_d, int out_d) {
    float *dst = cache + (size_t)(g_cpu_S - 1) * out_d;
    for (int d = 0; d < out_d; d++) {
        float acc = b ? bf16(b[d]) : 0.0f;
        for (int i = 0; i < in_d; i++)
            acc += x[i] * bf16(W[(size_t)d * in_d + i]);
        dst[d] = acc;
    }
}

void backend_sdp_attn_devS(float *out, const float *q,
                            const float *k, const float *v, float *work) {
    backend_sdp_attn(out, q, k, v, NULL, 1, g_cpu_S, work);
}

void backend_embed_decode(float *h,
                          const uint16_t *etw, const uint16_t *epw,
                          const uint16_t *elnw, const uint16_t *elnb) {
    backend_embed(h, g_cpu_tokens, 1, g_cpu_S - 1, etw, epw, elnw, elnb);
}

/* ============================================================ */
/* Batched decode-step helpers                                   */
/* ============================================================ */

void backend_embed_decode_batch(float *h, const int *tokens,
                                 const int *S_dev, const int *active, int B,
                                 const uint16_t *etw, const uint16_t *epw,
                                 const uint16_t *elnw, const uint16_t *elnb) {
    memcpy(g_cpu_S_batch, S_dev, (size_t)B * sizeof(int));
    for (int b = 0; b < B; b++) {
        if (!active[b]) continue;
        backend_embed(h + (size_t)b * DEC_D, &tokens[b], 1, S_dev[b] - 1,
                      etw, epw, elnw, elnb);
    }
}

void backend_lm_head_batch(float *logits, const float *h,
                            const uint16_t *etw, const uint16_t *hdb,
                            const int *active, int B) {
    for (int b = 0; b < B; b++) {
        if (!active[b]) continue;
        backend_lm_head(logits + (size_t)b * VOCAB, h + (size_t)b * DEC_D, etw, hdb);
    }
}

void backend_argmax_batch(int *out, const float *logits,
                           const int *active, int B) {
    for (int b = 0; b < B; b++) {
        if (!active[b]) continue;
        const float *lg = logits + (size_t)b * VOCAB;
        int best = 0; float best_v = lg[0];
        for (int v = 1; v < VOCAB; v++)
            if (lg[v] > best_v) { best_v = lg[v]; best = v; }
        out[b] = best;
    }
}

/* ============================================================ */
/* Batched decode helpers                                        */
/* ============================================================ */

void backend_upload_T_enc(const int *T_enc, int B) {
    memcpy(g_cpu_T_enc_batch, T_enc, (size_t)B * sizeof(int));
}

void backend_linear_to_kvcache_batch(float *cache, const float *x,
                                      const uint16_t *W, const uint16_t *b,
                                      int B, int cache_seq_stride,
                                      int in_d, int out_d) {
    for (int bi = 0; bi < B; bi++) {
        int slot = g_cpu_S_batch[bi] - 1;
        float *dst = cache + ((size_t)bi * cache_seq_stride + slot) * out_d;
        const float *xb = x + (size_t)bi * in_d;
        backend_linear(dst, xb, W, b, 1, in_d, out_d);
    }
}

void backend_sdp_attn_batch_decode_sa(float *out, const float *Q,
                                       const float *K, const float *V,
                                       int T_k_stride, int B, float *work) {
    for (int b = 0; b < B; b++) {
        backend_sdp_attn(out + (size_t)b * DEC_D,
                         Q   + (size_t)b * DEC_D,
                         K   + (size_t)b * T_k_stride * DEC_D,
                         V   + (size_t)b * T_k_stride * DEC_D,
                         NULL, 1, g_cpu_S_batch[b], work);
    }
}

void backend_sdp_attn_batch_decode_ca(float *out, const float *Q,
                                       const float *K, const float *V,
                                       int T_k_stride, int B, float *work) {
    for (int b = 0; b < B; b++) {
        backend_sdp_attn(out + (size_t)b * DEC_D,
                         Q   + (size_t)b * DEC_D,
                         K   + (size_t)b * T_k_stride * DEC_D,
                         V   + (size_t)b * T_k_stride * DEC_D,
                         NULL, 1, g_cpu_T_enc_batch[b], work);
    }
}
