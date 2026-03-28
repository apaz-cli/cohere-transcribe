/* backends/cuda/backend.cu — CUDA backend (naive PoC).
 *
 * Strategy: correctness first, not tuned.
 *   - BF16 weights raw-copied to device (binary-compatible with __nv_bfloat16).
 *   - Single cudaStream_t; all kernels launch on g_stream.
 *   - Device arena: cudaMallocAsync per transcribe_chunk call.
 *   - Mel computed on CPU; backend_htod uploads it before encoder_forward.
 *   - Argmax runs on CPU after backend_dtoh of the VOCAB-wide logit vector.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

extern "C" {
#include "../model_types.h"
#include "../backend.h"
}

/* ============================================================ */
/* Utilities                                                     */
/* ============================================================ */

#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

/* ============================================================ */
/* Device state                                                  */
/* ============================================================ */

/* Per-GPU: weights blob uploaded once before any threads start. */
#define MAX_DEVICES 8
static void        *g_dev_blob[MAX_DEVICES];   /* device copy of safetensors data */
static int          g_num_devices = 0;
static const void  *g_host_blob_base;          /* mmap'd host base for offset calc */

/* Per-thread state: each worker thread owns its own CUDA resources. */
static thread_local cudaStream_t  g_stream        = NULL;
static thread_local float        *g_arena         = NULL;
static thread_local size_t        g_arena_bytes    = 0;
static thread_local int          *g_dev_tokens     = NULL;
static thread_local int          *g_dev_S          = NULL;  /* device-side sequence counter */
static thread_local int           g_device_id      = -1;
static thread_local int           g_capture_tag    = -1;

/* ---- Graph cache (per-thread) ---- */
#define GRAPH_CACHE_CAP 1026   /* 1 encoder + 1 precompute + up to MAX_SEQ decoder */

typedef struct {
    int              tag;
    int              T_enc;
    int              S;
    void            *arena_base;  /* g_arena value at capture time */
    cudaGraph_t      graph;
    cudaGraphExec_t  exec;
} GraphEntry;

static thread_local GraphEntry g_graph_cache[GRAPH_CACHE_CAP];
static thread_local int        g_graph_cache_n = 0;

static void graph_invalidate_all(void) {
    for (int i = 0; i < g_graph_cache_n; i++) {
        cudaGraphExecDestroy(g_graph_cache[i].exec);
        cudaGraphDestroy(g_graph_cache[i].graph);
    }
    g_graph_cache_n = 0;
}

static GraphEntry *graph_find(int tag, int T_enc, int S) {
    for (int i = 0; i < g_graph_cache_n; i++) {
        GraphEntry *e = &g_graph_cache[i];
        if (e->tag == tag && e->T_enc == T_enc && e->S == S
                          && e->arena_base == (void *)g_arena)
            return e;
    }
    return NULL;
}

/* Convert a host blob pointer to the current thread's device pointer. */
static inline const void *dev_w(const void *host_ptr) {
    if (!host_ptr) return NULL;
    ptrdiff_t off = (const uint8_t *)host_ptr - (const uint8_t *)g_host_blob_base;
    return (const uint8_t *)g_dev_blob[g_device_id] + off;
}

/* ============================================================ */
/* Lifecycle                                                     */
/* ============================================================ */

extern "C" int backend_num_devices(void) {
    return g_num_devices;
}

extern "C" void backend_init(int device_id) {
    g_device_id = device_id;
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaStreamCreate(&g_stream));
    CUDA_CHECK(cudaMalloc(&g_dev_tokens, MAX_SEQ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_dev_S, sizeof(int)));
}

extern "C" void backend_thread_cleanup(void) {
    if (g_device_id < 0) return;
    CUDA_CHECK(cudaSetDevice(g_device_id));
    graph_invalidate_all();
    if (g_arena)      { CUDA_CHECK(cudaFree(g_arena));      g_arena = NULL; g_arena_bytes = 0; }
    if (g_dev_tokens) { CUDA_CHECK(cudaFree(g_dev_tokens)); g_dev_tokens = NULL; }
    if (g_dev_S)      { CUDA_CHECK(cudaFree(g_dev_S));      g_dev_S = NULL; }
    if (g_stream)     { CUDA_CHECK(cudaStreamDestroy(g_stream)); g_stream = NULL; }
    g_device_id = -1;
}

extern "C" void backend_destroy(void) {
    backend_thread_cleanup();  /* clean up main thread's per-thread resources */
    for (int d = 0; d < g_num_devices; d++) {
        if (g_dev_blob[d]) {
            CUDA_CHECK(cudaSetDevice(d));
            CUDA_CHECK(cudaFree(g_dev_blob[d]));
            g_dev_blob[d] = NULL;
        }
    }
    g_num_devices = 0;
}

extern "C" void backend_weights_upload_blob(const void *host_base, size_t bytes) {
    g_host_blob_base = host_base;
    int n = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n));
    g_num_devices = (n > MAX_DEVICES) ? MAX_DEVICES : n;
    for (int d = 0; d < g_num_devices; d++) {
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaMalloc(&g_dev_blob[d], bytes));
        CUDA_CHECK(cudaMemcpy(g_dev_blob[d], host_base, bytes, cudaMemcpyHostToDevice));
    }
}

extern "C" void *backend_device_ptr(const void *host_ptr) {
    ptrdiff_t off = (const uint8_t *)host_ptr - (const uint8_t *)g_host_blob_base;
    return (uint8_t *)g_dev_blob[g_device_id] + off;
}

extern "C" float *backend_arena_alloc(size_t floats) {
    size_t need = floats * sizeof(float);
    if (need > g_arena_bytes) {
        if (g_arena) {
            graph_invalidate_all();
            CUDA_CHECK(cudaFree(g_arena));
        }
        CUDA_CHECK(cudaMalloc(&g_arena, need));
        g_arena_bytes = need;
    }
    return g_arena;
}

extern "C" void backend_arena_free(float *arena) {
    (void)arena;
    /* Keep the arena alive for graph replay; just sync so the caller knows work is done. */
    CUDA_CHECK(cudaStreamSynchronize(g_stream));
}

extern "C" void backend_htod(float *dev_dst, const float *host_src, size_t floats) {
    CUDA_CHECK(cudaMemcpyAsync(dev_dst, host_src, floats * sizeof(float),
                               cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaStreamSynchronize(g_stream));
}

extern "C" void backend_dtoh(void *host_dst, const void *dev_src, size_t bytes) {
    CUDA_CHECK(cudaMemcpyAsync(host_dst, dev_src, bytes,
                               cudaMemcpyDeviceToHost, g_stream));
    CUDA_CHECK(cudaStreamSynchronize(g_stream));
}

/* ============================================================ */
/* LayerNorm                                                     */
/* ============================================================ */

/* One block per row; threads cooperate on mean/var via warp shuffle + shared. */
__global__ void layernorm_k(float *dst, const float *src,
                             const __nv_bfloat16 *w, const __nv_bfloat16 *b,
                             int T, int D) {
    int t = blockIdx.x;
    if (t >= T) return;
    const float *sr = src + (size_t)t * D;
    float       *dr = dst + (size_t)t * D;

    __shared__ float smem[32];

    /* mean */
    float sum = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) sum += sr[d];
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
    if (threadIdx.x % 32 == 0) smem[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        int nw = (blockDim.x + 31) / 32;
        for (int i = 0; i < nw; i++) s += smem[i];
        smem[0] = s / D;
    }
    __syncthreads();
    float mean = smem[0];

    /* variance */
    float var = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = sr[d] - mean; var += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) var += __shfl_down_sync(0xffffffff, var, off);
    if (threadIdx.x % 32 == 0) smem[threadIdx.x / 32] = var;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        int nw = (blockDim.x + 31) / 32;
        for (int i = 0; i < nw; i++) s += smem[i];
        smem[0] = rsqrtf(s / D + 1e-5f);
    }
    __syncthreads();
    float is = smem[0];

    for (int d = threadIdx.x; d < D; d += blockDim.x)
        dr[d] = (sr[d] - mean) * is * __bfloat162float(w[d]) + __bfloat162float(b[d]);
}

extern "C" void backend_layernorm(float *dst, const float *src,
                                   const uint16_t *w, const uint16_t *b,
                                   int T, int D) {
    int threads = (D < 256) ? 64 : 256;
    layernorm_k<<<T, threads, 0, g_stream>>>(
        dst, src,
        (const __nv_bfloat16 *)dev_w(w), (const __nv_bfloat16 *)dev_w(b),
        T, D);
}

/* ============================================================ */
/* Linear family                                                 */
/* ============================================================ */

/* Each thread computes one output element y[t, o].
 * Grid: (T, ceil(out_d/256))  Block: 256 */

__global__ void linear_k(float *y, const float *x,
                          const __nv_bfloat16 *W, const __nv_bfloat16 *b,
                          int T, int in_d, int out_d) {
    int t = blockIdx.x, o = (int)blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= T || o >= out_d) return;
    float s = b ? __bfloat162float(b[o]) : 0.0f;
    const float         *xr = x + (size_t)t * in_d;
    const __nv_bfloat16 *wr = W + (size_t)o * in_d;
    for (int k = 0; k < in_d; k++) s += xr[k] * __bfloat162float(wr[k]);
    y[(size_t)t * out_d + o] = s;
}

__global__ void linear_silu_k(float *y, const float *x,
                                const __nv_bfloat16 *W, const __nv_bfloat16 *b,
                                int T, int in_d, int out_d) {
    int t = blockIdx.x, o = (int)blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= T || o >= out_d) return;
    float s = __bfloat162float(b[o]);
    const float         *xr = x + (size_t)t * in_d;
    const __nv_bfloat16 *wr = W + (size_t)o * in_d;
    for (int k = 0; k < in_d; k++) s += xr[k] * __bfloat162float(wr[k]);
    y[(size_t)t * out_d + o] = s / (1.0f + expf(-s));
}

__global__ void linear_relu_k(float *y, const float *x,
                                const __nv_bfloat16 *W, const __nv_bfloat16 *b,
                                int T, int in_d, int out_d) {
    int t = blockIdx.x, o = (int)blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= T || o >= out_d) return;
    float s = __bfloat162float(b[o]);
    const float         *xr = x + (size_t)t * in_d;
    const __nv_bfloat16 *wr = W + (size_t)o * in_d;
    for (int k = 0; k < in_d; k++) s += xr[k] * __bfloat162float(wr[k]);
    y[(size_t)t * out_d + o] = fmaxf(s, 0.0f);
}

__global__ void linear_fmadd_k(float *acc, float scale, const float *x,
                                 const __nv_bfloat16 *W, const __nv_bfloat16 *b,
                                 int T, int in_d, int out_d) {
    int t = blockIdx.x, o = (int)blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= T || o >= out_d) return;
    float s = b ? __bfloat162float(b[o]) : 0.0f;
    const float         *xr = x + (size_t)t * in_d;
    const __nv_bfloat16 *wr = W + (size_t)o * in_d;
    for (int k = 0; k < in_d; k++) s += xr[k] * __bfloat162float(wr[k]);
    acc[(size_t)t * out_d + o] += scale * s;
}

extern "C" void backend_linear(float *y, const float *x,
                                 const uint16_t *W, const uint16_t *b,
                                 int T, int in_d, int out_d) {
    dim3 grid(T, (out_d + 255) / 256);
    linear_k<<<grid, 256, 0, g_stream>>>(
        y, x,
        (const __nv_bfloat16 *)dev_w(W), (const __nv_bfloat16 *)dev_w(b),
        T, in_d, out_d);
}

extern "C" void backend_linear_silu(float *y, const float *x,
                                      const uint16_t *W, const uint16_t *b,
                                      int T, int in_d, int out_d) {
    dim3 grid(T, (out_d + 255) / 256);
    linear_silu_k<<<grid, 256, 0, g_stream>>>(
        y, x,
        (const __nv_bfloat16 *)dev_w(W), (const __nv_bfloat16 *)dev_w(b),
        T, in_d, out_d);
}

extern "C" void backend_linear_relu(float *y, const float *x,
                                      const uint16_t *W, const uint16_t *b,
                                      int T, int in_d, int out_d) {
    dim3 grid(T, (out_d + 255) / 256);
    linear_relu_k<<<grid, 256, 0, g_stream>>>(
        y, x,
        (const __nv_bfloat16 *)dev_w(W), (const __nv_bfloat16 *)dev_w(b),
        T, in_d, out_d);
}

extern "C" void backend_linear_fmadd(float *acc, float scale, const float *x,
                                       const uint16_t *W, const uint16_t *b,
                                       int T, int in_d, int out_d) {
    dim3 grid(T, (out_d + 255) / 256);
    linear_fmadd_k<<<grid, 256, 0, g_stream>>>(
        acc, scale, x,
        (const __nv_bfloat16 *)dev_w(W), (const __nv_bfloat16 *)dev_w(b),
        T, in_d, out_d);
}

/* ============================================================ */
/* Element-wise add                                              */
/* ============================================================ */

__global__ void add_inplace_k(float *dst, const float *src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

extern "C" void backend_add_inplace(float *dst, const float *src, int n) {
    add_inplace_k<<<(n + 255) / 256, 256, 0, g_stream>>>(dst, src, n);
}

/* ============================================================ */
/* Softmax (shared-memory, one block per row)                    */
/* ============================================================ */

__global__ void softmax_rows_k(float *x, int rows, int cols) {
    int r = blockIdx.x;
    if (r >= rows) return;
    float *row = x + (size_t)r * cols;

    __shared__ float smem[33];

    /* max */
    float mx = -1e30f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        mx = fmaxf(mx, row[c]);
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, off));
    if (threadIdx.x % 32 == 0) smem[threadIdx.x / 32] = mx;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m2 = smem[0];
        int nw = (blockDim.x + 31) / 32;
        for (int i = 1; i < nw; i++) m2 = fmaxf(m2, smem[i]);
        smem[32] = m2;
    }
    __syncthreads();
    mx = smem[32];

    /* exp + sum */
    float s = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        row[c] = expf(row[c] - mx); s += row[c];
    }
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffff, s, off);
    if (threadIdx.x % 32 == 0) smem[threadIdx.x / 32] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s2 = smem[0];
        int nw = (blockDim.x + 31) / 32;
        for (int i = 1; i < nw; i++) s2 += smem[i];
        smem[32] = s2;
    }
    __syncthreads();
    s = smem[32];
    for (int c = threadIdx.x; c < cols; c += blockDim.x) row[c] /= s;
}

/* ============================================================ */
/* Relative-position MHSA (encoder)                              */
/* ============================================================ */

/* ac score: scores[q,k] = scale * sum_d (Q[q,h,d] + pbu[h,d]) * K[k,h,d] */
__global__ void mhsa_ac_k(float *scores, const float *Q, const float *K,
                            const __nv_bfloat16 *pbu,
                            float scale, int T, int h, int DK, int D) {
    int q = blockIdx.x, k = blockIdx.y * blockDim.x + threadIdx.x;
    if (q >= T || k >= T) return;
    float s = 0.0f;
    for (int d = 0; d < DK; d++)
        s += (Q[(size_t)q*D + h*DK+d] + __bfloat162float(pbu[h*DK+d]))
           *  K[(size_t)k*D + h*DK+d];
    scores[(size_t)q*T + k] = s * scale;
}

/* bd score and rel-shift add in one pass:
 * scores[q,k] += scale * sum_d (Q[q,h,d] + pbv[h,d]) * P[k+(T-1)-q, h, d] */
__global__ void mhsa_bd_k(float *scores, const float *Q, const float *P,
                            const __nv_bfloat16 *pbv,
                            float scale, int T, int pos_len, int h, int DK, int D) {
    int q = blockIdx.x, k = blockIdx.y * blockDim.x + threadIdx.x;
    if (q >= T || k >= T) return;
    int p = k + (T-1) - q;
    float s = 0.0f;
    for (int d = 0; d < DK; d++)
        s += (Q[(size_t)q*D + h*DK+d] + __bfloat162float(pbv[h*DK+d]))
           *  P[(size_t)p*D + h*DK+d];
    scores[(size_t)q*T + k] += s * scale;
}

__global__ void mhsa_aggregate_k(float *attn_out, const float *scores, const float *V,
                                   int T, int h, int DK, int D) {
    int q = blockIdx.x, d = blockIdx.y * blockDim.x + threadIdx.x;
    if (q >= T || d >= DK) return;
    float s = 0.0f;
    for (int k = 0; k < T; k++)
        s += scores[(size_t)q*T + k] * V[(size_t)k*D + h*DK+d];
    attn_out[(size_t)q*D + h*DK+d] = s;
}

extern "C" void backend_rel_pos_mhsa(float *x, const float *pos_emb,
                                       int T, const ELayer *L, float *work) {
    const int pos_len = 2 * T - 1;
    float *Q       = work;
    float *K       = Q + (size_t)T       * ENC_D;
    float *V       = K + (size_t)T       * ENC_D;
    float *P       = V + (size_t)T       * ENC_D;
    float *attn_out= P + (size_t)pos_len * ENC_D;
    float *scores  = attn_out + (size_t)T * ENC_D;

    backend_linear(Q, x,       L->sa_qw, L->sa_qb, T,       ENC_D, ENC_D);
    backend_linear(K, x,       L->sa_kw, L->sa_kb, T,       ENC_D, ENC_D);
    backend_linear(V, x,       L->sa_vw, L->sa_vb, T,       ENC_D, ENC_D);
    backend_linear(P, pos_emb, L->sa_pw, NULL,     pos_len,  ENC_D, ENC_D);

    CUDA_CHECK(cudaMemsetAsync(attn_out, 0, (size_t)T * ENC_D * sizeof(float), g_stream));

    const float scale = 1.0f / sqrtf((float)ENC_DK);

    for (int h = 0; h < ENC_H; h++) {
        /* ac scores */
        mhsa_ac_k<<<dim3(T, (T+255)/256), 256, 0, g_stream>>>(
            scores, Q, K, (const __nv_bfloat16 *)dev_w(L->sa_pbu),
            scale, T, h, ENC_DK, ENC_D);

        /* bd scores with rel-shift, added into scores */
        mhsa_bd_k<<<dim3(T, (T+255)/256), 256, 0, g_stream>>>(
            scores, Q, P, (const __nv_bfloat16 *)dev_w(L->sa_pbv),
            scale, T, pos_len, h, ENC_DK, ENC_D);

        /* softmax over k */
        softmax_rows_k<<<T, 256, 0, g_stream>>>(scores, T, T);

        /* aggregate V → attn_out */
        mhsa_aggregate_k<<<dim3(T, (ENC_DK+31)/32), 32, 0, g_stream>>>(
            attn_out, scores, V, T, h, ENC_DK, ENC_D);
    }

    backend_linear(x, attn_out, L->sa_ow, L->sa_ob, T, ENC_D, ENC_D);
}

/* ============================================================ */
/* Conformer conv module                                         */
/* ============================================================ */

__global__ void glu_k(float *out, const float *pw1, int T, int D) {
    int t = blockIdx.x, d = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= T || d >= D) return;
    float a = pw1[(size_t)t*2*D + d];
    float b = pw1[(size_t)t*2*D + d + D];
    out[(size_t)t*D + d] = a / (1.0f + expf(-b));
}

__global__ void depthwise_conv1d_k(float *out, const float *in,
                                    const __nv_bfloat16 *W, const __nv_bfloat16 *b,
                                    int T, int C, int K, int pad) {
    int t = blockIdx.x, c = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= T || c >= C) return;
    float s = __bfloat162float(b[c]);
    for (int k = 0; k < K; k++) {
        int ti = t - pad + k;
        if (ti >= 0 && ti < T) s += __bfloat162float(W[c*K + k]) * in[(size_t)ti*C + c];
    }
    out[(size_t)t*C + c] = s;
}

/* batchnorm + silu fused: one thread per (t, c) */
__global__ void batchnorm_silu_k(float *x,
                                   const __nv_bfloat16 *bnw, const __nv_bfloat16 *bnb,
                                   const __nv_bfloat16 *bnmean, const __nv_bfloat16 *bnvar,
                                   int T, int C) {
    int t = blockIdx.x, c = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= T || c >= C) return;
    float v = x[(size_t)t*C + c];
    v = (v - __bfloat162float(bnmean[c])) / sqrtf(__bfloat162float(bnvar[c]) + 1e-5f)
        * __bfloat162float(bnw[c]) + __bfloat162float(bnb[c]);
    x[(size_t)t*C + c] = v / (1.0f + expf(-v));
}

extern "C" void backend_conformer_conv(float *x, int T, const ELayer *L, float *work) {
    float *pw1_out = work;
    backend_linear(pw1_out, x, L->cv_pw1w, L->cv_pw1b, T, ENC_D, 2*ENC_D);

    float *glu_out = work + (size_t)T * 2 * ENC_D;
    {
        dim3 grid(T, (ENC_D + 255)/256);
        glu_k<<<grid, 256, 0, g_stream>>>(glu_out, pw1_out, T, ENC_D);
    }

    float *dw_out = work;
    {
        int pad = (ENC_CK - 1) / 2;
        dim3 grid(T, (ENC_D + 255)/256);
        depthwise_conv1d_k<<<grid, 256, 0, g_stream>>>(
            dw_out, glu_out,
            (const __nv_bfloat16 *)dev_w(L->cv_dww), (const __nv_bfloat16 *)dev_w(L->cv_dwb),
            T, ENC_D, ENC_CK, pad);
    }
    {
        dim3 grid(T, (ENC_D + 255)/256);
        batchnorm_silu_k<<<grid, 256, 0, g_stream>>>(
            dw_out,
            (const __nv_bfloat16 *)dev_w(L->cv_bnw),   (const __nv_bfloat16 *)dev_w(L->cv_bnb),
            (const __nv_bfloat16 *)dev_w(L->cv_bnmean), (const __nv_bfloat16 *)dev_w(L->cv_bnvar),
            T, ENC_D);
    }

    backend_linear(x, dw_out, L->cv_pw2w, L->cv_pw2b, T, ENC_D, ENC_D);
}

/* ============================================================ */
/* Encoder subsampling                                           */
/* ============================================================ */

__global__ void transpose_mel_k(float *out, const float *in, int T_mel, int M) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (t < T_mel && m < M) out[(size_t)t*M + m] = in[(size_t)m*T_mel + t];
}

__global__ void conv2d_k(float *out, const float *in, const __nv_bfloat16 *Wf, const __nv_bfloat16 *bf,
                          int in_c, int in_h, int in_w,
                          int out_c, int kH, int kW, int stride, int pad, int groups) {
    int out_h  = (in_h + 2*pad - kH) / stride + 1;
    int out_w  = (in_w + 2*pad - kW) / stride + 1;
    int ic_g   = in_c  / groups, oc_g = out_c / groups;
    int oc_gl  = blockIdx.x;
    int oh     = blockIdx.y;
    int ow     = (int)blockIdx.z * blockDim.x + threadIdx.x;
    if (oc_gl >= out_c || oh >= out_h || ow >= out_w) return;

    int g  = oc_gl / oc_g;
    float s = bf ? __bfloat162float(bf[oc_gl]) : 0.0f;
    for (int ic = 0; ic < ic_g; ic++) {
        int ic_gl = g*ic_g + ic;
        for (int kh = 0; kh < kH; kh++) {
            int ih = oh*stride - pad + kh;
            if (ih < 0 || ih >= in_h) continue;
            for (int kw = 0; kw < kW; kw++) {
                int iw = ow*stride - pad + kw;
                if (iw < 0 || iw >= in_w) continue;
                s += in[(size_t)ic_gl*(in_h*in_w) + ih*in_w + iw]
                   * __bfloat162float(Wf[(size_t)(oc_gl*ic_g + ic)*kH*kW + kh*kW + kw]);
            }
        }
    }
    out[(size_t)oc_gl*(out_h*out_w) + oh*out_w + ow] = s;
}

__global__ void relu_k(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = fmaxf(x[i], 0.0f);
}

static void launch_conv2d(float *out, const float *in,
                           const uint16_t *Wf, const uint16_t *bf_,
                           int in_c, int in_h, int in_w,
                           int out_c, int kH, int kW, int stride, int pad, int groups) {
    int out_h = (in_h + 2*pad - kH) / stride + 1;
    int out_w = (in_w + 2*pad - kW) / stride + 1;
    const int BW = 32;
    dim3 grid(out_c, out_h, (out_w + BW - 1) / BW);
    conv2d_k<<<grid, BW, 0, g_stream>>>(
        out, in,
        (const __nv_bfloat16 *)dev_w(Wf), (const __nv_bfloat16 *)dev_w(bf_),
        in_c, in_h, in_w, out_c, kH, kW, stride, pad, groups);
}

__global__ void flatten_sub_k(float *out, const float *in, int T_enc, int CH, int F) {
    int t  = blockIdx.x * blockDim.x + threadIdx.x;
    int ch = blockIdx.y;
    if (t >= T_enc || ch >= CH) return;
    for (int f = 0; f < F; f++)
        out[(size_t)t*(CH*F) + ch*F + f] = in[(size_t)ch*(T_enc*F) + t*F + f];
}

extern "C" void backend_encoder_subsampling(const float *mel, int T_mel,
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

    /* Transpose mel (N_MELS, T_mel) → (T_mel, N_MELS) */
    {
        dim3 block(16, 16), grid((T_mel+15)/16, (N_MELS+15)/16);
        transpose_mel_k<<<grid, block, 0, g_stream>>>(mel_t, mel, T_mel, N_MELS);
    }

    launch_conv2d(buf0, mel_t, W->s0w, W->s0b, 1, T_mel, N_MELS, SUB_CH, 3,3,2,1,1);
    { int n = SUB_CH*T1*F1; relu_k<<<(n+255)/256, 256, 0, g_stream>>>(buf0, n); }

    float *buf1 = work + B0;
    launch_conv2d(buf1, buf0, W->s2w, W->s2b, SUB_CH, T1, F1, SUB_CH, 3,3,2,1,SUB_CH);

    float *buf1b = work;
    launch_conv2d(buf1b, buf1, W->s3w, W->s3b, SUB_CH, T2, F2, SUB_CH, 1,1,1,0,1);
    { int n = SUB_CH*T2*F2; relu_k<<<(n+255)/256, 256, 0, g_stream>>>(buf1b, n); }

    float *buf2 = work + B1;
    launch_conv2d(buf2, buf1b, W->s5w, W->s5b, SUB_CH, T2, F2, SUB_CH, 3,3,2,1,SUB_CH);

    float *buf2b = work;
    launch_conv2d(buf2b, buf2, W->s6w, W->s6b, SUB_CH, T_enc, F3, SUB_CH, 1,1,1,0,1);
    { int n = SUB_CH*T_enc*F3; relu_k<<<(n+255)/256, 256, 0, g_stream>>>(buf2b, n); }

    float *flat = work + B2;
    {
        dim3 grid((T_enc+31)/32, SUB_CH);
        flatten_sub_k<<<grid, 32, 0, g_stream>>>(flat, buf2b, T_enc, SUB_CH, F3);
    }

    backend_linear(x_enc, flat, W->sow, W->sob, T_enc, in_feat, ENC_D);
}

/* ============================================================ */
/* Relative positional embedding                                 */
/* ============================================================ */

__global__ void rel_pos_emb_k(float *pe, int T, int D) {
    int p = blockIdx.x;
    int k = blockIdx.y * blockDim.x + threadIdx.x;
    if (p >= 2*T-1 || k >= D/2) return;
    float pos    = (float)(T - 1 - p);
    float log10k = logf(10000.0f);
    float dt     = expf(-(float)(2*k) * log10k / D);
    pe[(size_t)p*D + 2*k]     = sinf(pos * dt);
    pe[(size_t)p*D + 2*k + 1] = cosf(pos * dt);
}

extern "C" void backend_make_rel_pos_emb(float *pe, int T) {
    int L = 2 * T - 1;
    dim3 grid(L, (ENC_D/2 + 255)/256);
    rel_pos_emb_k<<<grid, 256, 0, g_stream>>>(pe, T, ENC_D);
}

/* ============================================================ */
/* Scaled dot-product attention (decoder)                        */
/* ============================================================ */

__global__ void sdp_score_k(float *scores, const float *q, const float *k,
                              const float *mask, float scale,
                              int T_q, int T_k, int h, int DK, int D) {
    int qi = blockIdx.x, ki = blockIdx.y * blockDim.x + threadIdx.x;
    if (qi >= T_q || ki >= T_k) return;
    float s = 0.0f;
    for (int d = 0; d < DK; d++)
        s += q[(size_t)qi*D + h*DK+d] * k[(size_t)ki*D + h*DK+d];
    s = s * scale + (mask ? mask[(size_t)qi*T_k + ki] : 0.0f);
    scores[(size_t)qi*T_k + ki] = s;
}

__global__ void sdp_aggregate_k(float *out, const float *scores, const float *v,
                                  int T_q, int T_k, int h, int DK, int D) {
    int qi = blockIdx.x, d = blockIdx.y * blockDim.x + threadIdx.x;
    if (qi >= T_q || d >= DK) return;
    float s = 0.0f;
    for (int ki = 0; ki < T_k; ki++)
        s += scores[(size_t)qi*T_k + ki] * v[(size_t)ki*D + h*DK+d];
    out[(size_t)qi*D + h*DK+d] = s;
}

extern "C" void backend_sdp_attn(float *out, const float *q, const float *k, const float *v,
                                   const float *mask, int T_q, int T_k, float *work) {
    float *scores = work;
    const float scale = 1.0f / sqrtf((float)DEC_DK);
    for (int h = 0; h < DEC_H; h++) {
        {
            dim3 grid(T_q, (T_k + 255)/256);
            sdp_score_k<<<grid, 256, 0, g_stream>>>(
                scores, q, k, mask, scale, T_q, T_k, h, DEC_DK, DEC_D);
        }
        softmax_rows_k<<<T_q, 256, 0, g_stream>>>(scores, T_q, T_k);
        {
            dim3 grid(T_q, (DEC_DK + 31)/32);
            sdp_aggregate_k<<<grid, 32, 0, g_stream>>>(out, scores, v, T_q, T_k, h, DEC_DK, DEC_D);
        }
    }
}

/* ============================================================ */
/* Token embedding + causal mask + LM head                       */
/* ============================================================ */

__global__ void embed_k(float *h, const int *tokens,
                         const __nv_bfloat16 *etw, const __nv_bfloat16 *epw,
                         int S, int D, int pos_start) {
    int t = blockIdx.x, d = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= S || d >= D) return;
    int id = tokens[t];
    h[(size_t)t*D + d] = __bfloat162float(etw[(size_t)id*D + d])
                        + __bfloat162float(epw[(size_t)(pos_start + t)*D + d]);
}

extern "C" void backend_embed(float *h, const int *tokens, int S, int pos_start,
                                const uint16_t *etw, const uint16_t *epw,
                                const uint16_t *elnw, const uint16_t *elnb) {
    /* tokens must have been staged to g_dev_tokens via backend_upload_tokens()
     * before graph capture; the pointer is stable so the kernel can be captured. */
    (void)tokens; /* host copy not used here; device copy is in g_dev_tokens */
    {
        dim3 grid(S, (DEC_D + 255)/256);
        embed_k<<<grid, 256, 0, g_stream>>>(
            h, g_dev_tokens,
            (const __nv_bfloat16 *)dev_w(etw), (const __nv_bfloat16 *)dev_w(epw),
            S, DEC_D, pos_start);
    }
    backend_layernorm(h, h, elnw, elnb, S, DEC_D);
}

__global__ void fill_causal_mask_k(float *mask, int S) {
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    int ki = blockIdx.y * blockDim.y + threadIdx.y;
    if (qi >= S || ki >= S) return;
    mask[(size_t)qi*S + ki] = (ki > qi) ? -1e9f : 0.0f;
}

extern "C" void backend_fill_causal_mask(float *mask, int S) {
    dim3 block(16, 16), grid((S+15)/16, (S+15)/16);
    fill_causal_mask_k<<<grid, block, 0, g_stream>>>(mask, S);
}

__global__ void lm_head_k(float *out, const float *last,
                            const __nv_bfloat16 *etw, const __nv_bfloat16 *hdb,
                            int V, int D) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= V) return;
    float s = __bfloat162float(hdb[v]);
    const __nv_bfloat16 *wr = etw + (size_t)v * D;
    for (int d = 0; d < D; d++) s += last[d] * __bfloat162float(wr[d]);
    out[v] = s;
}

extern "C" void backend_lm_head(float *logits_out, const float *last,
                                  const uint16_t *etw, const uint16_t *hdb) {
    lm_head_k<<<(VOCAB + 255) / 256, 256, 0, g_stream>>>(
        logits_out, last,
        (const __nv_bfloat16 *)dev_w(etw), (const __nv_bfloat16 *)dev_w(hdb),
        VOCAB, DEC_D);
}

/* ============================================================ */
/* CUDA Graph capture / replay                                   */
/* ============================================================ */

extern "C" void backend_upload_tokens(const int *host_tokens, int S) {
    /* Copy tokens to the stable device buffer BEFORE stream capture begins,
     * so this memcpy is outside the captured region but ordered before embed_k. */
    CUDA_CHECK(cudaMemcpyAsync(g_dev_tokens, host_tokens, (size_t)S * sizeof(int),
                               cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaStreamSynchronize(g_stream));
}

extern "C" int backend_graph_replay(int tag, int T_enc, int S) {
    GraphEntry *e = graph_find(tag, T_enc, S);
    if (!e) return 0;
    CUDA_CHECK(cudaGraphLaunch(e->exec, g_stream));
    CUDA_CHECK(cudaStreamSynchronize(g_stream));
    return 1;
}

extern "C" void backend_graph_begin_capture(int tag) {
    g_capture_tag = tag;
    CUDA_CHECK(cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeThreadLocal));
}

extern "C" void backend_graph_end_capture(int tag, int T_enc, int S) {
    if (g_graph_cache_n >= GRAPH_CACHE_CAP) {
        /* Cache full: end capture, launch once (kernels did NOT run during capture),
         * then destroy.  The result lands in the arena just like the cached path. */
        cudaGraph_t tmp; cudaGraphExec_t exec;
        CUDA_CHECK(cudaStreamEndCapture(g_stream, &tmp));
        CUDA_CHECK(cudaGraphInstantiate(&exec, tmp, NULL, NULL, 0));
        CUDA_CHECK(cudaGraphLaunch(exec, g_stream));
        CUDA_CHECK(cudaStreamSynchronize(g_stream));
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(tmp);
        g_capture_tag = -1;
        return;
    }

    GraphEntry *e = &g_graph_cache[g_graph_cache_n++];
    e->tag        = tag;
    e->T_enc      = T_enc;
    e->S          = S;
    e->arena_base = (void *)g_arena;

    CUDA_CHECK(cudaStreamEndCapture(g_stream, &e->graph));
    CUDA_CHECK(cudaGraphInstantiate(&e->exec, e->graph, NULL, NULL, 0));

    /* Launch once so the first (recording) pass also produces results. */
    CUDA_CHECK(cudaGraphLaunch(e->exec, g_stream));
    CUDA_CHECK(cudaStreamSynchronize(g_stream));

    g_capture_tag = -1;
}

/* ============================================================ */
/* Device-side sequence counter for single-graph decode          */
/* ============================================================ */

extern "C" void backend_decode_set_S(int n) {
    /* Called from host outside any captured region. */
    CUDA_CHECK(cudaMemcpyAsync(g_dev_S, &n, sizeof(int),
                               cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaStreamSynchronize(g_stream));
}

__global__ void inc_decode_S_k(int *S_ptr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) atomicAdd(S_ptr, 1);
}

extern "C" void backend_decode_inc_S(void) {
    inc_decode_S_k<<<1, 1, 0, g_stream>>>(g_dev_S);
}

/* Linear (T=1) → KV cache at slot (*g_dev_S - 1). */
__global__ void linear_to_kvcache_k(float *cache, const float *x,
                                     const __nv_bfloat16 *W, const __nv_bfloat16 *b,
                                     const int *S_ptr, int in_d, int out_d) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out_d) return;
    int slot = *S_ptr - 1;
    float s = b ? __bfloat162float(b[o]) : 0.0f;
    const __nv_bfloat16 *wr = W + (size_t)o * in_d;
    for (int k = 0; k < in_d; k++) s += x[k] * __bfloat162float(wr[k]);
    cache[(size_t)slot * out_d + o] = s;
}

extern "C" void backend_linear_to_kvcache(float *cache, const float *x,
                                           const uint16_t *W, const uint16_t *b,
                                           int in_d, int out_d) {
    linear_to_kvcache_k<<<(out_d + 255)/256, 256, 0, g_stream>>>(
        cache, x,
        (const __nv_bfloat16 *)dev_w(W), (const __nv_bfloat16 *)dev_w(b),
        g_dev_S, in_d, out_d);
}

/* SDP attention for T_q=1, T_k=*g_dev_S.
 * Grid is over-provisioned to MAX_SEQ; inactive lanes self-terminate. */
__global__ void sdp_score_decode_k(float *scores, const float *q, const float *k,
                                    float scale, int h, int DK, int D,
                                    const int *T_k_ptr) {
    int ki = blockIdx.x * blockDim.x + threadIdx.x;
    if (ki >= *T_k_ptr) return;
    float s = 0.0f;
    for (int d = 0; d < DK; d++)
        s += q[h*DK+d] * k[(size_t)ki*D + h*DK+d];
    scores[ki] = s * scale;
}

/* Softmax in-place over scores[0..*T_k_ptr-1], 1 block / 256 threads. */
__global__ void softmax_decode_k(float *scores, const int *T_k_ptr) {
    int T_k = *T_k_ptr;
    __shared__ float smem[33];

    /* max */
    float mx = -1e30f;
    for (int i = threadIdx.x; i < T_k; i += blockDim.x)
        mx = fmaxf(mx, scores[i]);
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, off));
    if (threadIdx.x % 32 == 0) smem[threadIdx.x / 32] = mx;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m2 = smem[0];
        int nw = (blockDim.x + 31) / 32;
        for (int i = 1; i < nw; i++) m2 = fmaxf(m2, smem[i]);
        smem[32] = m2;
    }
    __syncthreads();
    mx = smem[32];

    /* exp + sum */
    float s = 0.0f;
    for (int i = threadIdx.x; i < T_k; i += blockDim.x) {
        scores[i] = expf(scores[i] - mx);
        s += scores[i];
    }
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffff, s, off);
    if (threadIdx.x % 32 == 0) smem[threadIdx.x / 32] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s2 = smem[0];
        int nw = (blockDim.x + 31) / 32;
        for (int i = 1; i < nw; i++) s2 += smem[i];
        smem[32] = s2;
    }
    __syncthreads();
    s = smem[32];
    for (int i = threadIdx.x; i < T_k; i += blockDim.x) scores[i] /= s;
}

__global__ void sdp_aggregate_decode_k(float *out, const float *scores, const float *v,
                                        int h, int DK, int D, const int *T_k_ptr) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= DK) return;
    int T_k = *T_k_ptr;
    float s = 0.0f;
    for (int ki = 0; ki < T_k; ki++)
        s += scores[ki] * v[(size_t)ki*D + h*DK+d];
    out[h*DK+d] = s;
}

extern "C" void backend_sdp_attn_devS(float *out, const float *q,
                                       const float *k, const float *v, float *work) {
    float *scores = work;  /* MAX_SEQ floats */
    const float scale = 1.0f / sqrtf((float)DEC_DK);
    for (int h = 0; h < DEC_H; h++) {
        sdp_score_decode_k<<<(MAX_SEQ + 255)/256, 256, 0, g_stream>>>(
            scores, q, k, scale, h, DEC_DK, DEC_D, g_dev_S);
        softmax_decode_k<<<1, 256, 0, g_stream>>>(scores, g_dev_S);
        sdp_aggregate_decode_k<<<(DEC_DK + 31)/32, 32, 0, g_stream>>>(
            out, scores, v, h, DEC_DK, DEC_D, g_dev_S);
    }
}

/* Embed single token (from g_dev_tokens[0]) at position *g_dev_S - 1. */
__global__ void embed_decode_k(float *h,
                                const __nv_bfloat16 *etw, const __nv_bfloat16 *epw,
                                int D, const int *tokens, const int *S_ptr) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    int id  = tokens[0];
    int pos = *S_ptr - 1;
    h[d] = __bfloat162float(etw[(size_t)id*D + d])
         + __bfloat162float(epw[(size_t)pos*D + d]);
}

extern "C" void backend_embed_decode(float *h,
                                      const uint16_t *etw, const uint16_t *epw,
                                      const uint16_t *elnw, const uint16_t *elnb) {
    embed_decode_k<<<(DEC_D + 255)/256, 256, 0, g_stream>>>(
        h,
        (const __nv_bfloat16 *)dev_w(etw), (const __nv_bfloat16 *)dev_w(epw),
        DEC_D, g_dev_tokens, g_dev_S);
    backend_layernorm(h, h, elnw, elnb, 1, DEC_D);
}
