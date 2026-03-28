/* tests/test_kernels.c
 * Exercises each backend kernel with reproducible synthetic inputs.
 * Usage: test_kernels <output.bin>
 *
 * Writes all computed float results to a single binary file. Two binaries
 * built against different backends can be compared with compare_floats to
 * verify numerical agreement.
 *
 * Build (examples):
 *   CPU:  gcc -O3 -march=native -I. -o tests/test_kernels_cpu  \
 *               tests/test_kernels.c backends/cpu/backend.o -lm
 *   CUDA: gcc -O3 -march=native -DBACKEND_CUDA -I. -o tests/test_kernels_cuda \
 *               tests/test_kernels.c backends/cuda/backend.o   \
 *               -L<cuda>/lib64 -lcudart -lstdc++ -lm
 *   IRON: gcc -O3 -march=native -DBACKEND_IRON -I. -o tests/test_kernels_iron \
 *               tests/test_kernels.c backends/iron/backend.o -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "backends/model_types.h"
#include "backends/backend.h"

/* ---- Tiny LCG for deterministic random inputs ---- */
static uint32_t g_rng = 0xdeadbeef;
static float rng_f(void) {
    g_rng = g_rng * 1664525u + 1013904223u;
    return (float)(int32_t)g_rng * (1.0f / 2147483648.0f);
}
static uint16_t rng_bf16(void) {
    float f = rng_f() * 0.1f;   /* small values, no overflow */
    uint32_t u; memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);
}

/* ---- Test dimensions (small enough that CPU runs in < 1 s) ---- */
#define T_LN     8
#define D_LN     64
#define T_LIN    4
#define IN_LIN   64
#define OUT_LIN  128
#define N_ADD    256
#define S_CAUSAL 8
#define T_RPOS   4           /* rel_pos_emb T; output is (2T-1)*ENC_D = 7*1280 */
#define T_SDP_Q  4
#define T_SDP_K  6           /* sdp_attn uses DEC_D/DEC_H/DEC_DK from model_types.h */

/* ---- BF16 weight blob ---- */
/* Contiguous buffer whose host address is the "safetensors base" for the test. */
#define BLOB_LN_W      0
#define BLOB_LN_B      (BLOB_LN_W  + D_LN)
#define BLOB_LIN_W     (BLOB_LN_B  + D_LN)
#define BLOB_LIN_B     (BLOB_LIN_W + (size_t)OUT_LIN * IN_LIN)
#define BLOB_TOTAL     (BLOB_LIN_B + OUT_LIN)

static uint16_t g_blob[BLOB_TOTAL];

/* ---- Output ---- */
static FILE *g_out;

/* Transfer n floats from device to host and append to output file. */
static void dump(const float *dev, size_t n, const char *label) {
    float *h = (float *)malloc(n * sizeof(float));
    if (!h) { fputs("OOM\n", stderr); exit(1); }
    backend_dtoh(h, dev, n * sizeof(float));
    fwrite(h, sizeof(float), n, g_out);
    free(h);
    fprintf(stderr, "  %-32s %6zu floats\n", label, n);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <output.bin>\n", argv[0]);
        return 1;
    }
    g_out = fopen(argv[1], "wb");
    if (!g_out) { perror(argv[1]); return 1; }

    /* Fill the weight blob with deterministic random BF16 values.
     * backend_weights_upload_blob must come before backend_init so that
     * backend_init's cudaSetDevice(0) is the last device selection before
     * backend_arena_alloc, keeping the arena on the correct device. */
    for (size_t i = 0; i < BLOB_TOTAL; i++) g_blob[i] = rng_bf16();
    backend_weights_upload_blob(g_blob, BLOB_TOTAL * sizeof(uint16_t));

    backend_init(0);

    /* Host weight pointers into g_blob.  The backend kernel wrappers (backend_linear,
     * backend_layernorm, etc.) call dev_w() internally to convert host→device, so we
     * must hand them host pointers — exactly the same pattern as the main inference code
     * (model.h always keeps weight pointers as host pointers and never calls
     * backend_device_ptr / patch_weights_to_device). */
    const uint16_t *dw_ln_w  = g_blob + BLOB_LN_W;
    const uint16_t *dw_ln_b  = g_blob + BLOB_LN_B;
    const uint16_t *dw_lin_W = g_blob + BLOB_LIN_W;
    const uint16_t *dw_lin_b = g_blob + BLOB_LIN_B;

    /* ---- Single arena, partitioned by hand (mirrors transcribe_chunk style) ---- */
    size_t arena_floats =
        (size_t)2 * T_LN  * D_LN   +     /* layernorm: src + dst          */
        (size_t)T_LIN * IN_LIN      +     /* linear x                      */
        (size_t)T_LIN * OUT_LIN     +     /* linear y        (block 2)     */
        (size_t)T_LIN * IN_LIN      +     /* linear_silu x   (block 3)     */
        (size_t)T_LIN * OUT_LIN     +     /* linear_silu y                 */
        (size_t)T_LIN * IN_LIN      +     /* linear_relu x   (block 4)     */
        (size_t)T_LIN * OUT_LIN     +     /* linear_relu y                 */
        (size_t)T_LIN * IN_LIN      +     /* linear_fmadd x  (block 5)     */
        (size_t)T_LIN * OUT_LIN     +     /* linear_fmadd acc              */
        (size_t)2 * N_ADD           +     /* add_inplace: dst + src        */
        (size_t)S_CAUSAL * S_CAUSAL +     /* causal mask                   */
        (size_t)(2*T_RPOS - 1) * ENC_D + /* rel_pos_emb output            */
        (size_t)T_SDP_Q * DEC_D    +     /* sdp out                       */
        (size_t)T_SDP_Q * DEC_D    +     /* sdp q                         */
        (size_t)T_SDP_K * DEC_D    +     /* sdp k                         */
        (size_t)T_SDP_K * DEC_D    +     /* sdp v                         */
        (size_t)T_SDP_Q * T_SDP_K  +     /* sdp work (score scratch)      */
        1024;                             /* headroom                      */

    float *arena = backend_arena_alloc(arena_floats);
    float *p = arena;
#define TAKE(n) ((p += (size_t)(n)), (p - (size_t)(n)))

    /* ================================================================
     * 1. backend_layernorm
     * ================================================================ */
    fprintf(stderr, "backend_layernorm        T=%d D=%d\n", T_LN, D_LN);
    {
        float *src = TAKE(T_LN * D_LN);
        float *dst = TAKE(T_LN * D_LN);

        float *h = (float *)malloc((size_t)T_LN * D_LN * sizeof(float));
        for (int i = 0; i < T_LN * D_LN; i++) h[i] = rng_f();
        backend_htod(src, h, T_LN * D_LN);
        free(h);

        backend_layernorm(dst, src, dw_ln_w, dw_ln_b, T_LN, D_LN);
        dump(dst, (size_t)T_LN * D_LN, "layernorm");
    }

    /* ================================================================
     * 2. backend_linear
     * ================================================================ */
    fprintf(stderr, "backend_linear           T=%d in=%d out=%d\n",
            T_LIN, IN_LIN, OUT_LIN);
    {
        float *x = TAKE(T_LIN * IN_LIN);
        float *y = TAKE(T_LIN * OUT_LIN);

        float *h = (float *)malloc((size_t)T_LIN * IN_LIN * sizeof(float));
        for (int i = 0; i < T_LIN * IN_LIN; i++) h[i] = rng_f();
        backend_htod(x, h, T_LIN * IN_LIN);
        free(h);

        backend_linear(y, x, dw_lin_W, dw_lin_b, T_LIN, IN_LIN, OUT_LIN);
        dump(y, (size_t)T_LIN * OUT_LIN, "linear");
    }

    /* ================================================================
     * 3. backend_linear_silu
     * ================================================================ */
    fprintf(stderr, "backend_linear_silu\n");
    {
        float *x = TAKE(T_LIN * IN_LIN);
        float *y = TAKE(T_LIN * OUT_LIN);

        float *h = (float *)malloc((size_t)T_LIN * IN_LIN * sizeof(float));
        for (int i = 0; i < T_LIN * IN_LIN; i++) h[i] = rng_f();
        backend_htod(x, h, T_LIN * IN_LIN);
        free(h);

        backend_linear_silu(y, x, dw_lin_W, dw_lin_b, T_LIN, IN_LIN, OUT_LIN);
        dump(y, (size_t)T_LIN * OUT_LIN, "linear_silu");
    }

    /* ================================================================
     * 4. backend_linear_relu
     * ================================================================ */
    fprintf(stderr, "backend_linear_relu\n");
    {
        float *x = TAKE(T_LIN * IN_LIN);
        float *y = TAKE(T_LIN * OUT_LIN);

        float *h = (float *)malloc((size_t)T_LIN * IN_LIN * sizeof(float));
        for (int i = 0; i < T_LIN * IN_LIN; i++) h[i] = rng_f();
        backend_htod(x, h, T_LIN * IN_LIN);
        free(h);

        backend_linear_relu(y, x, dw_lin_W, dw_lin_b, T_LIN, IN_LIN, OUT_LIN);
        dump(y, (size_t)T_LIN * OUT_LIN, "linear_relu");
    }

    /* ================================================================
     * 5. backend_linear_fmadd  (acc += 0.5 * (x @ W + b))
     * ================================================================ */
    fprintf(stderr, "backend_linear_fmadd\n");
    {
        float *x   = TAKE(T_LIN * IN_LIN);
        float *acc = TAKE(T_LIN * OUT_LIN);

        size_t nx  = (size_t)T_LIN * IN_LIN;
        size_t nacc = (size_t)T_LIN * OUT_LIN;
        float *hx   = (float *)malloc(nx   * sizeof(float));
        float *hacc = (float *)malloc(nacc * sizeof(float));
        for (size_t i = 0; i < nx;   i++) hx[i]   = rng_f();
        for (size_t i = 0; i < nacc; i++) hacc[i] = rng_f();
        backend_htod(x,   hx,   nx);
        backend_htod(acc, hacc, nacc);
        free(hx); free(hacc);

        backend_linear_fmadd(acc, 0.5f, x, dw_lin_W, dw_lin_b, T_LIN, IN_LIN, OUT_LIN);
        dump(acc, nacc, "linear_fmadd");
    }

    /* ================================================================
     * 6. backend_add_inplace  (dst += src)
     * ================================================================ */
    fprintf(stderr, "backend_add_inplace      n=%d\n", N_ADD);
    {
        float *dst = TAKE(N_ADD);
        float *src = TAKE(N_ADD);

        float *hd = (float *)malloc((size_t)N_ADD * sizeof(float));
        float *hs = (float *)malloc((size_t)N_ADD * sizeof(float));
        for (int i = 0; i < N_ADD; i++) { hd[i] = rng_f(); hs[i] = rng_f(); }
        backend_htod(dst, hd, N_ADD);
        backend_htod(src, hs, N_ADD);
        free(hd); free(hs);

        backend_add_inplace(dst, src, N_ADD);
        dump(dst, N_ADD, "add_inplace");
    }

    /* ================================================================
     * 7. backend_fill_causal_mask
     * ================================================================ */
    fprintf(stderr, "backend_fill_causal_mask S=%d\n", S_CAUSAL);
    {
        float *mask = TAKE(S_CAUSAL * S_CAUSAL);
        backend_fill_causal_mask(mask, S_CAUSAL);
        dump(mask, (size_t)S_CAUSAL * S_CAUSAL, "fill_causal_mask");
    }

    /* ================================================================
     * 8. backend_make_rel_pos_emb
     * ================================================================ */
    fprintf(stderr, "backend_make_rel_pos_emb T=%d  out=%d floats\n",
            T_RPOS, (2*T_RPOS - 1) * ENC_D);
    {
        size_t pe_n = (size_t)(2 * T_RPOS - 1) * ENC_D;
        float *pe = TAKE(pe_n);
        backend_make_rel_pos_emb(pe, T_RPOS);
        dump(pe, pe_n, "make_rel_pos_emb");
    }

    /* ================================================================
     * 9. backend_sdp_attn
     * Uses the real decoder head dimensions: DEC_D, DEC_H, DEC_DK.
     * q/k/v are random float32 (no BF16 weights involved).
     * ================================================================ */
    fprintf(stderr, "backend_sdp_attn         T_q=%d T_k=%d D=%d\n",
            T_SDP_Q, T_SDP_K, DEC_D);
    {
        float *out  = TAKE((size_t)T_SDP_Q * DEC_D);
        float *q    = TAKE((size_t)T_SDP_Q * DEC_D);
        float *k    = TAKE((size_t)T_SDP_K * DEC_D);
        float *v    = TAKE((size_t)T_SDP_K * DEC_D);
        float *work = TAKE((size_t)T_SDP_Q * T_SDP_K);

        float *hq = (float *)malloc((size_t)T_SDP_Q * DEC_D * sizeof(float));
        float *hk = (float *)malloc((size_t)T_SDP_K * DEC_D * sizeof(float));
        float *hv = (float *)malloc((size_t)T_SDP_K * DEC_D * sizeof(float));
        if (!hq || !hk || !hv) { fputs("OOM\n", stderr); exit(1); }

        /* Scale q/k small to keep softmax numerically stable */
        for (int i = 0; i < T_SDP_Q * DEC_D; i++) hq[i] = rng_f() * 0.1f;
        for (int i = 0; i < T_SDP_K * DEC_D; i++) hk[i] = rng_f() * 0.1f;
        for (int i = 0; i < T_SDP_K * DEC_D; i++) hv[i] = rng_f();

        backend_htod(q, hq, (size_t)T_SDP_Q * DEC_D);
        backend_htod(k, hk, (size_t)T_SDP_K * DEC_D);
        backend_htod(v, hv, (size_t)T_SDP_K * DEC_D);
        free(hq); free(hk); free(hv);

        backend_sdp_attn(out, q, k, v, NULL, T_SDP_Q, T_SDP_K, work);
        dump(out, (size_t)T_SDP_Q * DEC_D, "sdp_attn");
    }

#undef TAKE

    backend_arena_free(arena);
    backend_destroy();
    fclose(g_out);

    fprintf(stderr, "\nDone. Results written to %s\n", argv[1]);
    return 0;
}
