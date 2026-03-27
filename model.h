#ifndef MODEL_H
#define MODEL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/* ============================================================
 * Model constants
 * ============================================================ */
#define ENC_D    1280
#define ENC_FF   5120
#define ENC_H    8
#define ENC_DK   160       /* ENC_D / ENC_H */
#define ENC_N    48
#define ENC_CK   9
#define SUB_CH   256       /* subsampling conv channels */

#define DEC_D    1024
#define DEC_FF   4096
#define DEC_H    8
#define DEC_DK   128       /* DEC_D / DEC_H */
#define DEC_N    8

#define VOCAB    16384
#define MAX_SEQ  1024
#define MAX_VOCAB 16384

/* ============================================================
 * Prompt token IDs (fixed positions in vocab.txt)
 * ============================================================ */
#define TOK_EOS              3
#define TOK_STARTOFTRANSCRIPT 4
#define TOK_PNC              5
#define TOK_NOPNC            6
#define TOK_STARTOFCONTEXT   7
#define TOK_ITN              8
#define TOK_NOITN            9
#define TOK_TIMESTAMP       10
#define TOK_NOTIMESTAMP     11
#define TOK_DIARIZE         12
#define TOK_NODIARIZE       13
#define TOK_EMO_UNDEFINED   16
#define TOK_LEADING_SPACE 13764

#define SR       16000
#define N_FFT    512
#define HOP_LEN  160
#define WIN_LEN  400
#define N_MELS   128
#define N_BINS   257

/* ============================================================
 * BF16 utilities
 * ============================================================ */
static inline float bf16(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

/* ============================================================
 * Safetensors loader
 * ============================================================ */
#ifndef EMBEDDED_MODEL
/* ── safetensors loader (file path, runtime JSON parsing) ─────────────────── */
static char    *sf_hdr;
static uint8_t *sf_data;

static void sf_load(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); exit(1); }
    struct stat st; fstat(fd, &st);
    uint8_t *m = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) { perror("mmap"); exit(1); }
    uint64_t hs; memcpy(&hs, m, 8);
    sf_hdr  = malloc(hs + 1);
    memcpy(sf_hdr, m + 8, hs);
    sf_hdr[hs] = '\0';
    sf_data = m + 8 + hs;
}

static const uint16_t *sf_get(const char *name, size_t *nelems) {
    char key[512];
    snprintf(key, sizeof(key), "\"%s\":", name);
    /* Search for the key, but verify it is preceded by '{' or ',' (not a prefix match) */
    char *pos = sf_hdr;
    for (;;) {
        pos = strstr(pos, key);
        if (!pos) { fprintf(stderr, "tensor not found: %s\n", name); exit(1); }
        if (pos == sf_hdr || pos[-1] == '{' || pos[-1] == ',') break;
        pos++;  /* skip this false match and keep searching */
    }
    char *d = strstr(pos, "\"data_offsets\"");
    if (!d)   { fprintf(stderr, "no data_offsets: %s\n", name);  exit(1); }
    char *br = strchr(d, '[');
    if (!br)  { fprintf(stderr, "malformed data_offsets: %s\n", name); exit(1); }
    long s, e;
    if (sscanf(br + 1, "%ld, %ld", &s, &e) != 2 || s < 0 || e < s) {
        fprintf(stderr, "bad data_offsets for %s\n", name); exit(1);
    }
    if (nelems) *nelems = (size_t)(e - s) / 2;
    return (const uint16_t *)(sf_data + s);
}
#endif /* !EMBEDDED_MODEL */

/* ============================================================
 * Weight structs
 * ============================================================ */
typedef struct {
    const uint16_t *ff1_lnw, *ff1_lnb, *ff1_w1, *ff1_b1, *ff1_w2, *ff1_b2;
    const uint16_t *sa_lnw, *sa_lnb, *sa_qw, *sa_qb, *sa_kw, *sa_kb,
                   *sa_vw, *sa_vb, *sa_pw, *sa_ow, *sa_ob, *sa_pbu, *sa_pbv;
    const uint16_t *cv_lnw, *cv_lnb, *cv_pw1w, *cv_pw1b, *cv_dww, *cv_dwb,
                   *cv_bnw, *cv_bnb, *cv_bnmean, *cv_bnvar, *cv_pw2w, *cv_pw2b;
    const uint16_t *ff2_lnw, *ff2_lnb, *ff2_w1, *ff2_b1, *ff2_w2, *ff2_b2;
    const uint16_t *out_lnw, *out_lnb;
} ELayer;

typedef struct {
    const uint16_t *sa_lnw, *sa_lnb, *sa_qw, *sa_qb, *sa_kw, *sa_kb,
                   *sa_vw, *sa_vb, *sa_ow, *sa_ob;
    const uint16_t *ca_lnw, *ca_lnb, *ca_qw, *ca_qb, *ca_kw, *ca_kb,
                   *ca_vw, *ca_vb, *ca_ow, *ca_ob;
    const uint16_t *ffn_lnw, *ffn_lnb, *ffn_w1, *ffn_b1, *ffn_w2, *ffn_b2;
} DLayer;

typedef struct {
    const uint16_t *s0w, *s0b, *s2w, *s2b, *s3w, *s3b, *s5w, *s5b, *s6w, *s6b, *sow, *sob;
    ELayer enc[ENC_N];
    const uint16_t *prw, *prb;
    const uint16_t *etw, *epw, *elnw, *elnb;
    DLayer dec[DEC_N];
    const uint16_t *dlnw, *dlnb, *hdb;
} Weights;

static Weights W;

#ifdef EMBEDDED_MODEL
/* ── embedded path: precomputed offsets, no parsing, no allocation ─────────── */
__asm__(
    ".section .rodata\n"
    ".balign 8\n"
    ".global sf_embedded_data\n"
    "sf_embedded_data:\n"
    ".incbin \"model_files/model.safetensors\"\n"
    ".global sf_embedded_data_end\n"
    "sf_embedded_data_end:\n"
    ".previous\n"
);
extern const uint8_t sf_embedded_data[];
#include "model_offsets.h"
#define SP(off) ((const uint16_t *)(sf_embedded_data + (off)))
static void load_weights(void) {
    W.s0w  = SP(SFOFF_s0w);  W.s0b  = SP(SFOFF_s0b);
    W.s2w  = SP(SFOFF_s2w);  W.s2b  = SP(SFOFF_s2b);
    W.s3w  = SP(SFOFF_s3w);  W.s3b  = SP(SFOFF_s3b);
    W.s5w  = SP(SFOFF_s5w);  W.s5b  = SP(SFOFF_s5b);
    W.s6w  = SP(SFOFF_s6w);  W.s6b  = SP(SFOFF_s6b);
    W.sow  = SP(SFOFF_sow);  W.sob  = SP(SFOFF_sob);
    for (int i = 0; i < ENC_N; i++) {
        ELayer *L = &W.enc[i];
        L->ff1_lnw  = SP(SFOFF_enc_ff1_lnw[i]);  L->ff1_lnb  = SP(SFOFF_enc_ff1_lnb[i]);
        L->ff1_w1   = SP(SFOFF_enc_ff1_w1[i]);   L->ff1_b1   = SP(SFOFF_enc_ff1_b1[i]);
        L->ff1_w2   = SP(SFOFF_enc_ff1_w2[i]);   L->ff1_b2   = SP(SFOFF_enc_ff1_b2[i]);
        L->sa_lnw   = SP(SFOFF_enc_sa_lnw[i]);   L->sa_lnb   = SP(SFOFF_enc_sa_lnb[i]);
        L->sa_qw    = SP(SFOFF_enc_sa_qw[i]);    L->sa_qb    = SP(SFOFF_enc_sa_qb[i]);
        L->sa_kw    = SP(SFOFF_enc_sa_kw[i]);    L->sa_kb    = SP(SFOFF_enc_sa_kb[i]);
        L->sa_vw    = SP(SFOFF_enc_sa_vw[i]);    L->sa_vb    = SP(SFOFF_enc_sa_vb[i]);
        L->sa_pw    = SP(SFOFF_enc_sa_pw[i]);
        L->sa_ow    = SP(SFOFF_enc_sa_ow[i]);    L->sa_ob    = SP(SFOFF_enc_sa_ob[i]);
        L->sa_pbu   = SP(SFOFF_enc_sa_pbu[i]);   L->sa_pbv   = SP(SFOFF_enc_sa_pbv[i]);
        L->cv_lnw   = SP(SFOFF_enc_cv_lnw[i]);   L->cv_lnb   = SP(SFOFF_enc_cv_lnb[i]);
        L->cv_pw1w  = SP(SFOFF_enc_cv_pw1w[i]);  L->cv_pw1b  = SP(SFOFF_enc_cv_pw1b[i]);
        L->cv_dww   = SP(SFOFF_enc_cv_dww[i]);   L->cv_dwb   = SP(SFOFF_enc_cv_dwb[i]);
        L->cv_bnw   = SP(SFOFF_enc_cv_bnw[i]);   L->cv_bnb   = SP(SFOFF_enc_cv_bnb[i]);
        L->cv_bnmean= SP(SFOFF_enc_cv_bnmean[i]);L->cv_bnvar = SP(SFOFF_enc_cv_bnvar[i]);
        L->cv_pw2w  = SP(SFOFF_enc_cv_pw2w[i]);  L->cv_pw2b  = SP(SFOFF_enc_cv_pw2b[i]);
        L->ff2_lnw  = SP(SFOFF_enc_ff2_lnw[i]);  L->ff2_lnb  = SP(SFOFF_enc_ff2_lnb[i]);
        L->ff2_w1   = SP(SFOFF_enc_ff2_w1[i]);   L->ff2_b1   = SP(SFOFF_enc_ff2_b1[i]);
        L->ff2_w2   = SP(SFOFF_enc_ff2_w2[i]);   L->ff2_b2   = SP(SFOFF_enc_ff2_b2[i]);
        L->out_lnw  = SP(SFOFF_enc_out_lnw[i]);  L->out_lnb  = SP(SFOFF_enc_out_lnb[i]);
    }
    W.prw  = SP(SFOFF_prw);   W.prb  = SP(SFOFF_prb);
    W.etw  = SP(SFOFF_etw);   W.epw  = SP(SFOFF_epw);
    W.elnw = SP(SFOFF_elnw);  W.elnb = SP(SFOFF_elnb);
    for (int i = 0; i < DEC_N; i++) {
        DLayer *L = &W.dec[i];
        L->sa_lnw  = SP(SFOFF_dec_sa_lnw[i]);  L->sa_lnb  = SP(SFOFF_dec_sa_lnb[i]);
        L->sa_qw   = SP(SFOFF_dec_sa_qw[i]);   L->sa_qb   = SP(SFOFF_dec_sa_qb[i]);
        L->sa_kw   = SP(SFOFF_dec_sa_kw[i]);   L->sa_kb   = SP(SFOFF_dec_sa_kb[i]);
        L->sa_vw   = SP(SFOFF_dec_sa_vw[i]);   L->sa_vb   = SP(SFOFF_dec_sa_vb[i]);
        L->sa_ow   = SP(SFOFF_dec_sa_ow[i]);   L->sa_ob   = SP(SFOFF_dec_sa_ob[i]);
        L->ca_lnw  = SP(SFOFF_dec_ca_lnw[i]);  L->ca_lnb  = SP(SFOFF_dec_ca_lnb[i]);
        L->ca_qw   = SP(SFOFF_dec_ca_qw[i]);   L->ca_qb   = SP(SFOFF_dec_ca_qb[i]);
        L->ca_kw   = SP(SFOFF_dec_ca_kw[i]);   L->ca_kb   = SP(SFOFF_dec_ca_kb[i]);
        L->ca_vw   = SP(SFOFF_dec_ca_vw[i]);   L->ca_vb   = SP(SFOFF_dec_ca_vb[i]);
        L->ca_ow   = SP(SFOFF_dec_ca_ow[i]);   L->ca_ob   = SP(SFOFF_dec_ca_ob[i]);
        L->ffn_lnw = SP(SFOFF_dec_ffn_lnw[i]); L->ffn_lnb = SP(SFOFF_dec_ffn_lnb[i]);
        L->ffn_w1  = SP(SFOFF_dec_ffn_w1[i]);  L->ffn_b1  = SP(SFOFF_dec_ffn_b1[i]);
        L->ffn_w2  = SP(SFOFF_dec_ffn_w2[i]);  L->ffn_b2  = SP(SFOFF_dec_ffn_b2[i]);
    }
    W.dlnw = SP(SFOFF_dlnw);  W.dlnb = SP(SFOFF_dlnb);
    W.hdb  = SP(SFOFF_hdb);
}
#undef SP

#else
/* ── file path: load and parse safetensors header at runtime ──────────────── */
static void load_weights(void) {
    sf_load("model_files/model.safetensors");
    W.s0w = sf_get("encoder.pre_encode.conv.0.weight", NULL);
    W.s0b = sf_get("encoder.pre_encode.conv.0.bias",   NULL);
    W.s2w = sf_get("encoder.pre_encode.conv.2.weight", NULL);
    W.s2b = sf_get("encoder.pre_encode.conv.2.bias",   NULL);
    W.s3w = sf_get("encoder.pre_encode.conv.3.weight", NULL);
    W.s3b = sf_get("encoder.pre_encode.conv.3.bias",   NULL);
    W.s5w = sf_get("encoder.pre_encode.conv.5.weight", NULL);
    W.s5b = sf_get("encoder.pre_encode.conv.5.bias",   NULL);
    W.s6w = sf_get("encoder.pre_encode.conv.6.weight", NULL);
    W.s6b = sf_get("encoder.pre_encode.conv.6.bias",   NULL);
    W.sow = sf_get("encoder.pre_encode.out.weight",    NULL);
    W.sob = sf_get("encoder.pre_encode.out.bias",      NULL);
    for (int i = 0; i < ENC_N; i++) {
        char n[256]; ELayer *L = &W.enc[i];
#define EG(f,s) snprintf(n,sizeof(n),"encoder.layers.%d." s,i); L->f=sf_get(n,NULL)
        EG(ff1_lnw,"norm_feed_forward1.weight"); EG(ff1_lnb,"norm_feed_forward1.bias");
        EG(ff1_w1,"feed_forward1.linear1.weight"); EG(ff1_b1,"feed_forward1.linear1.bias");
        EG(ff1_w2,"feed_forward1.linear2.weight"); EG(ff1_b2,"feed_forward1.linear2.bias");
        EG(sa_lnw,"norm_self_att.weight"); EG(sa_lnb,"norm_self_att.bias");
        EG(sa_qw,"self_attn.linear_q.weight"); EG(sa_qb,"self_attn.linear_q.bias");
        EG(sa_kw,"self_attn.linear_k.weight"); EG(sa_kb,"self_attn.linear_k.bias");
        EG(sa_vw,"self_attn.linear_v.weight"); EG(sa_vb,"self_attn.linear_v.bias");
        EG(sa_pw,"self_attn.linear_pos.weight");
        EG(sa_ow,"self_attn.linear_out.weight"); EG(sa_ob,"self_attn.linear_out.bias");
        EG(sa_pbu,"self_attn.pos_bias_u"); EG(sa_pbv,"self_attn.pos_bias_v");
        EG(cv_lnw,"norm_conv.weight"); EG(cv_lnb,"norm_conv.bias");
        EG(cv_pw1w,"conv.pointwise_conv1.weight"); EG(cv_pw1b,"conv.pointwise_conv1.bias");
        EG(cv_dww,"conv.depthwise_conv.weight"); EG(cv_dwb,"conv.depthwise_conv.bias");
        EG(cv_bnw,"conv.batch_norm.weight"); EG(cv_bnb,"conv.batch_norm.bias");
        EG(cv_bnmean,"conv.batch_norm.running_mean"); EG(cv_bnvar,"conv.batch_norm.running_var");
        EG(cv_pw2w,"conv.pointwise_conv2.weight"); EG(cv_pw2b,"conv.pointwise_conv2.bias");
        EG(ff2_lnw,"norm_feed_forward2.weight"); EG(ff2_lnb,"norm_feed_forward2.bias");
        EG(ff2_w1,"feed_forward2.linear1.weight"); EG(ff2_b1,"feed_forward2.linear1.bias");
        EG(ff2_w2,"feed_forward2.linear2.weight"); EG(ff2_b2,"feed_forward2.linear2.bias");
        EG(out_lnw,"norm_out.weight"); EG(out_lnb,"norm_out.bias");
#undef EG
    }
    W.prw  = sf_get("encoder_decoder_proj.weight", NULL);
    W.prb  = sf_get("encoder_decoder_proj.bias",   NULL);
    W.etw  = sf_get("transf_decoder._embedding.token_embedding.weight",     NULL);
    W.epw  = sf_get("transf_decoder._embedding.position_embedding.pos_enc", NULL);
    W.elnw = sf_get("transf_decoder._embedding.layer_norm.weight",          NULL);
    W.elnb = sf_get("transf_decoder._embedding.layer_norm.bias",            NULL);
    for (int i = 0; i < DEC_N; i++) {
        char n[256], p[128]; DLayer *L = &W.dec[i];
        snprintf(p,sizeof(p),"transf_decoder._decoder.layers.%d.",i);
#define DG(f,s) snprintf(n,sizeof(n),"%s" s,p); L->f=sf_get(n,NULL)
        DG(sa_lnw,"layer_norm_1.weight"); DG(sa_lnb,"layer_norm_1.bias");
        DG(sa_qw,"first_sub_layer.query_net.weight"); DG(sa_qb,"first_sub_layer.query_net.bias");
        DG(sa_kw,"first_sub_layer.key_net.weight"); DG(sa_kb,"first_sub_layer.key_net.bias");
        DG(sa_vw,"first_sub_layer.value_net.weight"); DG(sa_vb,"first_sub_layer.value_net.bias");
        DG(sa_ow,"first_sub_layer.out_projection.weight"); DG(sa_ob,"first_sub_layer.out_projection.bias");
        DG(ca_lnw,"layer_norm_2.weight"); DG(ca_lnb,"layer_norm_2.bias");
        DG(ca_qw,"second_sub_layer.query_net.weight"); DG(ca_qb,"second_sub_layer.query_net.bias");
        DG(ca_kw,"second_sub_layer.key_net.weight"); DG(ca_kb,"second_sub_layer.key_net.bias");
        DG(ca_vw,"second_sub_layer.value_net.weight"); DG(ca_vb,"second_sub_layer.value_net.bias");
        DG(ca_ow,"second_sub_layer.out_projection.weight"); DG(ca_ob,"second_sub_layer.out_projection.bias");
        DG(ffn_lnw,"layer_norm_3.weight"); DG(ffn_lnb,"layer_norm_3.bias");
        DG(ffn_w1,"third_sub_layer.dense_in.weight"); DG(ffn_b1,"third_sub_layer.dense_in.bias");
        DG(ffn_w2,"third_sub_layer.dense_out.weight"); DG(ffn_b2,"third_sub_layer.dense_out.bias");
#undef DG
    }
    W.dlnw = sf_get("transf_decoder._decoder.final_layer_norm.weight", NULL);
    W.dlnb = sf_get("transf_decoder._decoder.final_layer_norm.bias",   NULL);
    W.hdb  = sf_get("log_softmax.mlp.layer0.bias",                     NULL);
}
#endif /* EMBEDDED_MODEL */

/* ============================================================
 * Math primitives
 * ============================================================ */
static void layernorm(float *dst, const float *src, const uint16_t *w, const uint16_t *b, int T, int D) {
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

static void linear(float *y, const float *x, const uint16_t *Wb, const uint16_t *bb,
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

static void silu(float *x, int n) {
    for (int i = 0; i < n; i++) x[i] = x[i] / (1.0f + expf(-x[i]));
}

static void linear_silu(float *y, const float *x, const uint16_t *Wb, const uint16_t *bb,
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

static void linear_relu(float *y, const float *x, const uint16_t *Wb, const uint16_t *bb,
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

static void linear_fmadd(float *acc, float scale, const float *x, const uint16_t *Wb, const uint16_t *bb,
                         int T, int in_d, int out_d) {
    for (int t = 0; t < T; t++) {
        const float    *xr = x   + t * in_d;
        float          *ar = acc + t * out_d;
        for (int j = 0; j < out_d; j++) {
            float s = bf16(bb[j]);
            const uint16_t *wr = Wb + j * in_d;
            for (int k = 0; k < in_d; k++) s += xr[k] * bf16(wr[k]);
            ar[j] += scale * s;
        }
    }
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

/* ============================================================
 * 2D convolution
 * in: (in_c, in_h, in_w)  W: (out_c, in_c/groups, kH, kW) BF16
 * out: (out_c, out_h, out_w)
 * ============================================================ */
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

/* ============================================================
 * Encoder subsampling: mel(N_MELS, T_mel) → x_enc(T_enc, ENC_D)
 *
 * work layout (all floats, total = B0+B1 + SUB_CH²+SUB_CH):
 *
 *   B0 = SUB_CH × T1 × 64                     (≈ 8.2M for 10 s audio)
 *   B1 = SUB_CH × T2 × 32                     (≈ 2.1M)
 *
 *   Stage 0  conv0 (mel_t → buf0):
 *     buf0  at work[0 .. B0-1]                 SUB_CH × T1 × 64  written
 *     mel_t at work[B0 .. B0+T_mel*N_MELS-1]  T_mel × 128       read
 *     [mel_t fits inside the B1 region since B1 ≥ T_mel×128]
 *     (conv weights/biases read inline from bf16, no scratch)
 *
 *   Stage 1  conv2 (buf0 → buf1):
 *     buf0  at work[0 .. B0-1]                 read → consumed
 *     buf1  at work[B0 .. B0+B1-1]             SUB_CH × T2 × 32  written (reuses mel_t space)
 *     PEAK: B0 + B1
 *
 *   Stage 2  conv3 (buf1 → buf1b):
 *     buf1  at work[B0 .. B0+B1-1]             read → consumed
 *     buf1b at work[0 .. B1-1]                 reuses buf0 space
 *
 *   Stage 3  conv5 (buf1b → buf2):
 *     buf1b at work[0 .. B1-1]                 read → consumed
 *     buf2  at work[B1 .. B1+B2-1]             B2 = SUB_CH × T_enc × 16 = T_enc × 4096
 *
 *   Stage 4  conv6 (buf2 → buf2b):
 *     buf2  at work[B1 .. B1+B2-1]             read → consumed
 *     buf2b at work[0 .. B2-1]                 reuses buf1b space
 *
 *   Stage 5  reshape (buf2b → flat):
 *     buf2b at work[0 .. B2-1]                 read → consumed
 *     flat  at work[ENC_D*4096 .. ENC_D*4096+B2-1]  T_enc × 4096
 *
 *   Stage 5  reshape (buf2b → flat):
 *     buf2b at work[0 .. B2-1]                 read → consumed
 *     flat  at work[B2 .. 2*B2-1]              T_enc × 4096       written
 *
 *   Stage 6  linear (flat → x_enc):
 *     flat  at work[B2 .. 2*B2-1]              read → consumed
 *     x_enc in permanent arena region           written
 *     (linear weights read inline from bf16, no scratch; B0+B1 >> 2*B2)
 * ============================================================ */
static void encoder_subsampling(const float *mel, int T_mel,
                                 float *x_enc, int T_enc, float *work) {
    const int T1 = (T_mel+1)/2, T2 = (T1+1)/2;
    const int F1 = 64, F2 = 32, F3 = 16;
    const int in_feat = SUB_CH * F3;  /* 4096 */
    const size_t B0 = (size_t)SUB_CH * T1 * F1;  /* SUB_CH × T1 × 64 */
    const size_t B1 = (size_t)SUB_CH * T2 * F2;  /* SUB_CH × T2 × 32; ≥ T_mel×128 */
    const size_t B2 = (size_t)SUB_CH * T_enc * F3; /* T_enc × 4096 */

    float *buf0  = work;
    float *mel_t = work + B0;  /* T_mel × N_MELS, inside B1 region */

    /* Transpose mel: (N_MELS, T_mel) → (T_mel, N_MELS) */
    for (int t = 0; t < T_mel; t++)
        for (int m = 0; m < N_MELS; m++)
            mel_t[t * N_MELS + m] = mel[m * T_mel + t];

    /* Stage 0: Conv2d(1→256, 3×3, stride=2, pad=1) */
    conv2d(buf0, mel_t, W.s0w, W.s0b, 1, T_mel, N_MELS, SUB_CH, 3, 3, 2, 1, 1);
    relu(buf0, SUB_CH * T1 * F1);

    /* Stage 1: DepthwiseConv2d(256, 3×3, stride=2, pad=1) — buf0 → buf1 */
    float *buf1 = work + B0;  /* reuses mel_t space */
    conv2d(buf1, buf0, W.s2w, W.s2b, SUB_CH, T1, F1, SUB_CH, 3, 3, 2, 1, SUB_CH);
    /* buf0 consumed */

    /* Stage 2: PointwiseConv2d(256→256, 1×1) — buf1 → buf1b */
    float *buf1b = work;  /* reuses buf0 space */
    conv2d(buf1b, buf1, W.s3w, W.s3b, SUB_CH, T2, F2, SUB_CH, 1, 1, 1, 0, 1);
    relu(buf1b, SUB_CH * T2 * F2);
    /* buf1 consumed */

    /* Stage 3: DepthwiseConv2d(256, 3×3, stride=2, pad=1) — buf1b → buf2 */
    float *buf2 = work + B1;  /* starts after buf1b */
    conv2d(buf2, buf1b, W.s5w, W.s5b, SUB_CH, T2, F2, SUB_CH, 3, 3, 2, 1, SUB_CH);
    /* buf1b consumed */

    /* Stage 4: PointwiseConv2d(256→256, 1×1) — buf2 → buf2b */
    float *buf2b = work;  /* reuses buf1b space */
    conv2d(buf2b, buf2, W.s6w, W.s6b, SUB_CH, T_enc, F3, SUB_CH, 1, 1, 1, 0, 1);
    relu(buf2b, SUB_CH * T_enc * F3);
    /* buf2 consumed */

    /* Stage 5: Reshape (SUB_CH, T_enc, F3) → (T_enc, SUB_CH×F3=4096)
     * buf2b[ch][t][f] = buf2b[ch*(T_enc*F3) + t*F3 + f]
     * flat[t][ch*F3+f] = flat[t*in_feat + ch*F3 + f]  */
    float *flat = work + B2;  /* T_enc × 4096; B0+B1 >> 2*B2 so fits within work */
    for (int t = 0; t < T_enc; t++)
        for (int ch = 0; ch < SUB_CH; ch++)
            for (int f = 0; f < F3; f++)
                flat[t * in_feat + ch * F3 + f] = buf2b[ch * T_enc * F3 + t * F3 + f];
    /* buf2b consumed */

    /* Stage 6: Linear(4096 → ENC_D=1280) → x_enc (permanent region) */
    linear(x_enc, flat, W.sow, W.sob, T_enc, in_feat, ENC_D);
}

/* ============================================================
 * Relative positional encoding → pe (2T−1, ENC_D)
 * Positions [T−1, T−2, ..., −(T−1)]
 * ============================================================ */
static void make_rel_pos_emb(int T, float *pe) {
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

/* ============================================================
 * RelPos MHSA:  x (T, ENC_D) + pos_emb (2T−1, ENC_D) → x (T, ENC_D)
 *
 * work layout (floats), peak = all of Q..attn_out simultaneous:
 *   [0         .. T×D−1]                Q         T × ENC_D
 *   [T×D       .. 2T×D−1]               K         T × ENC_D
 *   [2T×D      .. 3T×D−1]               V         T × ENC_D
 *   [3T×D      .. 3T×D+(2T−1)×D−1]      P         (2T−1) × ENC_D
 *   [3T×D+P    .. 3T×D+P+T×D−1]         shared    max(T²+T×(2T−1), T×D)
 *     → used as scores (T×T) + scores_bd (T×(2T−1)) during scoring
 *     → reused as proj_out (T×D) after scores freed
 *   [3T×D+P+T×D .. +T×D−1]              attn_out  T × ENC_D
 *
 *   Total ≈ 7×T×D − D  (proj_out and scores share one slot; no tw_mhsa scratch)
 *
 *   Note: T×D = 161280  >  T² + T×(2T−1) = 47502  for T_enc=126, so shared slot fits both.
 * ============================================================ */
static void rel_pos_mhsa(float *x, const float *pos_emb, int T, ELayer *L, float *work) {
    const int pos_len = 2 * T - 1;
    /* Assign sub-regions */
    float *Q       = work;
    float *K       = Q + (size_t)T * ENC_D;               /* T × ENC_D */
    float *V       = K + (size_t)T * ENC_D;               /* T × ENC_D */
    float *P       = V + (size_t)T * ENC_D;               /* pos_len × ENC_D */
    float *shared  = P + (size_t)pos_len * ENC_D;         /* T × ENC_D (scores fit here too) */
    float *attn_out= shared  + (size_t)T * ENC_D;         /* T × ENC_D */

    float *scores    = shared;                  /* T × T       ← during scoring */
    float *scores_bd = shared + (size_t)T * T;  /* T × pos_len ← during scoring */

    linear(Q, x,       L->sa_qw, L->sa_qb, T,       ENC_D, ENC_D);
    linear(K, x,       L->sa_kw, L->sa_kb, T,       ENC_D, ENC_D);
    linear(V, x,       L->sa_vw, L->sa_vb, T,       ENC_D, ENC_D);
    linear(P, pos_emb, L->sa_pw, NULL,     pos_len,  ENC_D, ENC_D);

    const float scale = 1.0f / sqrtf((float)ENC_DK);

    memset(attn_out, 0, (size_t)T * ENC_D * sizeof(float));

    for (int h = 0; h < ENC_H; h++) {
        /* matrix_ac: scores[q][k] = scale × Σ_d (Q[q][h×DK+d] + pbu[h×DK+d]) × K[k][h×DK+d] */
        for (int q = 0; q < T; q++)
        for (int k = 0; k < T; k++) {
            float s = 0;
            for (int d = 0; d < ENC_DK; d++)
                s += (Q[q*ENC_D + h*ENC_DK+d] + bf16(L->sa_pbu[h*ENC_DK+d]))
                   * K[k*ENC_D + h*ENC_DK+d];
            scores[q*T + k] = s * scale;
        }
        /* matrix_bd: scores_bd[q][p] = Σ_d (Q[q][h×DK+d] + pbv[h×DK+d]) × P[p][h×DK+d] */
        for (int q = 0; q < T; q++)
        for (int p = 0; p < pos_len; p++) {
            float s = 0;
            for (int d = 0; d < ENC_DK; d++)
                s += (Q[q*ENC_D + h*ENC_DK+d] + bf16(L->sa_pbv[h*ENC_DK+d]))
                   * P[p*ENC_D + h*ENC_DK+d];
            scores_bd[q*pos_len + p] = s;
        }
        /* rel_shift: scores[q][k] += scale × scores_bd[q][k + (T−1) − q] */
        for (int q = 0; q < T; q++)
        for (int k = 0; k < T; k++)
            scores[q*T + k] += scores_bd[q*pos_len + k + (T-1) - q] * scale;

        for (int q = 0; q < T; q++) softmax(scores + q*T, T);

        /* attn_out[q][h×DK+d] = Σ_k scores[q][k] × V[k][h×DK+d] */
        for (int q = 0; q < T; q++)
        for (int d = 0; d < ENC_DK; d++) {
            float s = 0;
            for (int k = 0; k < T; k++)
                s += scores[q*T + k] * V[k*ENC_D + h*ENC_DK+d];
            attn_out[q*ENC_D + h*ENC_DK+d] = s;
        }
    }
    /* scores, scores_bd consumed; attn_out consumed → write directly to x */
    linear(x, attn_out, L->sa_ow, L->sa_ob, T, ENC_D, ENC_D);
}

/* ============================================================
 * ConformerConv: x (T, ENC_D) → x (T, ENC_D)
 *
 * work layout (floats), peak during GLU:
 *   [0            .. T×2D−1]             pw1_out   T × 2×ENC_D   ← PEAK with glu_out
 *   [T×2D         .. T×3D−1]             glu_out   T × ENC_D
 *
 * After pw1_out consumed (reuse [0..]):
 *   [0            .. T×D−1]              dw_out    T × ENC_D
 *   (dw weights, bn params read inline from bf16; no scratch)
 *
 *   Total scratch = T×3D  (peak during GLU)
 * ============================================================ */
static void conformer_conv(float *x, int T, ELayer *L, float *work) {
    /* pointwise_conv1: (T, ENC_D) → (T, 2×ENC_D) */
    float *pw1_out = work;                              /* T × 2×ENC_D */
    linear(pw1_out, x, L->cv_pw1w, L->cv_pw1b, T, ENC_D, 2*ENC_D);

    /* GLU: glu_out[t][d] = pw1[t][d] × σ(pw1[t][d+ENC_D]) */
    float *glu_out = work + (size_t)T * 2 * ENC_D;    /* T × ENC_D; after pw1_out */
    for (int t = 0; t < T; t++)
    for (int d = 0; d < ENC_D; d++) {
        float a = pw1_out[t*2*ENC_D + d];
        float b = pw1_out[t*2*ENC_D + d + ENC_D];
        glu_out[t*ENC_D + d] = a / (1.0f + expf(-b));
    }
    /* pw1_out consumed */

    /* Depthwise conv1d: kernel ENC_CK=9, padding=(ENC_CK−1)/2=4; inline bf16 reads */
    const int pad_dw = (ENC_CK - 1) / 2;
    float *dw_out = work;  /* T × ENC_D; reuses pw1_out space */
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
    /* glu_out consumed */

    /* BatchNorm1d (inference): normalize dw_out per channel; inline bf16 reads */
    const float bn_eps = 1e-5f;
    for (int t = 0; t < T; t++)
    for (int c = 0; c < ENC_D; c++) {
        float v = dw_out[t*ENC_D + c];
        v = (v - bf16(L->cv_bnmean[c])) / sqrtf(bf16(L->cv_bnvar[c]) + bn_eps)
            * bf16(L->cv_bnw[c]) + bf16(L->cv_bnb[c]);
        dw_out[t*ENC_D + c] = v;
    }

    silu(dw_out, T * ENC_D);

    /* pointwise_conv2: (T, ENC_D) → (T, ENC_D) */
    linear(x, dw_out, L->cv_pw2w, L->cv_pw2b, T, ENC_D, ENC_D);
}

/* ============================================================
 * Conformer layer: x (T, ENC_D) + pos_emb → x (T, ENC_D)
 *
 * work layout:
 *   [0             .. T×ENC_FF−1]           tmp     T × 5120     (FFN intermediate)
 *   [T×ENC_FF      ]:                        nest    (shared across all sub-blocks)
 *
 *   Within nest, sub-blocks are sequential and share the same region:
 *     FFN blocks:     ln (T×D)                  reads/writes tmp for linear intermediates
 *     MHSA block:     ln (T×D) + rel_pos_mhsa work ((7T−1)×D)
 *     ConvMod block:  ln (T×D) + conformer_conv work (T×3D)   ← MHSA dominates
 *
 *   Peak: tmp + ln + (7T−1)×D  (MHSA path)
 *       = T×ENC_FF + T×ENC_D + (7T−1)×ENC_D
 *       = T×(ENC_FF + 8×ENC_D) − ENC_D
 *       ≈ T×15360 − 1280  floats
 * ============================================================ */
static void conformer_layer(float *x, const float *pos_emb, int T, ELayer *L, float *work) {
    float *tmp  = work;                          /* T × ENC_FF */
    float *nest = work + (size_t)T * ENC_FF;    /* remaining nested scratch */

    /* FFN1: x += 0.5 × ffn(layernorm(x)) */
    {
        float *ln = nest;  /* T × ENC_D */
        layernorm(ln, x, L->ff1_lnw, L->ff1_lnb, T, ENC_D);
        linear_silu(tmp, ln, L->ff1_w1, L->ff1_b1, T, ENC_D, ENC_FF);
        linear_fmadd(x, 0.5f, tmp, L->ff1_w2, L->ff1_b2, T, ENC_FF, ENC_D);
    }

    /* Self-attn: x += mhsa(layernorm(x)) */
    {
        float *ln = nest;  /* T × ENC_D; passed as in/out to rel_pos_mhsa */
        layernorm(ln, x, L->sa_lnw, L->sa_lnb, T, ENC_D);
        rel_pos_mhsa(ln, pos_emb, T, L, nest + (size_t)T * ENC_D);
        for (int i = 0; i < T * ENC_D; i++) x[i] += ln[i];
    }

    /* ConformerConv: x += conv(layernorm(x)) */
    {
        float *ln = nest;  /* T × ENC_D; passed as in/out to conformer_conv */
        layernorm(ln, x, L->cv_lnw, L->cv_lnb, T, ENC_D);
        conformer_conv(ln, T, L, nest + (size_t)T * ENC_D);
        for (int i = 0; i < T * ENC_D; i++) x[i] += ln[i];
    }

    /* FFN2: x += 0.5 × ffn(layernorm(x)) */
    {
        float *ln = nest;  /* T × ENC_D */
        layernorm(ln, x, L->ff2_lnw, L->ff2_lnb, T, ENC_D);
        linear_silu(tmp, ln, L->ff2_w1, L->ff2_b1, T, ENC_D, ENC_FF);
        linear_fmadd(x, 0.5f, tmp, L->ff2_w2, L->ff2_b2, T, ENC_FF, ENC_D);
    }

    layernorm(x, x, L->out_lnw, L->out_lnb, T, ENC_D);
}

/* ============================================================
 * Encoder forward: mel(N_MELS, T_mel) → enc_h(T_enc, DEC_D)
 *
 * x_enc: T_enc × ENC_D (permanent)
 * pos_emb: (2×T_enc−1) × ENC_D (permanent)
 * enc_h: T_enc × DEC_D (permanent)
 * work: scratch region (sized for conformer peak)
 *   Also used briefly for enc→dec projection tw (DEC_D × ENC_D = 1024×1280 = 1.31M)
 * ============================================================ */
static void encoder_forward(const float *mel, int T_mel,
                             float *x_enc, int T_enc,
                             float *pos_emb, float *enc_h, float *work) {
    encoder_subsampling(mel, T_mel, x_enc, T_enc, work);
    make_rel_pos_emb(T_enc, pos_emb);

    for (int l = 0; l < ENC_N; l++) {
        conformer_layer(x_enc, pos_emb, T_enc, &W.enc[l], work);
        if (verbose) fprintf(stderr, "  Finished layer %d/%d\n", l+1, ENC_N);
    }

    /* Projection: ENC_D → DEC_D */
    linear(enc_h, x_enc, W.prw, W.prb, T_enc, ENC_D, DEC_D);
}

/* ============================================================
 * Decoder scaled dot-product attention
 * q (T_q, DEC_D), k (T_k, DEC_D), v (T_k, DEC_D) → out (T_q, DEC_D)
 * mask: (T_q, T_k) or NULL
 * work: scores (T_q × T_k)
 * ============================================================ */
static void sdp_attn(float *out, const float *q, const float *k, const float *v,
                     const float *mask, int T_q, int T_k, float *work) {
    float *scores = work;  /* T_q × T_k */
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

/* ============================================================
 * Decoder layer: h (S, DEC_D) → h (S, DEC_D)
 *
 * work layout (nest = work[0..]):
 *   [0      .. S×D−1]               ln       S × DEC_D
 *   [S×D    .. 2S×D−1]              Q        S × DEC_D
 *   [2S×D   .. 3S×D−1]              K        S × DEC_D
 *   [3S×D   .. 4S×D−1]              V        S × DEC_D
 *   [4S×D   .. 5S×D−1]              attn_out S × DEC_D
 *   [5S×D   .. 5S×D+S×S−1]         scores   S × S  (inside sdp_attn)
 *   proj reuses Q slot ([S×D..]) after Q freed
 *   Peak (self-attn) = 5×S×D + S²
 *                    = 5×1048576 + 1048576 = 6,291,456  for S=MAX_SEQ=1024
 *
 *   Cross-attn (sequential, shares work from top):
 *     ln [0..], Q [S×D..], attn_out [2S×D..], scores [3S×D..], proj reuses Q
 *
 *   FFN (sequential, shares work from top):
 *     ln [0..S×D−1], mid [S×D..S×D+S×DEC_FF−1]  (S × DEC_FF = 4×S×D)
 *     Peak = S×D + S×DEC_FF = 5×S×D  <  self-attn peak
 * ============================================================ */
static void decoder_layer(float *h, const float *ca_k, const float *ca_v,
                          const float *mask, int S, int T_enc, DLayer *L, float *work) {
    float *ln      = work;                              /* S × DEC_D */
    float *Q       = work + (size_t)S * DEC_D;         /* S × DEC_D */
    float *K       = work + (size_t)2 * S * DEC_D;     /* S × DEC_D */
    float *V       = work + (size_t)3 * S * DEC_D;     /* S × DEC_D */
    float *attn_out= work + (size_t)4 * S * DEC_D;     /* S × DEC_D */
    float *sdp_work= work + (size_t)5 * S * DEC_D;     /* scores: S×S (self) or S×T_enc (cross) */
    float *proj    = Q;                                 /* reuse Q after Q freed */

    /* Self-attention with causal mask */
    layernorm(ln, h, L->sa_lnw, L->sa_lnb, S, DEC_D);
    linear(Q,       ln, L->sa_qw, L->sa_qb, S, DEC_D, DEC_D);
    linear(K,       ln, L->sa_kw, L->sa_kb, S, DEC_D, DEC_D);
    linear(V,       ln, L->sa_vw, L->sa_vb, S, DEC_D, DEC_D);
    sdp_attn(attn_out, Q, K, V, mask, S, S, sdp_work);
    /* Q, K, V consumed */
    linear(proj, attn_out, L->sa_ow, L->sa_ob, S, DEC_D, DEC_D);
    for (int i = 0; i < S * DEC_D; i++) h[i] += proj[i];
    /* ln, proj (=Q), attn_out, K, V consumed */

    /* Cross-attention (no mask); reuse work from top */
    layernorm(ln, h, L->ca_lnw, L->ca_lnb, S, DEC_D);
    linear(Q, ln, L->ca_qw, L->ca_qb, S, DEC_D, DEC_D);
    sdp_attn(attn_out, Q, ca_k, ca_v, NULL, S, T_enc, sdp_work);
    /* Q consumed; proj reuses Q slot */
    linear(proj, attn_out, L->ca_ow, L->ca_ob, S, DEC_D, DEC_D);
    for (int i = 0; i < S * DEC_D; i++) h[i] += proj[i];
    /* ln, Q, attn_out, proj consumed */

    /* FFN; reuse work from top */
    float *mid = Q;  /* S × DEC_FF = 4×S×DEC_D; starts at Q's offset, Q consumed */
    layernorm(ln, h, L->ffn_lnw, L->ffn_lnb, S, DEC_D);
    linear_relu(mid, ln, L->ffn_w1, L->ffn_b1, S, DEC_D, DEC_FF);
    linear_fmadd(h, 1.0f, mid, L->ffn_w2, L->ffn_b2, S, DEC_FF, DEC_D);
}

/* ============================================================
 * Pre-compute cross-attention K/V from encoder hidden states.
 * ============================================================ */
static void precompute_ca_kv(const float *enc_h, int T_enc,
                              float *ca_k[DEC_N], float *ca_v[DEC_N]) {
    for (int l = 0; l < DEC_N; l++) {
        linear(ca_k[l], enc_h, W.dec[l].ca_kw, W.dec[l].ca_kb, T_enc, DEC_D, DEC_D);
        linear(ca_v[l], enc_h, W.dec[l].ca_vw, W.dec[l].ca_vb, T_enc, DEC_D, DEC_D);
    }
}

/* ============================================================
 * Decoder step: tokens[S] → next token id (greedy argmax)
 *
 * h:    MAX_SEQ × DEC_D (permanent, reused each step)
 * mask: MAX_SEQ × MAX_SEQ (permanent, rebuilt each step)
 * work: scratch (decoder_layer peak = 5×S×D + S²)
 * ============================================================ */
static int decoder_step(const int *tokens, int S, int T_enc,
                        float *ca_k[DEC_N], float *ca_v[DEC_N],
                        float *h, float *mask, float *work) {
    /* Embedding: h[t] = emb_tok[tokens[t]] + emb_pos[t], then LayerNorm */
    for (int t = 0; t < S; t++) {
        int id = tokens[t];
        for (int d = 0; d < DEC_D; d++)
            h[t*DEC_D + d] = bf16(W.etw[id*DEC_D + d]) + bf16(W.epw[t*DEC_D + d]);
    }
    layernorm(h, h, W.elnw, W.elnb, S, DEC_D);

    /* Causal mask: mask[qi][ki] = −1e9 if ki > qi */
    for (int qi = 0; qi < S; qi++)
    for (int ki = 0; ki < S; ki++)
        mask[qi*S + ki] = (ki > qi) ? -1e9f : 0.0f;

    for (int l = 0; l < DEC_N; l++)
        decoder_layer(h, ca_k[l], ca_v[l], mask, S, T_enc, &W.dec[l], work);

    layernorm(h, h, W.dlnw, W.dlnb, S, DEC_D);

    /* Head: argmax over vocab (weight-tied to token embedding) */
    const float *last = h + (S - 1) * DEC_D;
    float best = -1e38f; int best_id = 0;
    for (int v = 0; v < VOCAB; v++) {
        float logit = bf16(W.hdb[v]);
        for (int d = 0; d < DEC_D; d++)
            logit += last[d] * bf16(W.etw[v*DEC_D + d]);
        if (logit > best) { best = logit; best_id = v; }
    }
    return best_id;
}

#endif /* MODEL_H */
