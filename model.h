/*
 * model.h — weight loading and inference wrappers.
 *
 * Compute is dispatched to the selected backend via backends/backend.h.
 * Build with -DBACKEND_CUDA or -DBACKEND_IRON; default is CPU.
 */
#ifndef MODEL_H
#define MODEL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "backends/model_types.h"

/* ============================================================
 * Token IDs
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

/* ============================================================
 * Safetensors loader
 * ============================================================ */
#ifndef EMBEDDED_MODEL
static char    *sf_hdr;
static uint8_t *sf_data;
static size_t   sf_data_bytes;

static void sf_load(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); exit(1); }
    struct stat st; fstat(fd, &st);
    uint8_t *m = (uint8_t*)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) { perror("mmap"); exit(1); }
    uint64_t hs; memcpy(&hs, m, 8);
    sf_hdr  = (char*)malloc(hs + 1);
    memcpy(sf_hdr, m + 8, hs);
    sf_hdr[hs] = '\0';
    sf_data       = m + 8 + hs;
    sf_data_bytes = (size_t)st.st_size - 8 - hs;
}

static const uint16_t *sf_get(const char *name, size_t *nelems) {
    char key[512];
    snprintf(key, sizeof(key), "\"%s\":", name);
    char *pos = sf_hdr;
    for (;;) {
        pos = strstr(pos, key);
        if (!pos) { fprintf(stderr, "tensor not found: %s\n", name); exit(1); }
        if (pos == sf_hdr || pos[-1] == '{' || pos[-1] == ',') break;
        pos++;
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
 * Global weights and load_weights
 * ============================================================ */
static Weights W;

#ifdef EMBEDDED_MODEL
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

#else /* !EMBEDDED_MODEL */
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
 * Backend interface (declared after ELayer/DLayer/Weights above)
 * ============================================================ */
#include "backends/backend.h"

/* ============================================================
 * Patch all weight pointers in W from host to device.
 * Called after backend_weights_upload_blob() on CUDA/IRON.
 * ============================================================ */
#if defined(BACKEND_CUDA)
static void patch_weights_to_device(void) {
#define PW(f)  (W.f)  = (const uint16_t *)backend_device_ptr((const void *)(W.f))
#define PL(p,f) (p)->f = (const uint16_t *)backend_device_ptr((const void *)(p)->f)
    PW(s0w); PW(s0b); PW(s2w); PW(s2b); PW(s3w); PW(s3b); PW(s5w); PW(s5b); PW(s6w); PW(s6b);
    PW(sow); PW(sob);
    for (int i = 0; i < ENC_N; i++) {
        ELayer *L = &W.enc[i];
        PL(L,ff1_lnw); PL(L,ff1_lnb); PL(L,ff1_w1); PL(L,ff1_b1); PL(L,ff1_w2); PL(L,ff1_b2);
        PL(L,sa_lnw);  PL(L,sa_lnb);  PL(L,sa_qw);  PL(L,sa_qb);  PL(L,sa_kw);  PL(L,sa_kb);
        PL(L,sa_vw);   PL(L,sa_vb);   PL(L,sa_pw);
        PL(L,sa_ow);   PL(L,sa_ob);   PL(L,sa_pbu);  PL(L,sa_pbv);
        PL(L,cv_lnw);  PL(L,cv_lnb);  PL(L,cv_pw1w); PL(L,cv_pw1b);
        PL(L,cv_dww);  PL(L,cv_dwb);
        PL(L,cv_bnw);  PL(L,cv_bnb);  PL(L,cv_bnmean); PL(L,cv_bnvar);
        PL(L,cv_pw2w); PL(L,cv_pw2b);
        PL(L,ff2_lnw); PL(L,ff2_lnb); PL(L,ff2_w1); PL(L,ff2_b1); PL(L,ff2_w2); PL(L,ff2_b2);
        PL(L,out_lnw); PL(L,out_lnb);
    }
    PW(prw); PW(prb); PW(etw); PW(epw); PW(elnw); PW(elnb);
    for (int i = 0; i < DEC_N; i++) {
        DLayer *L = &W.dec[i];
        PL(L,sa_lnw);  PL(L,sa_lnb);  PL(L,sa_qw);  PL(L,sa_qb);
        PL(L,sa_kw);   PL(L,sa_kb);   PL(L,sa_vw);  PL(L,sa_vb);
        PL(L,sa_ow);   PL(L,sa_ob);
        PL(L,ca_lnw);  PL(L,ca_lnb);  PL(L,ca_qw);  PL(L,ca_qb);
        PL(L,ca_kw);   PL(L,ca_kb);   PL(L,ca_vw);  PL(L,ca_vb);
        PL(L,ca_ow);   PL(L,ca_ob);
        PL(L,ffn_lnw); PL(L,ffn_lnb); PL(L,ffn_w1); PL(L,ffn_b1); PL(L,ffn_w2); PL(L,ffn_b2);
    }
    PW(dlnw); PW(dlnb); PW(hdb);
#undef PW
#undef PL
}
#endif /* BACKEND_CUDA */

/* ============================================================
 * Conformer layer  x (T, ENC_D) → x (T, ENC_D)
 * ============================================================ */
static void conformer_layer(float *x, const float *pos_emb, int T,
                             const ELayer *L, float *work) {
    float *tmp  = work;
    float *nest = work + (size_t)T * ENC_FF;

    /* FFN1: x += 0.5 * ffn(layernorm(x)) */
    {
        float *ln = nest;
        backend_layernorm(ln, x, L->ff1_lnw, L->ff1_lnb, T, ENC_D);
        backend_linear_silu(tmp, ln, L->ff1_w1, L->ff1_b1, T, ENC_D, ENC_FF);
        backend_linear_fmadd(x, 0.5f, tmp, L->ff1_w2, L->ff1_b2, T, ENC_FF, ENC_D);
    }

    /* Self-attn: x += mhsa(layernorm(x)) */
    {
        float *ln = nest;
        backend_layernorm(ln, x, L->sa_lnw, L->sa_lnb, T, ENC_D);
        backend_rel_pos_mhsa(ln, pos_emb, T, L, nest + (size_t)T * ENC_D);
        backend_add_inplace(x, ln, T * ENC_D);
    }

    /* ConformerConv: x += conv(layernorm(x)) */
    {
        float *ln = nest;
        backend_layernorm(ln, x, L->cv_lnw, L->cv_lnb, T, ENC_D);
        backend_conformer_conv(ln, T, L, nest + (size_t)T * ENC_D);
        backend_add_inplace(x, ln, T * ENC_D);
    }

    /* FFN2: x += 0.5 * ffn(layernorm(x)) */
    {
        float *ln = nest;
        backend_layernorm(ln, x, L->ff2_lnw, L->ff2_lnb, T, ENC_D);
        backend_linear_silu(tmp, ln, L->ff2_w1, L->ff2_b1, T, ENC_D, ENC_FF);
        backend_linear_fmadd(x, 0.5f, tmp, L->ff2_w2, L->ff2_b2, T, ENC_FF, ENC_D);
    }

    backend_layernorm(x, x, L->out_lnw, L->out_lnb, T, ENC_D);
}

/* ============================================================
 * Encoder forward
 * ============================================================ */
static void encoder_forward(const float *mel, int T_mel,
                             float *x_enc, int T_enc,
                             float *pos_emb, float *enc_h, float *work) {
    extern int verbose;
    /* Key the encoder graph by T_mel, not T_enc.  Multiple T_mel values can
     * produce the same T_enc through the ceiling-division subsampling chain,
     * so keying by T_enc would replay a graph with stale T_mel-dependent grid
     * sizes (transpose_mel, conv2d launches) on mismatched mel data. */
    if (!backend_graph_replay(GRAPH_TAG_ENCODER, T_mel, 0, 1)) {
        backend_graph_begin_capture(GRAPH_TAG_ENCODER);

        backend_encoder_subsampling(mel, T_mel, T_mel, x_enc, T_enc, work, &W);
        backend_make_rel_pos_emb(pos_emb, T_enc);

        for (int l = 0; l < ENC_N; l++) {
            conformer_layer(x_enc, pos_emb, T_enc, &W.enc[l], work);
#if !defined(BACKEND_CUDA) && !defined(BACKEND_IRON)
            if (verbose) fprintf(stderr, "  Finished layer %d/%d\n", l+1, ENC_N);
#endif
        }

        backend_linear(enc_h, x_enc, W.prw, W.prb, T_enc, ENC_D, DEC_D);

        backend_graph_end_capture(GRAPH_TAG_ENCODER, T_mel, 0, 1);
    }
}

/* ============================================================
 * Pre-compute cross-attention K/V
 * ============================================================ */
static void precompute_ca_kv(const float *enc_h, int T_mel, int T_enc,
                              float *ca_k[DEC_N], float *ca_v[DEC_N]) {
    /* Key by T_mel: ca_k/ca_v live at T_mel-dependent arena offsets, so two
     * chunks sharing T_enc but differing in T_mel would otherwise replay a
     * graph whose captured ca_k/ca_v point into the wrong arena region. */
    if (!backend_graph_replay(GRAPH_TAG_PRECOMPUTE, T_mel, 0, 1)) {
        backend_graph_begin_capture(GRAPH_TAG_PRECOMPUTE);

        for (int l = 0; l < DEC_N; l++) {
            backend_linear(ca_k[l], enc_h, W.dec[l].ca_kw, W.dec[l].ca_kb, T_enc, DEC_D, DEC_D);
            backend_linear(ca_v[l], enc_h, W.dec[l].ca_vw, W.dec[l].ca_vb, T_enc, DEC_D, DEC_D);
        }

        backend_graph_end_capture(GRAPH_TAG_PRECOMPUTE, T_mel, 0, 1);
    }
}

/* ============================================================
 * Decoder layer with KV cache
 *
 * h         (S_q, DEC_D) — query token embeddings (1 during decode, n_prompt during prefill)
 * sa_k_cache (S_kv, DEC_D) — accumulated self-attn K cache for this layer
 * sa_v_cache (S_kv, DEC_D) — accumulated self-attn V cache for this layer
 * S_q       — number of new query tokens being processed
 * S_kv      — total KV sequence length after this step (S_q tokens appended to prior S_kv-S_q)
 * mask      — (S_q, S_kv) additive bias, or NULL (pass NULL during decode, S_q==1)
 * ============================================================ */
static void decoder_layer(float *h, const float *ca_k, const float *ca_v,
                          const float *mask, int S_q, int S_kv, int T_enc,
                          float *sa_k_cache, float *sa_v_cache,
                          const DLayer *L, float *work) {
    float *ln       = work;
    float *Q        = work + (size_t)S_q * DEC_D;
    float *attn_out = work + (size_t)2 * S_q * DEC_D;
    float *sdp_work = work + (size_t)3 * S_q * DEC_D;
    /* K/V for the S_q new tokens written directly into the cache at
     * positions [S_kv - S_q .. S_kv - 1], so all S_kv entries are valid
     * for the sdp_attn call below. */
    float *K_new    = sa_k_cache + (size_t)(S_kv - S_q) * DEC_D;
    float *V_new    = sa_v_cache + (size_t)(S_kv - S_q) * DEC_D;

    /* Self-attention: Q over S_q tokens, K/V over full cache (S_kv tokens) */
    backend_layernorm(ln, h, L->sa_lnw, L->sa_lnb, S_q, DEC_D);
    backend_linear(Q,     ln, L->sa_qw, L->sa_qb, S_q, DEC_D, DEC_D);
    backend_linear(K_new, ln, L->sa_kw, L->sa_kb, S_q, DEC_D, DEC_D);
    backend_linear(V_new, ln, L->sa_vw, L->sa_vb, S_q, DEC_D, DEC_D);
    backend_sdp_attn(attn_out, Q, sa_k_cache, sa_v_cache, mask, S_q, S_kv, sdp_work);
    backend_linear_fmadd(h, 1.0f, attn_out, L->sa_ow, L->sa_ob, S_q, DEC_D, DEC_D);

    /* Cross-attention */
    backend_layernorm(ln, h, L->ca_lnw, L->ca_lnb, S_q, DEC_D);
    backend_linear(Q, ln, L->ca_qw, L->ca_qb, S_q, DEC_D, DEC_D);
    backend_sdp_attn(attn_out, Q, ca_k, ca_v, NULL, S_q, T_enc, sdp_work);
    backend_linear_fmadd(h, 1.0f, attn_out, L->ca_ow, L->ca_ob, S_q, DEC_D, DEC_D);

    /* FFN */
    float *mid = Q;  /* reuse Q after consumed */
    backend_layernorm(ln, h, L->ffn_lnw, L->ffn_lnb, S_q, DEC_D);
    backend_linear_relu(mid, ln, L->ffn_w1, L->ffn_b1, S_q, DEC_D, DEC_FF);
    backend_linear_fmadd(h, 1.0f, mid, L->ffn_w2, L->ffn_b2, S_q, DEC_FF, DEC_D);
}

/* ============================================================
 * Decoder layer (device-side S variant): processes a single query token.
 *
 * Uses backend_linear_to_kvcache to write K/V at slot (*g_dev_S - 1), and
 * backend_sdp_attn_devS to attend over the full cache (T_k = *g_dev_S).
 * backend_decode_inc_S() must have already run before any call to this layer.
 * Cross-attention T_enc is a compile-time-visible scalar — baked into the
 * captured graph but correct because the graph is keyed by T_mel→T_enc.
 * ============================================================ */
static void decoder_layer_decode(float *h, const float *ca_k, const float *ca_v,
                                  int T_enc,
                                  float *sa_k_cache, float *sa_v_cache,
                                  const DLayer *L, float *work) {
    float *ln       = work;
    float *Q        = work + (size_t)DEC_D;
    float *attn_out = work + (size_t)2 * DEC_D;
    float *sdp_work = work + (size_t)3 * DEC_D;  /* needs MAX_SEQ floats for scores */

    /* Self-attention: Q from this token, K/V written to cache at *g_dev_S-1 slot */
    backend_layernorm(ln, h, L->sa_lnw, L->sa_lnb, 1, DEC_D);
    backend_linear(Q, ln, L->sa_qw, L->sa_qb, 1, DEC_D, DEC_D);
    backend_linear_to_kvcache(sa_k_cache, ln, L->sa_kw, L->sa_kb, DEC_D, DEC_D);
    backend_linear_to_kvcache(sa_v_cache, ln, L->sa_vw, L->sa_vb, DEC_D, DEC_D);
    backend_sdp_attn_devS(attn_out, Q, sa_k_cache, sa_v_cache, sdp_work);
    backend_linear_fmadd(h, 1.0f, attn_out, L->sa_ow, L->sa_ob, 1, DEC_D, DEC_D);

    /* Cross-attention: T_enc is fixed per chunk, baked into graph at capture time */
    backend_layernorm(ln, h, L->ca_lnw, L->ca_lnb, 1, DEC_D);
    backend_linear(Q, ln, L->ca_qw, L->ca_qb, 1, DEC_D, DEC_D);
    backend_sdp_attn(attn_out, Q, ca_k, ca_v, NULL, 1, T_enc, sdp_work);
    backend_linear_fmadd(h, 1.0f, attn_out, L->ca_ow, L->ca_ob, 1, DEC_D, DEC_D);

    /* FFN */
    float *mid = Q;
    backend_layernorm(ln, h, L->ffn_lnw, L->ffn_lnb, 1, DEC_D);
    backend_linear_relu(mid, ln, L->ffn_w1, L->ffn_b1, 1, DEC_D, DEC_FF);
    backend_linear_fmadd(h, 1.0f, mid, L->ffn_w2, L->ffn_b2, 1, DEC_FF, DEC_D);
}

/* ============================================================
 * Decoder step: tokens[S] → next token id (greedy argmax)
 *
 * S_q: number of new tokens to process this step.
 *   S_q == S  (prefill): embed all S prompt tokens, populate KV cache from scratch.
 *   S_q == 1  (decode):  embed only the last token (tokens[S-1]) at absolute
 *                        position S-1, append one new K/V entry to each cache.
 *
 * During decode (S_q==1) the self-attention causal mask is skipped: a single
 * query token attends to all prior cached positions unconditionally.
 * ============================================================ */
static int decoder_step(const int *tokens, int S, int S_q, int T_mel, int T_enc,
                        float *ca_k[DEC_N], float *ca_v[DEC_N],
                        float *sa_k_cache[DEC_N], float *sa_v_cache[DEC_N],
                        float *h, float *mask, float *work) {
    float *dev_logits = work;  /* reuse work; VOCAB << work_size */

    /* Upload only the S_q new tokens; they land at g_dev_tokens[0..S_q-1]. */
    backend_upload_tokens(tokens + (S - S_q), S_q);

    /* Key the decoder graph by T_mel (not T_enc).  ca_k/ca_v and h live at
     * T_mel-dependent arena offsets, so two chunks sharing T_enc but differing
     * in T_mel would otherwise replay a graph whose captured pointers point
     * into the wrong arena region. */
    if (!backend_graph_replay(GRAPH_TAG_DECODER, T_mel, S, 1)) {
        backend_graph_begin_capture(GRAPH_TAG_DECODER);

        /* Embed S_q tokens at absolute positions [S-S_q .. S-1]. */
        backend_embed(h, tokens + (S - S_q), S_q, S - S_q,
                      W.etw, W.epw, W.elnw, W.elnb);
        /* Causal mask only needed for prefill where multiple queries are present. */
        if (S_q > 1)
            backend_fill_causal_mask(mask, S_q);

        for (int l = 0; l < DEC_N; l++)
            decoder_layer(h, ca_k[l], ca_v[l], S_q > 1 ? mask : NULL,
                          S_q, S, T_enc, sa_k_cache[l], sa_v_cache[l],
                          &W.dec[l], work);

        backend_layernorm(h, h, W.dlnw, W.dlnb, S_q, DEC_D);
        backend_lm_head(dev_logits, h + (size_t)(S_q - 1) * DEC_D, W.etw, W.hdb);

        backend_graph_end_capture(GRAPH_TAG_DECODER, T_mel, S, 1);
    }

    /* DtoH and argmax are always outside the captured region. */
    float logits_cpu[VOCAB];
    backend_dtoh(logits_cpu, dev_logits, VOCAB * sizeof(float));

    int best_id = 0; float best = logits_cpu[0];
    for (int v = 1; v < VOCAB; v++)
        if (logits_cpu[v] > best) { best = logits_cpu[v]; best_id = v; }
    return best_id;
}

/* ============================================================
 * Decoder step using device-side S counter (decode only, not prefill).
 *
 * A single CUDA graph keyed (GRAPH_TAG_DECODER, T_mel, -1) serves every
 * decode step: backend_decode_inc_S() fires first so all subsequent kernels
 * in the body see the post-increment *g_dev_S.  On CPU/IRON this reduces to
 * plain scalar increments with no overhead.
 *
 * next_token: the token produced by the previous step (or prefill).
 * Caller must have called backend_decode_set_S(n_prompt) before the first
 * call to this function for the current chunk.
 * ============================================================ */
static int decoder_step_decode(int next_token, int T_mel, int T_enc,
                               float *ca_k[DEC_N], float *ca_v[DEC_N],
                               float *sa_k_cache[DEC_N], float *sa_v_cache[DEC_N],
                               float *h, float *work) {
    float *dev_logits = work;  /* VOCAB floats; reused from scratch before lm_head */

    /* Stage the single new token to device (outside capture). */
    int tok[1] = { next_token };
    backend_upload_tokens(tok, 1);

    /* Sentinel S=-1 never collides with prefill key (S=n_prompt≥1) or old
     * per-step decode keys (S≥2). */
    if (!backend_graph_replay(GRAPH_TAG_DECODER, T_mel, -1, 1)) {
        backend_graph_begin_capture(GRAPH_TAG_DECODER);

        /* Increment must be the first captured kernel so all subsequent
         * kernels see *g_dev_S == new sequence length. */
        backend_decode_inc_S();
        backend_embed_decode(h, W.etw, W.epw, W.elnw, W.elnb);

        for (int l = 0; l < DEC_N; l++)
            decoder_layer_decode(h, ca_k[l], ca_v[l], T_enc,
                                 sa_k_cache[l], sa_v_cache[l], &W.dec[l], work);

        backend_layernorm(h, h, W.dlnw, W.dlnb, 1, DEC_D);
        backend_lm_head(dev_logits, h, W.etw, W.hdb);

        backend_graph_end_capture(GRAPH_TAG_DECODER, T_mel, -1, 1);
    }

    float logits_cpu[VOCAB];
    backend_dtoh(logits_cpu, dev_logits, VOCAB * sizeof(float));

    int best_id = 0; float best = logits_cpu[0];
    for (int v = 1; v < VOCAB; v++)
        if (logits_cpu[v] > best) { best = logits_cpu[v]; best_id = v; }
    return best_id;
}

/* ============================================================
 * Batch encoder forward: runs the encoder on B zero-padded mel spectrograms.
 *
 * mel_batch:   [B, N_MELS, T_mel_max]  — all items zero-padded to T_mel_max
 * T_mel_max:   padded mel length (buffer stride for subsampling reads)
 * T_enc[b]:    actual valid encoder length for item b
 * T_enc_max:   max of T_enc[], used for pos_emb allocation
 * x_enc_batch: [B, T_enc_max, ENC_D]  — encoder outputs (out)
 * pos_emb:     [2*T_enc_max-1, ENC_D] — sinusoidal embeddings (written once)
 * enc_h_batch: [B, T_enc_max, DEC_D]  — post-projection encoder hidden (out)
 *
 * Each item b is processed with its actual T_enc[b] positions, using an offset
 * into the shared pos_emb so that relative position embeddings are bit-identical
 * to the single-item path.  The subsampling still reads with stride T_mel_max
 * (correct for the zero-padded batch buffer), but the conformer and projection
 * only touch the first T_enc[b] rows — giving results identical to B=1.
 * ============================================================ */
static void encoder_forward_batch(const float *mel_batch, int B,
                                   int T_mel_max, const int *T_mel,
                                   const int *T_enc, int T_enc_max,
                                   float *x_enc_batch, float *pos_emb,
                                   float *enc_h_batch, float *work) {
    extern int verbose;
    /* pos_emb built for T_enc_max: pos_emb[i] = embedding for relative position
     * i - (T_enc_max-1).  For item b with T_enc[b] <= T_enc_max, the conformer
     * with sequence length T_enc[b] needs embeddings for relative positions
     * -(T_enc[b]-1)..+(T_enc[b]-1), which live at indices
     * (T_enc_max-T_enc[b])..(T_enc_max+T_enc[b]-2) in pos_emb.
     * Passing pos_emb + (T_enc_max-T_enc[b])*ENC_D makes index 0 of the shifted
     * pointer correspond to relative position -(T_enc[b]-1), matching B=1. */
    backend_make_rel_pos_emb(pos_emb, T_enc_max);
    for (int b = 0; b < B; b++) {
        const float *mel_b     = mel_batch   + (size_t)b * N_MELS    * T_mel_max;
        float       *x_enc_b   = x_enc_batch + (size_t)b * T_enc_max * ENC_D;
        float       *enc_h_b   = enc_h_batch + (size_t)b * T_enc_max * DEC_D;
        int          Te        = T_enc[b];
        const float *pos_emb_b = pos_emb + (size_t)(T_enc_max - Te) * ENC_D;

        /* Subsampling uses T_mel[b] as the actual length and T_mel_max as the
         * row stride (the mel is zero-padded to T_mel_max columns in mel_batch).
         * Passing T_mel[b] ensures conv2d sees the correct input size, giving
         * bit-identical encoder output to the single-item path. */
        backend_encoder_subsampling(mel_b, T_mel[b], T_mel_max, x_enc_b, Te, work, &W);
        for (int l = 0; l < ENC_N; l++) {
            /* Process only the Te valid rows; pos_emb_b is shifted so that relative
             * position indices match the B=1 path exactly. */
            conformer_layer(x_enc_b, pos_emb_b, Te, &W.enc[l], work);
#if !defined(BACKEND_CUDA) && !defined(BACKEND_IRON)
            if (verbose) fprintf(stderr, "  batch item %d layer %d/%d\n", b, l+1, ENC_N);
#endif
        }
        backend_linear(enc_h_b, x_enc_b, W.prw, W.prb, Te, ENC_D, DEC_D);
    }
}

/* ============================================================
 * Batch cross-attention K/V precompute.
 *
 * enc_h_batch: [B, T_enc_max, DEC_D]
 * T_enc[b]:    actual valid encoder length for item b (rows beyond T_enc[b]
 *              in enc_h_batch[b] are near-zero from padding and ignored)
 * ca_k_batch[l]: [B, T_enc_max, DEC_D] — only first T_enc[b] rows written
 * ca_v_batch[l]: [B, T_enc_max, DEC_D] — only first T_enc[b] rows written
 * ============================================================ */
static void precompute_ca_kv_batch(const float *enc_h_batch, int B,
                                    const int *T_enc, int T_enc_max,
                                    float *ca_k_batch[DEC_N],
                                    float *ca_v_batch[DEC_N]) {
    for (int b = 0; b < B; b++) {
        const float *enc_h_b = enc_h_batch + b * (size_t)T_enc_max * DEC_D;
        for (int l = 0; l < DEC_N; l++) {
            float *ca_k_b = ca_k_batch[l] + b * (size_t)T_enc_max * DEC_D;
            float *ca_v_b = ca_v_batch[l] + b * (size_t)T_enc_max * DEC_D;
            backend_linear(ca_k_b, enc_h_b, W.dec[l].ca_kw, W.dec[l].ca_kb,
                           T_enc[b], DEC_D, DEC_D);
            backend_linear(ca_v_b, enc_h_b, W.dec[l].ca_vw, W.dec[l].ca_vb,
                           T_enc[b], DEC_D, DEC_D);
        }
    }
}

/* ============================================================
 * Batch decoder layer — decode step (T_q=1 per active item).
 *
 * h_dec:        [B, DEC_D]          — hidden state, updated in-place
 * ca_k_batch_l: [B, T_enc_max, DEC_D] — cross-attn keys for this layer
 * ca_v_batch_l: [B, T_enc_max, DEC_D] — cross-attn values for this layer
 * T_enc[b]:     actual cross-attn length per item
 * sa_k/v_batch: [B, MAX_SEQ, DEC_D] — self-attn KV cache for this layer
 * S_dev[b]:     current sequence length (already incremented; KV written to
 *               slot S_dev[b]-1, attention over S_dev[b] entries)
 * active[b]:    1 = process, 0 = skip
 * work:         scratch, sized for a single item's decode step
 * ============================================================ */
/* decoder_layer_decode_batch: one decode step for B sequences in parallel.
 *
 * All B items are processed together using batched backend calls.  Inactive
 * items produce garbage outputs that are discarded by the caller (lm_head_batch
 * and argmax_batch both check active[] before writing results).
 *
 * Work layout (from work pointer):
 *   [B, DEC_D]   ln       — layernorm scratch
 *   [B, DEC_D]   Q        — query projections
 *   [B, DEC_D]   attn_out — attention output accumulator
 *   [B * DEC_H * max(MAX_SEQ, T_enc_max)]  sdp_work — attention scores
 *
 * Device-side S counter array (g_dev_S_batch) must have been set by the
 * preceding backend_embed_decode_batch call this step.
 * Device-side T_enc array (g_dev_T_enc) must have been set once before the
 * decode loop via backend_upload_T_enc. */
static void decoder_layer_decode_batch(
        float *h_dec,
        const float *ca_k_batch_l, const float *ca_v_batch_l,
        int T_enc_max,
        float *sa_k_batch_l, float *sa_v_batch_l,
        int B, const DLayer *L, float *work) {
    float *ln       = work;
    float *Q        = work + (size_t)B * DEC_D;
    float *attn_out = work + (size_t)2 * B * DEC_D;
    float *sdp_work = work + (size_t)3 * B * DEC_D;

    /* Self-attention: process all B items in one set of batched calls. */
    backend_layernorm(ln, h_dec, L->sa_lnw, L->sa_lnb, B, DEC_D);
    backend_linear(Q, ln, L->sa_qw, L->sa_qb, B, DEC_D, DEC_D);
    backend_linear_to_kvcache_batch(sa_k_batch_l, ln, L->sa_kw, L->sa_kb,
                                     B, MAX_SEQ, DEC_D, DEC_D);
    backend_linear_to_kvcache_batch(sa_v_batch_l, ln, L->sa_vw, L->sa_vb,
                                     B, MAX_SEQ, DEC_D, DEC_D);
    backend_sdp_attn_batch_decode_sa(attn_out, Q,
                                      sa_k_batch_l, sa_v_batch_l,
                                      MAX_SEQ, B, sdp_work);
    backend_linear_fmadd(h_dec, 1.0f, attn_out, L->sa_ow, L->sa_ob, B, DEC_D, DEC_D);

    /* Cross-attention: T_k[b] = T_enc[b] from device state. */
    backend_layernorm(ln, h_dec, L->ca_lnw, L->ca_lnb, B, DEC_D);
    backend_linear(Q, ln, L->ca_qw, L->ca_qb, B, DEC_D, DEC_D);
    backend_sdp_attn_batch_decode_ca(attn_out, Q,
                                      ca_k_batch_l, ca_v_batch_l,
                                      T_enc_max, B, sdp_work);
    backend_linear_fmadd(h_dec, 1.0f, attn_out, L->ca_ow, L->ca_ob, B, DEC_D, DEC_D);

    /* FFN */
    float *mid = Q;
    backend_layernorm(ln, h_dec, L->ffn_lnw, L->ffn_lnb, B, DEC_D);
    backend_linear_relu(mid, ln, L->ffn_w1, L->ffn_b1, B, DEC_D, DEC_FF);
    backend_linear_fmadd(h_dec, 1.0f, mid, L->ffn_w2, L->ffn_b2, B, DEC_FF, DEC_D);
}

/* ============================================================
 * Batch decoder decode step: one autoregressive step for B sequences.
 *
 * tokens_cur[b]: token to embed in this step (= tokens[b][S[b]-1], before
 *                S was incremented). Active items only; inactive ignored.
 * S_dev[b]:      sequence counter AFTER increment (S_dev[b] was incremented
 *                by the caller before this call, same semantics as g_dev_S).
 * active[b]:     1 = active, 0 = already finished.
 * T_enc[b]:      actual encoder output length per item.
 * T_enc_max:     stride of ca_k/ca_v/sa_k/sa_v batch buffers.
 * ca_k/v_batch[l]: [B, T_enc_max, DEC_D] per decoder layer l.
 * sa_k/v_batch[l]: [B, MAX_SEQ, DEC_D]  per decoder layer l.
 * h_dec:         [B, DEC_D] scratch for current hidden states (out).
 * logits_batch:  [B, VOCAB] scratch for logits.
 * work:          scratch sized for one item's decode step.
 * next_tokens[b]: output — argmax token for each active item.
 * ============================================================ */
static void decoder_step_decode_batch(
        const int *tokens_cur, const int *S_dev, const int *active,
        int T_enc_max,
        float *ca_k_batch[DEC_N], float *ca_v_batch[DEC_N],
        float *sa_k_batch[DEC_N], float *sa_v_batch[DEC_N],
        float *h_dec, float *logits_batch, float *work, int B,
        int *next_tokens) {
    /* Embed current tokens (also uploads S_dev to device for kvcache/sdp use). */
    backend_embed_decode_batch(h_dec, tokens_cur, S_dev, active, B,
                                W.etw, W.epw, W.elnw, W.elnb);

    /* Decoder layers: all B items processed in parallel per layer. */
    for (int l = 0; l < DEC_N; l++)
        decoder_layer_decode_batch(h_dec,
                                    ca_k_batch[l], ca_v_batch[l],
                                    T_enc_max,
                                    sa_k_batch[l], sa_v_batch[l],
                                    B, &W.dec[l], work);

    /* Final layernorm for all B (inactive items produce garbage, discarded below). */
    backend_layernorm(h_dec, h_dec, W.dlnw, W.dlnb, B, DEC_D);
    backend_lm_head_batch(logits_batch, h_dec, W.etw, W.hdb, active, B);
    backend_argmax_batch(next_tokens, logits_batch, active, B);
}

#endif /* MODEL_H */
