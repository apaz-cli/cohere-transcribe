/* backends/model_types.h — model constants, bf16, weight struct types.
 * Included by both model.h and backend implementation files. */
#ifndef BACKENDS_MODEL_TYPES_H
#define BACKENDS_MODEL_TYPES_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

/* ============================================================ */
/* Model constants                                               */
/* ============================================================ */
#define ENC_D    1280
#define ENC_FF   5120
#define ENC_H    8
#define ENC_DK   160
#define ENC_N    48
#define ENC_CK   9
#define SUB_CH   256

#define DEC_D    1024
#define DEC_FF   4096
#define DEC_H    8
#define DEC_DK   128
#define DEC_N    8

#define VOCAB    16384
#define MAX_SEQ  1024
#define MAX_VOCAB 16384

#define SR       16000
#define N_FFT    512
#define HOP_LEN  160
#define WIN_LEN  400
#define N_MELS   128
#define N_BINS   257

/* ============================================================ */
/* BF16 → float conversion (CPU only; CUDA uses __bfloat162float) */
/* ============================================================ */
static inline float bf16(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

/* ============================================================ */
/* Weight struct types                                           */
/* ============================================================ */
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

#endif /* BACKENDS_MODEL_TYPES_H */
