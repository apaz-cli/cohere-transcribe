/*
 * transcribe.c — cohere-transcribe-03-2026 inference, no ONNX Runtime
 *
 * Single arena allocation.  After audio is loaded (n_samples known) all
 * inference memory comes from one malloc().  The arena is divided into:
 *
 *   Permanent regions (named, carve off the front):
 *     mel        N_MELS × T_mel                 = 128 × T_mel
 *     x_enc      T_enc  × ENC_D                 = T_enc × 1280
 *     pos_emb    (2×T_enc−1) × ENC_D            = (2T−1) × 1280
 *     enc_h      T_enc  × DEC_D                 = T_enc × 1024
 *     ca_k[8]    DEC_N  × T_enc × DEC_D         = 8 × T_enc × 1024
 *     ca_v[8]    DEC_N  × T_enc × DEC_D         = 8 × T_enc × 1024
 *     h_dec      MAX_SEQ × DEC_D                = 1024 × 1024
 *     mask       MAX_SEQ × MAX_SEQ              = 1024 × 1024
 *
 *   Scratch region (work, reused across all phases):
 *     Sized for the maximum peak across phases (see work_size formula below).
 *     Functions receive work and assign named sub-pointers at explicit offsets.
 *
 * Build:  gcc -O3 -march=native -o transcribe transcribe.c -lm -lpthread
 *         (embed model: gcc -O3 -march=native -DEMBEDDED_MODEL -o transcribe transcribe.c -lm -lpthread)
 * Run:    ./transcribe audio.mp3
 */
#define _GNU_SOURCE
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/inotify.h>
#include <dirent.h>
#include <signal.h>
#include <pthread.h>
#include <time.h>

static volatile sig_atomic_t g_stop = 0;
static void handle_sigint(int sig) { (void)sig; g_stop = 1; }

static int verbose = 0;

#include "model.h"
#include "vocab_txt.h"

/* ============================================================
 * Audio loading + mel spectrogram
 * ============================================================ */
static float hann[WIN_LEN];
static void init_hann(void) {
    for (int i = 0; i < WIN_LEN; i++)
        hann[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (WIN_LEN - 1)));
}

/* Mel filterbank: matches librosa.filters.mel(sr=16000, n_fft=512,
 * n_mels=128, fmin=0, fmax=8000, norm='slaney'). */
static float filterbank[N_MELS * N_BINS];

static double hz_to_mel(double f) {
    const double f_sp       = 200.0 / 3.0;
    const double min_log_hz = 1000.0;
    const double min_log_mel = min_log_hz / f_sp;
    const double logstep    = log(6.4) / 27.0;
    return f < min_log_hz ? f / f_sp
                          : min_log_mel + log(f / min_log_hz) / logstep;
}

static double mel_to_hz(double m) {
    const double f_sp       = 200.0 / 3.0;
    const double min_log_hz = 1000.0;
    const double min_log_mel = min_log_hz / f_sp;
    const double logstep    = log(6.4) / 27.0;
    return m < min_log_mel ? f_sp * m
                           : min_log_hz * exp(logstep * (m - min_log_mel));
}

static void init_mel_filterbank(void) {
    double f_pts[N_MELS + 2];
    double mel_lo = hz_to_mel(0.0), mel_hi = hz_to_mel(SR / 2.0);
    for (int i = 0; i < N_MELS + 2; i++)
        f_pts[i] = mel_to_hz(mel_lo + (mel_hi - mel_lo) * i / (N_MELS + 1));

    double fdiff[N_MELS + 1];
    for (int i = 0; i < N_MELS + 1; i++)
        fdiff[i] = f_pts[i + 1] - f_pts[i];

    for (int m = 0; m < N_MELS; m++) {
        double enorm = 2.0 / (f_pts[m + 2] - f_pts[m]);
        for (int i = 0; i < N_BINS; i++) {
            double freq = (SR / 2.0) * i / (N_BINS - 1);
            double lo = (freq - f_pts[m])     / fdiff[m];
            double hi = (f_pts[m+2] - freq)   / fdiff[m+1];
            double w  = lo < hi ? lo : hi;
            filterbank[m * N_BINS + i] = (float)(w > 0.0 ? w * enorm : 0.0);
        }
    }
}

static void fft(float *re, float *im, int N) {
    for (int i = 1, j = 0; i < N; i++) {
        int b = N >> 1;
        for (; j & b; b >>= 1) j ^= b;
        j ^= b;
        if (i < j) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
                  t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= N; len <<= 1) {
        float ang = -2.0f * (float)M_PI / len;
        float wr = cosf(ang), wi = sinf(ang);
        for (int i = 0; i < N; i += len) {
            float cr = 1, ci = 0;
            for (int k = 0; k < len / 2; k++) {
                float ur = re[i+k],        ui = im[i+k];
                float vr = re[i+k+len/2] * cr - im[i+k+len/2] * ci;
                float vi = re[i+k+len/2] * ci + im[i+k+len/2] * cr;
                re[i+k]       = ur + vr;  im[i+k]       = ui + vi;
                re[i+k+len/2] = ur - vr;  im[i+k+len/2] = ui - vi;
                float nc = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = nc;
            }
        }
    }
}

static float *load_audio(const char *path, int *n_out) {
    int pipefd[2];
    if (pipe(pipefd) < 0) { perror("pipe"); exit(1); }

    pid_t pid = fork();
    if (pid < 0) { perror("fork"); exit(1); }
    if (pid == 0) {
        close(pipefd[0]);
        if (dup2(pipefd[1], STDOUT_FILENO) < 0) { perror("dup2"); _exit(1); }
        close(pipefd[1]);
        char sr_str[16];
        snprintf(sr_str, sizeof(sr_str), "%d", SR);
        execlp("ffmpeg", "ffmpeg",
               "-hide_banner", "-loglevel", "error",
               "-i", path,
               "-f", "f32le", "-ar", sr_str, "-ac", "1",
               "pipe:1", (char *)NULL);
        perror("ffmpeg");
        _exit(1);
    }
    close(pipefd[1]);

    FILE *f = fdopen(pipefd[0], "rb");
    if (!f) { perror("fdopen"); exit(1); }

    size_t cap = SR * 60, n = 0;
    float *buf = malloc(cap * sizeof(float));
    if (!buf) { fputs("OOM\n", stderr); exit(1); }
    size_t r;
    while ((r = fread(buf + n, sizeof(float), cap - n, f)) > 0) {
        n += r;
        if (n == cap) {
            cap *= 2;
            float *tmp = realloc(buf, cap * sizeof(float));
            if (!tmp) { free(buf); fputs("OOM\n", stderr); exit(1); }
            buf = tmp;
        }
    }
    fclose(f);

    int status;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "ffmpeg failed for: %s\n", path);
        exit(1);
    }

    *n_out = (int)n;
    return buf;
}

/* compute_mel: writes into mel (N_MELS × T_mel, permanent region)
 * work[0 .. n_samples+WIN_LEN-1]: padded signal (always << work_size) */
static void compute_mel(const float *samples, int N, int T_mel,
                        float *mel, float *work) {
    const int pad = WIN_LEN / 2;
    const int Np  = N + 2 * pad;
    float *s = work;  /* Np floats; Np = N + WIN_LEN */
    memset(s, 0, (size_t)Np * sizeof(float));
    memcpy(s + pad, samples, (size_t)N * sizeof(float));

    for (int i = Np - 1; i > 0; i--) s[i] -= 0.97f * s[i-1];

    float re[N_FFT], im[N_FFT];
    for (int t = 0; t < T_mel; t++) {
        int offset = t * HOP_LEN;
        for (int i = 0; i < N_FFT; i++) re[i] = im[i] = 0.0f;
        for (int i = 0; i < WIN_LEN && offset+i < Np; i++)
            re[i] = s[offset + i] * hann[i];
        fft(re, im, N_FFT);
        float ps[N_BINS];
        for (int i = 0; i < N_BINS; i++) ps[i] = re[i]*re[i] + im[i]*im[i];
        for (int m = 0; m < N_MELS; m++) {
            float v = 0;
            for (int i = 0; i < N_BINS; i++) v += filterbank[m * N_BINS + i] * ps[i];
            mel[m * T_mel + t] = v;
        }
    }

    const float log_zero_guard = powf(2.0f, -24.0f);
    for (int m = 0; m < N_MELS; m++) {
        float *row = mel + m * T_mel;
        float mean = 0, var = 0;
        for (int t = 0; t < T_mel; t++) { row[t] = logf(row[t] + log_zero_guard); mean += row[t]; }
        mean /= T_mel;
        for (int t = 0; t < T_mel; t++) { float d = row[t] - mean; var += d*d; }
        var /= (T_mel - 1);
        float is = 1.0f / sqrtf(var + 1e-5f);
        for (int t = 0; t < T_mel; t++) row[t] = (row[t] - mean) * is;
    }
}

/* ============================================================
 * Vocabulary and token printing
 * ============================================================ */
static char *vocab[MAX_VOCAB];
static int   vocab_size = 0;

static void load_vocab(void) {
    /* vocab_txt is null-delimited (built with '\n' → '\0'): just index it. */
    char *p   = (char *)vocab_txt;
    char *end = p + vocab_txt_len;
    while (p < end && vocab_size < MAX_VOCAB) {
        vocab[vocab_size++] = p;
        while (p < end && *p) p++;
        p++;
    }
}

static int lookup_token(const char *name) {
    for (int i = 0; i < vocab_size; i++)
        if (strcmp(vocab[i], name) == 0) return i;
    return -1;
}

static void print_tokens(const int *tokens, int start, int end, FILE *out) {
    int any = 0;
    for (int i = start; i < end; i++) {
        int id = tokens[i];
        if (id < 0 || id >= vocab_size) continue;
        const char *tok = vocab[id];
        if (tok[0]=='<' && tok[1]=='|') continue;
        const char *p = tok;
        while (*p) {
            if ((unsigned char)p[0]==0xE2 && (unsigned char)p[1]==0x96 && (unsigned char)p[2]==0x81) {
                if (any) fputc(' ', out);
                any = 1;
                p += 3;
            } else {
                fputc(*p++, out);
                any = 1;
            }
        }
    }
    fputc('\n', out);
}

/* ============================================================
 * transcribe_chunk: run full inference on a sample slice
 * ============================================================ */
static void transcribe_chunk(const float *samples, int n_samples,
                              const int *prompt_ids, int n_prompt, FILE *out,
                              int *n_tokens_out) {
    const int T_mel = 1 + n_samples / HOP_LEN;
    const int T1    = (T_mel + 1) / 2;
    const int T2    = (T1   + 1) / 2;
    const int T_enc = (T2   + 1) / 2;
    const int F1 = 64, F2 = 32;

    const size_t n_mel    = (size_t)N_MELS  * T_mel;
    const size_t n_xenc   = (size_t)T_enc   * ENC_D;
    const size_t n_posemb = (size_t)(2*T_enc - 1) * ENC_D;
    const size_t n_ench   = (size_t)T_enc   * DEC_D;
    const size_t n_cakv   = (size_t)DEC_N   * T_enc * DEC_D;
    const size_t n_hdec   = (size_t)MAX_SEQ * DEC_D;
    const size_t n_mask   = (size_t)MAX_SEQ * MAX_SEQ;
    const size_t n_sakv   = (size_t)DEC_N   * MAX_SEQ * DEC_D;  /* per-layer SA KV cache */

    const size_t work_sub  = (size_t)SUB_CH * ((size_t)T1*F1 + (size_t)T2*F2);
    /* MHSA needs T×T + T×(2T-1) = T×(3T-1) floats for scores+scores_bd.
     * This dominates ConvMod for T_enc > ENC_D/3 ≈ 427 (roughly a few minutes of audio). */
    const size_t mhsa_scores = (size_t)T_enc * ((size_t)3 * T_enc - 1);
    const size_t mhsa_attn   = (size_t)2 * T_enc * ENC_D;  /* shared + attn_out */
    const size_t mhsa_shared = mhsa_scores > mhsa_attn ? mhsa_scores : mhsa_attn;
    const size_t work_conf_mhsa = (size_t)T_enc * ENC_FF            /* tmp */
                                + (size_t)T_enc * ENC_D              /* ln */
                                + (size_t)(5*(size_t)T_enc - 1)*ENC_D  /* Q+K+V+P */
                                + mhsa_shared;
    const size_t work_conf_conv = (size_t)T_enc * (ENC_FF + 4*ENC_D);
    const size_t work_conf = work_conf_mhsa > work_conf_conv ? work_conf_mhsa : work_conf_conv;
    const size_t work_dec  = (size_t)5 * MAX_SEQ * DEC_D
                           + (size_t)MAX_SEQ * MAX_SEQ;
    size_t work_size = work_sub;
    if (work_conf > work_size) work_size = work_conf;
    if (work_dec  > work_size) work_size = work_dec;

    const size_t arena_floats = n_mel + n_xenc + n_posemb + n_ench
                              + 2*n_cakv + n_hdec + n_mask + 2*n_sakv + work_size;

    if (verbose) fprintf(stderr, "  chunk %.2fs: T_enc=%d arena=%.1fMB\n",
                         (float)n_samples / SR, T_enc,
                         (double)arena_floats * sizeof(float) / (1024.0 * 1024.0));

    float *arena = backend_arena_alloc(arena_floats);
    if (!arena) { fputs("OOM\n", stderr); exit(1); }

    size_t off = 0;
    float *mel    = arena + off; off += n_mel;
    float *x_enc  = arena + off; off += n_xenc;
    float *pos_emb= arena + off; off += n_posemb;
    float *enc_h  = arena + off; off += n_ench;
    float *ca_k[DEC_N], *ca_v[DEC_N];
    for (int l = 0; l < DEC_N; l++) { ca_k[l] = arena + off; off += (size_t)T_enc * DEC_D; }
    for (int l = 0; l < DEC_N; l++) { ca_v[l] = arena + off; off += (size_t)T_enc * DEC_D; }
    float *h_dec  = arena + off; off += n_hdec;
    float *mask   = arena + off; off += n_mask;
    float *sa_k_cache[DEC_N], *sa_v_cache[DEC_N];
    for (int l = 0; l < DEC_N; l++) { sa_k_cache[l] = arena + off; off += (size_t)MAX_SEQ * DEC_D; }
    for (int l = 0; l < DEC_N; l++) { sa_v_cache[l] = arena + off; off += (size_t)MAX_SEQ * DEC_D; }
    float *work   = arena + off;

    /* Mel is computed on CPU; backend_htod uploads to arena (no-op on CPU). */
    {
        float *mel_cpu  = malloc(n_mel * sizeof(float));
        float *work_cpu = malloc(((size_t)n_samples + WIN_LEN + 1) * sizeof(float));
        if (!mel_cpu || !work_cpu) { fputs("OOM\n", stderr); exit(1); }
        compute_mel(samples, n_samples, T_mel, mel_cpu, work_cpu);
        backend_htod(mel, mel_cpu, n_mel);
        free(mel_cpu);
        free(work_cpu);
    }
    encoder_forward(mel, T_mel, x_enc, T_enc, pos_emb, enc_h, work);
    precompute_ca_kv(enc_h, T_mel, T_enc, ca_k, ca_v);

    int tokens[MAX_SEQ], S = 0;
    for (int i = 0; i < n_prompt; i++) tokens[S++] = prompt_ids[i];
    int prompt_end = S;

    /* Initialise device-side counter so first decode step's backend_decode_inc_S()
     * advances it to n_prompt+1 (the slot for the first generated token). */
    backend_decode_set_S(S);

    for (int step = 0; step < MAX_SEQ - S; step++) {
        int next_id;
        if (step == 0) {
            /* Prefill: embed all prompt tokens, populate KV cache from scratch. */
            next_id = decoder_step(tokens, S, S, T_mel, T_enc,
                                   ca_k, ca_v, sa_k_cache, sa_v_cache,
                                   h_dec, mask, work);
        } else {
            /* Decode: single-graph path keyed by (GRAPH_TAG_DECODER, T_mel, -1). */
            next_id = decoder_step_decode(tokens[S-1], T_mel, T_enc,
                                          ca_k, ca_v, sa_k_cache, sa_v_cache,
                                          h_dec, work);
        }
        tokens[S++] = next_id;
        if (verbose && next_id >= 0 && next_id < vocab_size) {
            const char *tok = vocab[next_id];
            if (!(tok[0] == '<' && tok[1] == '|')) {
                const char *p = tok;
                while (*p) {
                    if ((unsigned char)p[0]==0xE2 && (unsigned char)p[1]==0x96 && (unsigned char)p[2]==0x81) {
                        fputc(' ', stderr);
                        p += 3;
                    } else {
                        fputc(*p++, stderr);
                    }
                }
                fflush(stderr);
            }
        }
        if (next_id == TOK_EOS) break;
    }
    if (verbose) fputc('\n', stderr);

    print_tokens(tokens, prompt_end, S, out);
    if (n_tokens_out) *n_tokens_out = S - prompt_end;
    backend_arena_free(arena);
}

/* ============================================================
 * estimate_silence_thresh: derive RMS silence threshold from
 * the recording's own noise floor (5th-percentile frame RMS,
 * +20 dB headroom).
 * ============================================================ */
static void compute_frame_rms(const float *samples, int n_frames, float *rms) {
    for (int f = 0; f < n_frames; f++) {
        const float *frame = samples + (size_t)f * HOP_LEN;
        float sum = 0.0f;
        for (int s = 0; s < HOP_LEN; s++) sum += frame[s] * frame[s];
        rms[f] = sqrtf(sum / HOP_LEN);
    }
}

static float estimate_silence_thresh(const float *rms, int n_frames) {
    enum { N_BUCKETS = 1024 };
    int hist[N_BUCKETS] = {0};

    for (int f = 0; f < n_frames; f++) {
        int b = (int)(rms[f] * N_BUCKETS);
        hist[b < N_BUCKETS ? b : N_BUCKETS - 1]++;
    }

    int target = n_frames / 20, cumsum = 0;
    for (int b = 0; b < N_BUCKETS; b++) {
        cumsum += hist[b];
        if (cumsum >= target)
            return ((float)b / N_BUCKETS) * 10.0f;  /* +20 dB headroom */
    }
    return 1.0f;  /* unreachable */
}

/* ============================================================
 * find_splits: locate silence-based chunk boundaries
 *
 * Returns split points as sample indices (midpoint of each
 * silence run >= min_silence_sec). Caller provides splits[].
 * ============================================================ */
static int find_splits(const float *rms, int n_frames,
                       float thresh_rms, float min_silence_sec,
                       int *splits, int max_splits) {
    const int min_frames = (int)(min_silence_sec * SR / HOP_LEN);
    int n_splits = 0, in_silence = 0, silence_start = 0;

    for (int f = 0; f < n_frames; f++) {
        if (rms[f] < thresh_rms) {
            if (!in_silence) { silence_start = f; in_silence = 1; }
        } else if (in_silence) {
            if (f - silence_start >= min_frames && n_splits < max_splits)
                splits[n_splits++] = ((silence_start + f) / 2) * HOP_LEN;
            in_silence = 0;
        }
    }
    return n_splits;
}

/* ============================================================
 * apply_target: merge/split silence candidates to hit target chunk size
 *
 * Walks the candidate split points and only emits a split once the
 * accumulated samples since the last split reach target_samp.  This
 * combines short adjacent segments and forces long silent-free stretches
 * to split at the best available silence boundary.
 * ============================================================ */
static int apply_target(int *splits, int n_splits, int target_samp) {
    int out = 0, prev = 0;
    for (int i = 0; i < n_splits; i++) {
        if (splits[i] - prev >= target_samp) {
            splits[out++] = splits[i];
            prev = splits[i];
        }
    }
    return out;
}

/* ============================================================
 * Parallel chunk runner
 *
 * To revert to sequential processing, delete this section and
 * replace the run_chunks_parallel() call in main with:
 *
 *   int start = 0;
 *   for (int c = 0; c <= n_splits; c++) {
 *       int end = (c < n_splits) ? splits[c] : n_samples;
 *       if (end > start)
 *           transcribe_chunk(samples+start, end-start, prompt_ids, n_prompt, stdout);
 *       start = end;
 *   }
 * ============================================================ */
typedef struct {
    const float *samples;
    int          n_samples;
    const int   *prompt_ids;
    int          n_prompt;
    char        *text;      /* written by thread via open_memstream */
    size_t       text_len;
    int          n_tokens;
    int          orig_idx;  /* position in original audio order, for sorted output */
} ChunkWork;

/* ============================================================
 * Batch transcription
 *
 * Runs the encoder batched on 'count' chunks (all padded to T_mel_max),
 * then runs one joint autoregressive decode loop across all sequences,
 * masking out sequences that have already produced EOS.
 * Falls through to transcribe_chunk for count == 1.
 * ============================================================ */
static void transcribe_batch(ChunkWork *jobs, int count,
                              const int *prompt_ids, int n_prompt) {
    if (count == 1) {
        FILE *f = open_memstream(&jobs[0].text, &jobs[0].text_len);
        transcribe_chunk(jobs[0].samples, jobs[0].n_samples,
                         jobs[0].prompt_ids, jobs[0].n_prompt,
                         f, &jobs[0].n_tokens);
        fclose(f);
        return;
    }

    /* ---- 1. Compute T_mel / T_enc for each item ---- */
    int *T_mel = malloc((size_t)count * sizeof(int));
    int *T_enc = malloc((size_t)count * sizeof(int));
    if (!T_mel || !T_enc) { fputs("OOM\n", stderr); exit(1); }

    int T_mel_max = 0;
    for (int b = 0; b < count; b++) {
        T_mel[b] = 1 + jobs[b].n_samples / HOP_LEN;
        int T1 = (T_mel[b]+1)/2, T2 = (T1+1)/2;
        T_enc[b] = (T2+1)/2;
        if (T_mel[b] > T_mel_max) T_mel_max = T_mel[b];
    }
    int T1_max = (T_mel_max+1)/2, T2_max = (T1_max+1)/2;
    int T_enc_max = (T2_max+1)/2;

    /* ---- 2. Arena sizing ---- */
    const size_t n_mel_b   = (size_t)count * N_MELS * T_mel_max;
    const size_t n_xenc_b  = (size_t)count * T_enc_max * ENC_D;
    const size_t n_posemb  = (size_t)(2*T_enc_max - 1) * ENC_D;
    const size_t n_ench_b  = (size_t)count * T_enc_max * DEC_D;
    const size_t n_cakv_b  = (size_t)count * DEC_N * T_enc_max * DEC_D; /* ca_k + ca_v */
    const size_t n_sakv_b  = (size_t)count * DEC_N * MAX_SEQ * DEC_D;  /* sa_k + sa_v */
    const size_t n_hdec    = (size_t)MAX_SEQ * DEC_D;     /* prefill hidden (one item) */
    const size_t n_mask    = (size_t)MAX_SEQ * MAX_SEQ;   /* prefill causal mask */
    const size_t n_logits  = (size_t)count * VOCAB;       /* batch logit scratch */
    const size_t n_h_dec_b = (size_t)count * DEC_D;       /* decode step hidden */

    const size_t F1 = 64, F2 = 32;
    const size_t work_sub = (size_t)SUB_CH *
                            ((size_t)T1_max * F1 + (size_t)T2_max * F2);
    const size_t mhsa_scores = (size_t)T_enc_max * ((size_t)3*T_enc_max - 1);
    const size_t mhsa_attn   = (size_t)2 * T_enc_max * ENC_D;
    const size_t mhsa_shared = mhsa_scores > mhsa_attn ? mhsa_scores : mhsa_attn;
    const size_t work_conf_mhsa = (size_t)T_enc_max * ENC_FF
                                + (size_t)T_enc_max * ENC_D
                                + (size_t)(5*(size_t)T_enc_max - 1) * ENC_D
                                + mhsa_shared;
    const size_t work_conf_conv = (size_t)T_enc_max * (ENC_FF + 4*ENC_D);
    const size_t work_conf = work_conf_mhsa > work_conf_conv ? work_conf_mhsa : work_conf_conv;
    const size_t work_dec  = (size_t)5 * MAX_SEQ * DEC_D + (size_t)MAX_SEQ * MAX_SEQ;
    size_t work_size = work_sub;
    if (work_conf > work_size) work_size = work_conf;
    if (work_dec  > work_size) work_size = work_dec;

    const size_t arena_floats = n_mel_b + n_xenc_b + n_posemb + n_ench_b
                              + 2*n_cakv_b + 2*n_sakv_b
                              + n_hdec + n_mask + n_logits + n_h_dec_b
                              + work_size;

    if (verbose)
        fprintf(stderr, "  batch %d chunks: T_enc_max=%d arena=%.1fMB\n",
                count, T_enc_max,
                (double)arena_floats * sizeof(float) / (1024.0*1024.0));

    float *arena = backend_arena_alloc(arena_floats);
    if (!arena) { fputs("OOM\n", stderr); exit(1); }

    /* ---- 3. Lay out the arena ---- */
    size_t off = 0;
    float *mel_batch  = arena + off; off += n_mel_b;
    float *x_enc_batch= arena + off; off += n_xenc_b;
    float *pos_emb    = arena + off; off += n_posemb;
    float *enc_h_batch= arena + off; off += n_ench_b;
    float *ca_k_batch[DEC_N], *ca_v_batch[DEC_N];
    for (int l = 0; l < DEC_N; l++) {
        ca_k_batch[l] = arena + off; off += (size_t)count * T_enc_max * DEC_D;
    }
    for (int l = 0; l < DEC_N; l++) {
        ca_v_batch[l] = arena + off; off += (size_t)count * T_enc_max * DEC_D;
    }
    float *sa_k_batch[DEC_N], *sa_v_batch[DEC_N];
    for (int l = 0; l < DEC_N; l++) {
        sa_k_batch[l] = arena + off; off += (size_t)count * MAX_SEQ * DEC_D;
    }
    for (int l = 0; l < DEC_N; l++) {
        sa_v_batch[l] = arena + off; off += (size_t)count * MAX_SEQ * DEC_D;
    }
    float *h_dec      = arena + off; off += n_hdec;
    float *mask       = arena + off; off += n_mask;
    float *logits_b   = arena + off; off += n_logits;
    float *h_dec_b    = arena + off; off += n_h_dec_b;
    float *work       = arena + off;

    /* ---- 4. Compute mels on CPU; upload to batch arena (padded) ---- */
    for (int b = 0; b < count; b++) {
        size_t n_mel_cpu = (size_t)N_MELS * T_mel[b];
        float *mel_cpu  = malloc(n_mel_cpu * sizeof(float));
        float *work_cpu = malloc(((size_t)jobs[b].n_samples + WIN_LEN + 1) * sizeof(float));
        if (!mel_cpu || !work_cpu) { fputs("OOM\n", stderr); exit(1); }
        compute_mel(jobs[b].samples, jobs[b].n_samples, T_mel[b], mel_cpu, work_cpu);
        free(work_cpu);

        /* Build padded mel host buffer [N_MELS][T_mel_max] */
        float *mel_pad = calloc((size_t)N_MELS * T_mel_max, sizeof(float));
        if (!mel_pad) { fputs("OOM\n", stderr); exit(1); }
        for (int m = 0; m < N_MELS; m++)
            memcpy(mel_pad + (size_t)m * T_mel_max,
                   mel_cpu + (size_t)m * T_mel[b],
                   T_mel[b] * sizeof(float));
        free(mel_cpu);

        backend_htod(mel_batch + (size_t)b * N_MELS * T_mel_max,
                     mel_pad, (size_t)N_MELS * T_mel_max);
        free(mel_pad);
    }

    /* ---- 5. Batched encoder forward ---- */
    encoder_forward_batch(mel_batch, count, T_mel_max, T_mel, T_enc, T_enc_max,
                          x_enc_batch, pos_emb, enc_h_batch, work);

    /* ---- 6. Batched cross-attention K/V precompute ---- */
    precompute_ca_kv_batch(enc_h_batch, count, T_enc, T_enc_max,
                           ca_k_batch, ca_v_batch);

    /* ---- 7. Per-item prefill (sequential) ---- */
    int   *tokens  = malloc((size_t)count * MAX_SEQ * sizeof(int));
    int   *S       = malloc((size_t)count * sizeof(int));
    int   *S_dev   = malloc((size_t)count * sizeof(int));
    if (!tokens || !S || !S_dev) { fputs("OOM\n", stderr); exit(1); }

    for (int b = 0; b < count; b++) {
        int *tok_b = tokens + b * MAX_SEQ;
        for (int i = 0; i < n_prompt; i++) tok_b[i] = prompt_ids[i];
        S[b]     = n_prompt;
        S_dev[b] = n_prompt;

        float *ca_k_b[DEC_N], *ca_v_b[DEC_N];
        float *sa_k_b[DEC_N], *sa_v_b[DEC_N];
        for (int l = 0; l < DEC_N; l++) {
            ca_k_b[l] = ca_k_batch[l] + (size_t)b * T_enc_max * DEC_D;
            ca_v_b[l] = ca_v_batch[l] + (size_t)b * T_enc_max * DEC_D;
            sa_k_b[l] = sa_k_batch[l] + (size_t)b * MAX_SEQ * DEC_D;
            sa_v_b[l] = sa_v_batch[l] + (size_t)b * MAX_SEQ * DEC_D;
        }

        /* backend_decode_set_S initialises the device counter for the prefill
         * graph key; the batch decode loop uses S_dev[] directly instead. */
        backend_decode_set_S(n_prompt);

        /* Use a graph key that encodes both T_mel and item index b so that two
         * items within the same batch with identical T_mel never share a graph
         * (their sa_k/ca_k addresses differ by b * stride). */
        int next_id = decoder_step(tok_b, S[b], S[b],
                                    T_mel[b] + (b + 1) * 100000, T_enc[b],
                                    ca_k_b, ca_v_b, sa_k_b, sa_v_b,
                                    h_dec, mask, work);
        tok_b[S[b]++] = next_id;
        if (verbose) fprintf(stderr, "  prefill item %d: %d tokens\n", b, S[b]);
    }

    /* Upload per-item T_enc to device once; used by sdp_attn_batch_decode_ca. */
    backend_upload_T_enc(T_enc, count);

    /* ---- 8. Batch decode loop ---- */
    int *active = malloc((size_t)count * sizeof(int));
    if (!active) { fputs("OOM\n", stderr); exit(1); }

    int n_active = 0;
    for (int b = 0; b < count; b++) {
        int *tok_b = tokens + b * MAX_SEQ;
        active[b] = (tok_b[S[b]-1] != TOK_EOS && S[b] < MAX_SEQ);
        if (active[b]) n_active++;
        /* S_dev[b] = S[b]-1: device counter is one behind host count (same semantics
         * as the single-item path where backend_decode_set_S(n_prompt) is called before
         * the first decoder_step_decode that increments S_dev to n_prompt+1). */
        S_dev[b] = S[b] - 1;
    }

    int  *tokens_cur  = malloc((size_t)count * sizeof(int));
    int  *next_tokens = malloc((size_t)count * sizeof(int));
    if (!tokens_cur || !next_tokens) { fputs("OOM\n", stderr); exit(1); }

    while (n_active > 0) {
        /* Collect the token each active item will embed this step */
        for (int b = 0; b < count; b++) {
            int *tok_b = tokens + b * MAX_SEQ;
            tokens_cur[b] = active[b] ? tok_b[S[b]-1] : 0;
        }

        /* Advance device sequence counters for active items */
        for (int b = 0; b < count; b++) if (active[b]) S_dev[b]++;

        /* Run one batched decode step */
        decoder_step_decode_batch(tokens_cur, S_dev, active,
                                   T_enc_max,
                                   ca_k_batch, ca_v_batch,
                                   sa_k_batch, sa_v_batch,
                                   h_dec_b, logits_b, work, count,
                                   next_tokens);

        /* Store tokens and update active flags */
        for (int b = 0; b < count; b++) {
            if (!active[b]) continue;
            int *tok_b = tokens + b * MAX_SEQ;
            tok_b[S[b]] = next_tokens[b];
            S[b]++;
            if (verbose && next_tokens[b] >= 0 && next_tokens[b] < vocab_size) {
                const char *tok = vocab[next_tokens[b]];
                if (!(tok[0]=='<' && tok[1]=='|')) {
                    const char *p = tok;
                    while (*p) {
                        if ((unsigned char)p[0]==0xE2 && (unsigned char)p[1]==0x96
                                                       && (unsigned char)p[2]==0x81)
                            { fputc(' ', stderr); p+=3; }
                        else fputc(*p++, stderr);
                    }
                    fflush(stderr);
                }
            }
            if (next_tokens[b] == TOK_EOS || S[b] >= MAX_SEQ) {
                active[b] = 0;
                n_active--;
                if (verbose) fputc('\n', stderr);
            }
        }
    }

    /* ---- 9. Output results in order ---- */
    for (int b = 0; b < count; b++) {
        FILE *f = open_memstream(&jobs[b].text, &jobs[b].text_len);
        print_tokens(tokens + b * MAX_SEQ, n_prompt, S[b], f);
        fclose(f);
        jobs[b].n_tokens = S[b] - n_prompt;
    }

    backend_arena_free(arena);
    free(T_mel); free(T_enc); free(tokens); free(S); free(S_dev);
    free(active); free(tokens_cur); free(next_tokens);
}

typedef struct {
    ChunkWork *jobs;
    int        count;     /* number of jobs in this group (≤ batch_size) */
} BatchGroup;

typedef struct {
    BatchGroup     *groups;
    int             n_groups;
    int             next;      /* index of next unclaimed group */
    int             completed; /* jobs completed so far (for progress reporting) */
    int             n_jobs;    /* total individual chunks */
    pthread_mutex_t mu;
} Queue;

typedef struct {
    Queue *q;
    int    gpu_id;
    int    batch_size;
} WorkerArg;

static int cmp_n_samples_desc(const void *a, const void *b) {
    return ((const ChunkWork *)b)->n_samples - ((const ChunkWork *)a)->n_samples;
}

static void *worker(void *varg) {
    WorkerArg *arg = (WorkerArg *)varg;
    backend_init(arg->gpu_id);
    Queue *q = arg->q;
    for (;;) {
        pthread_mutex_lock(&q->mu);
        int gi = q->next++;
        pthread_mutex_unlock(&q->mu);
        if (gi >= q->n_groups) break;
        BatchGroup *g = &q->groups[gi];
        transcribe_batch(g->jobs, g->count,
                         g->jobs[0].prompt_ids, g->jobs[0].n_prompt);
        pthread_mutex_lock(&q->mu);
        int done = (q->completed += g->count);
        pthread_mutex_unlock(&q->mu);
        if (verbose) fprintf(stderr, "[%d/%d batch=%d]\n", done, q->n_jobs, g->count);
    }
    backend_thread_cleanup();
    return NULL;
}

static void run_chunks_parallel(const float *samples,
                                const int *splits, int n_splits, int n_samples,
                                const int *prompt_ids, int n_prompt,
                                int batch_size, FILE *out) {
    /* Build job list, skipping zero-length chunks */
    int n_chunks = n_splits + 1;
    ChunkWork *jobs = calloc((size_t)n_chunks, sizeof(ChunkWork));
    if (!jobs) { fputs("OOM\n", stderr); exit(1); }

    int n_jobs = 0, start = 0;
    for (int c = 0; c < n_chunks; c++) {
        int end = (c < n_splits) ? splits[c] : n_samples;
        if (end > start) {
            jobs[n_jobs].samples    = samples + start;
            jobs[n_jobs].n_samples  = end - start;
            jobs[n_jobs].prompt_ids = prompt_ids;
            jobs[n_jobs].n_prompt   = n_prompt;
            jobs[n_jobs].orig_idx   = n_jobs;
            n_jobs++;
        }
        start = end;
    }

    /* Sort jobs by length descending so batches contain similarly-sized chunks,
     * minimising padding in the encoder.  orig_idx preserves audio ordering. */
    qsort(jobs, (size_t)n_jobs, sizeof(ChunkWork), cmp_n_samples_desc);

    /* Partition into batch groups */
    int n_groups = (n_jobs + batch_size - 1) / batch_size;
    BatchGroup *groups = calloc((size_t)n_groups, sizeof(BatchGroup));
    if (!groups) { fputs("OOM\n", stderr); exit(1); }
    for (int g = 0; g < n_groups; g++) {
        groups[g].jobs  = jobs + g * batch_size;
        groups[g].count = (g * batch_size + batch_size <= n_jobs)
                          ? batch_size : (n_jobs - g * batch_size);
    }

    int n_devices = backend_num_devices();
    long nthreads = n_devices;
    if (nthreads < 1) nthreads = 1;
    if (nthreads > n_groups) nthreads = n_groups;
    if (verbose) fprintf(stderr, "  running %d chunks in %d batch(es) on %ld thread(s)\n",
                         n_jobs, n_groups, nthreads);
    pthread_t  *tids = calloc((size_t)nthreads, sizeof(pthread_t));
    WorkerArg  *args = calloc((size_t)nthreads, sizeof(WorkerArg));
    if (!tids || !args) { fputs("OOM\n", stderr); exit(1); }

    Queue q = { groups, n_groups, 0, 0, n_jobs, PTHREAD_MUTEX_INITIALIZER };
    for (int i = 0; i < (int)nthreads; i++) {
        args[i].q          = &q;
        args[i].gpu_id     = i % n_devices;
        args[i].batch_size = batch_size;
        pthread_create(&tids[i], NULL, worker, &args[i]);
    }
    for (int i = 0; i < (int)nthreads; i++)
        pthread_join(tids[i], NULL);

    /* Restore audio order: jobs were sorted by length, output must follow
     * the original silence-split sequence. */
    char   **texts     = calloc((size_t)n_jobs, sizeof(char *));
    size_t  *text_lens = calloc((size_t)n_jobs, sizeof(size_t));
    if (!texts || !text_lens) { fputs("OOM\n", stderr); exit(1); }
    for (int i = 0; i < n_jobs; i++) {
        texts    [jobs[i].orig_idx] = jobs[i].text;
        text_lens[jobs[i].orig_idx] = jobs[i].text_len;
    }
    for (int i = 0; i < n_jobs; i++) {
        fwrite(texts[i], 1, text_lens[i], out);
        free(texts[i]);
    }
    free(texts);
    free(text_lens);

    pthread_mutex_destroy(&q.mu);
    free(groups);
    free(args);
    free(tids);
    free(jobs);
}

/* ============================================================
 * Transcribe a single audio file, writing output to `out`.
 * Returns 0 on success, 1 on error (does not call exit).
 * ============================================================ */
static int do_transcribe_file(const char *audio_file, FILE *out,
                               const int *prompt_ids, int n_prompt,
                               int opt_chunk, int opt_batch_size,
                               float opt_silence_db, float opt_silence_dur,
                               float opt_target_sec) {
    int n_samples;
    if (verbose) fprintf(stderr, "Loading %s ...\n", audio_file);
    float *samples = load_audio(audio_file, &n_samples);
    if (verbose) fprintf(stderr, "  Loaded %.2fs of audio\n", (float)n_samples / SR);

    if (!opt_chunk && n_samples > SR * 120) {
        fprintf(stderr,
            "Error: %.0fs of audio is too long for single-pass mode.\n"
            "Use --chunk to split on silence and transcribe in segments.\n",
            (float)n_samples / SR);
        free(samples);
        return 1;
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    if (!opt_chunk) {
        if (verbose) fprintf(stderr, "Transcribing %.2fs single-pass ...\n", (float)n_samples / SR);
        int n_tokens = 0;
        transcribe_chunk(samples, n_samples, prompt_ids, n_prompt, out, &n_tokens);
        if (verbose) fprintf(stderr, "  %d tokens\n", n_tokens);
    } else {
        int n_frames = n_samples / HOP_LEN;
        float *frame_rms = malloc(n_frames * sizeof(float));
        if (!frame_rms) { fputs("OOM\n", stderr); free(samples); return 1; }
        compute_frame_rms(samples, n_frames, frame_rms);

        float thresh_rms;
        if (isnan(opt_silence_db)) {
            thresh_rms = estimate_silence_thresh(frame_rms, n_frames);
            if (verbose) fprintf(stderr, "  silence threshold: %.1f dB (auto)\n",
                    20.0f * log10f(thresh_rms + 1e-9f));
        } else {
            thresh_rms = powf(10.0f, opt_silence_db / 20.0f);
        }

        int splits[4096];
        int n_splits = find_splits(frame_rms, n_frames,
                                   thresh_rms, opt_silence_dur,
                                   splits, 4096);
        free(frame_rms);
        n_splits = apply_target(splits, n_splits, (int)(opt_target_sec * SR));
        if (verbose) fprintf(stderr, "  %d splits → %d chunks (target %.0fs)\n",
                             n_splits, n_splits + 1, opt_target_sec);

        run_chunks_parallel(samples, splits, n_splits, n_samples, prompt_ids, n_prompt,
                            opt_batch_size, out);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (verbose) {
        double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        fprintf(stderr, "Done in %.2fs (%.1fx realtime)\n",
                elapsed, (float)n_samples / SR / elapsed);
    }

    free(samples);
    return 0;
}

/* Run ffprobe on `path`; returns 1 if it contains an audio stream, 0 otherwise. */
static int ffprobe_run(const char *path) {
    int pipefd[2];
    if (pipe(pipefd) < 0) { perror("pipe"); return 0; }
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); close(pipefd[0]); close(pipefd[1]); return 0; }
    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);
        execlp("ffprobe", "ffprobe",
               "-v", "error",
               "-select_streams", "a",
               "-show_entries", "stream=codec_type",
               "-of", "csv=p=0",
               path, (char *)NULL);
        _exit(1);
    }
    close(pipefd[1]);
    char buf[256];
    int found = 0;
    ssize_t n;
    while ((n = read(pipefd[0], buf, sizeof(buf))) > 0) {
        for (ssize_t k = 0; k + 4 < n; k++) {
            if (buf[k]=='a' && buf[k+1]=='u' && buf[k+2]=='d' && buf[k+3]=='i' && buf[k+4]=='o')
                found = 1;
        }
    }
    close(pipefd[0]);
    int status;
    waitpid(pid, &status, 0);
    return found && WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

typedef struct {
    char **paths;
    int   *results;
    int    n;
    int    next;
    pthread_mutex_t mu;
} ProbeQueue;

static void *probe_worker(void *varg) {
    ProbeQueue *q = (ProbeQueue *)varg;
    for (;;) {
        pthread_mutex_lock(&q->mu);
        int i = q->next++;
        pthread_mutex_unlock(&q->mu);
        if (i >= q->n) break;
        q->results[i] = ffprobe_run(q->paths[i]);
    }
    return NULL;
}

/* Probe all `n` paths using 2*nproc threads; writes results into `results[i]`. */
static void ffprobe_batch(char **paths, int n, int *results) {
    if (n == 0) return;
    long nproc = sysconf(_SC_NPROCESSORS_ONLN);
    int nthreads = (nproc > 0 ? (int)nproc : 1) * 2;
    if (nthreads > n) nthreads = n;

    ProbeQueue q = { paths, results, n, 0, PTHREAD_MUTEX_INITIALIZER };
    pthread_t *tids = malloc((size_t)nthreads * sizeof(pthread_t));
    if (!tids) { fputs("OOM\n", stderr); memset(results, 0, (size_t)n * sizeof(int)); return; }
    for (int i = 0; i < nthreads; i++)
        pthread_create(&tids[i], NULL, probe_worker, &q);
    for (int i = 0; i < nthreads; i++)
        pthread_join(tids[i], NULL);
    free(tids);
    pthread_mutex_destroy(&q.mu);
}

/* ============================================================
 * Watch mode: initial scan + inotify loop
 * ============================================================ */
static void run_watch(const char *input_folder, const char *output_folder,
                      const int *prompt_ids, int n_prompt,
                      int opt_chunk, int opt_batch_size,
                      float opt_silence_db, float opt_silence_dur,
                      float opt_target_sec) {
    /* ------ initial scan: process any files not yet transcribed ------ */
    DIR *dir = opendir(input_folder);
    if (!dir) { perror(input_folder); return; }

    char **entries = NULL;
    int n_entries = 0, cap_entries = 0;
    struct dirent *de;
    while ((de = readdir(dir)) != NULL) {
        if (de->d_name[0] == '.') continue;
        if (n_entries == cap_entries) {
            cap_entries = cap_entries ? cap_entries * 2 : 64;
            char **tmp = realloc(entries, (size_t)cap_entries * sizeof(char *));
            if (!tmp) { fputs("OOM\n", stderr); closedir(dir); return; }
            entries = tmp;
        }
        entries[n_entries++] = strdup(de->d_name);
    }
    closedir(dir);

    int cmp_str(const void *a, const void *b) {
        return strcmp(*(const char **)a, *(const char **)b);
    }
    qsort(entries, (size_t)n_entries, sizeof(char *), cmp_str);

    char **in_paths = malloc((size_t)n_entries * sizeof(char *));
    int  *is_audio  = malloc((size_t)n_entries * sizeof(int));
    if (!in_paths || !is_audio) { fputs("OOM\n", stderr); return; }
    for (int f = 0; f < n_entries; f++) {
        in_paths[f] = malloc(4096);
        if (!in_paths[f]) { fputs("OOM\n", stderr); return; }
        snprintf(in_paths[f], 4096, "%s/%s", input_folder, entries[f]);
    }
    if (n_entries > 0)
        ffprobe_batch(in_paths, n_entries, is_audio);

    for (int f = 0; f < n_entries; f++) {
        if (!is_audio[f]) { free(entries[f]); free(in_paths[f]); continue; }

        char base[1024], out_path[4096];
        snprintf(base, sizeof(base), "%s", entries[f]);
        char *dot = strrchr(base, '.');
        if (dot) *dot = '\0';
        snprintf(out_path, sizeof(out_path), "%s/%s.txt", output_folder, base);

        struct stat st;
        if (stat(out_path, &st) == 0) {
            if (verbose) fprintf(stderr, "Skipping %s (output exists)\n", entries[f]);
            free(entries[f]); free(in_paths[f]);
            continue;
        }

        FILE *out = fopen(out_path, "w");
        if (!out) { fprintf(stderr, "Cannot open %s\n", out_path); free(entries[f]); free(in_paths[f]); continue; }
        do_transcribe_file(in_paths[f], out, prompt_ids, n_prompt,
                           opt_chunk, opt_batch_size,
                           opt_silence_db, opt_silence_dur, opt_target_sec);
        fclose(out);
        free(entries[f]); free(in_paths[f]);
    }
    free(entries); free(in_paths); free(is_audio);

    /* ------ inotify loop ------ */
    int ifd = inotify_init1(IN_NONBLOCK);
    if (ifd < 0) { perror("inotify_init1"); return; }
    if (inotify_add_watch(ifd, input_folder, IN_CLOSE_WRITE | IN_MOVED_TO) < 0) {
        perror("inotify_add_watch"); close(ifd); return;
    }

    signal(SIGINT,  handle_sigint);
    signal(SIGTERM, handle_sigint);

    if (verbose) fprintf(stderr, "Watching %s ...\n", input_folder);

    char buf[4096] __attribute__((aligned(__alignof__(struct inotify_event))));
    while (!g_stop) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(ifd, &rfds);
        struct timeval tv = { 1, 0 };
        if (select(ifd + 1, &rfds, NULL, NULL, &tv) <= 0) continue;

        ssize_t len = read(ifd, buf, sizeof(buf));
        if (len <= 0) continue;

        char *p = buf;
        while (p < buf + len) {
            struct inotify_event *ev = (struct inotify_event *)p;
            p += sizeof(struct inotify_event) + ev->len;

            if (!ev->len || (ev->mask & IN_ISDIR)) continue;

            char in_path[4096], base[1024], out_path[4096];
            snprintf(in_path, sizeof(in_path), "%s/%s", input_folder, ev->name);
            snprintf(base, sizeof(base), "%s", ev->name);
            char *dot = strrchr(base, '.');
            if (dot) *dot = '\0';
            snprintf(out_path, sizeof(out_path), "%s/%s.txt", output_folder, base);

            struct stat st;
            if (stat(out_path, &st) == 0) continue;  /* already transcribed */

            if (!ffprobe_run(in_path)) {
                if (verbose) fprintf(stderr, "Skipping %s (not audio)\n", ev->name);
                continue;
            }

            FILE *out = fopen(out_path, "w");
            if (!out) { fprintf(stderr, "Cannot open %s\n", out_path); continue; }
            do_transcribe_file(in_path, out, prompt_ids, n_prompt,
                               opt_chunk, opt_batch_size,
                               opt_silence_db, opt_silence_dur, opt_target_sec);
            fclose(out);
        }
    }
    close(ifd);
}

/* ============================================================
 * main
 * ============================================================ */
int main(int argc, char **argv) {
    const char *lang              = "en";
    int opt_pnc                   = 1;
    int opt_itn                   = 0;
    int opt_timestamps            = 0;
    int opt_diarize               = 0;
    int opt_chunk                 = 0;
    int opt_batch_size            = 1;
    float opt_silence_db          = NAN;   /* NAN = auto-estimate from recording */
    float opt_silence_dur         = 0.2f;
    float opt_target_sec          = 20.0f;
    const char *opt_output        = NULL;  /* -o / --output */
    const char *opt_input_folder  = NULL;  /* --input_folder */
    const char *opt_output_folder = NULL;  /* --output_folder */
    int opt_skip_existing         = 0;     /* --skip_existing_output */
    int opt_watch                 = 0;     /* --watch */

    int i;
    for (i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "--lang")             == 0 && i+1 < argc) lang = argv[++i];
        else if (strcmp(argv[i], "--no-pnc")           == 0) opt_pnc = 0;
        else if (strcmp(argv[i], "--itn")              == 0) opt_itn = 1;
        else if (strcmp(argv[i], "--timestamps")       == 0) opt_timestamps = 1;
        else if (strcmp(argv[i], "--diarize")          == 0) opt_diarize = 1;
        else if (strcmp(argv[i], "--chunk")            == 0) opt_chunk = 1;
        else if (strcmp(argv[i], "--batch-size")       == 0 && i+1 < argc) opt_batch_size  = atoi(argv[++i]);
        else if (strcmp(argv[i], "--silence-db")       == 0 && i+1 < argc) opt_silence_db  = atof(argv[++i]);
        else if (strcmp(argv[i], "--silence-dur")      == 0 && i+1 < argc) opt_silence_dur = atof(argv[++i]);
        else if (strcmp(argv[i], "--target-sec")       == 0 && i+1 < argc) opt_target_sec  = atof(argv[++i]);
        else if ((strcmp(argv[i], "-o")  == 0 ||
                  strcmp(argv[i], "--output") == 0)    && i+1 < argc) opt_output = argv[++i];
        else if (strcmp(argv[i], "--input_folder")     == 0 && i+1 < argc) opt_input_folder  = argv[++i];
        else if (strcmp(argv[i], "--output_folder")    == 0 && i+1 < argc) opt_output_folder = argv[++i];
        else if (strcmp(argv[i], "--skip_existing_output") == 0) opt_skip_existing = 1;
        else if (strcmp(argv[i], "--watch")                == 0) opt_watch = 1;
        else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) verbose = 1;
        else if (argv[i][0] == '-') {
            fprintf(stderr, "Unknown flag: %s\n", argv[i]);
            fprintf(stderr,
                "Usage: %s [-v] [--lang CODE] [--no-pnc] [--itn] [--timestamps] [--diarize]\n"
                "       [--chunk [--batch-size N] [--silence-db DB] [--silence-dur SEC]\n"
                "        [--target-sec SEC]]\n"
                "       [-o OUTPUT_FILE] audio_file\n"
                "       --input_folder DIR [--output_folder DIR] [--skip_existing_output]\n",
                argv[0]);
            return 1;
        } else break;
    }

    /* Must have either a positional audio_file or --input_folder */
    int has_positional = (i < argc);
    if (!has_positional && !opt_input_folder) {
        fprintf(stderr,
            "Usage: %s [-v] [--lang CODE] [--no-pnc] [--itn] [--timestamps] [--diarize]\n"
            "       [--chunk [--batch-size N] [--silence-db DB] [--silence-dur SEC]\n"
            "        [--target-sec SEC]]\n"
            "       [-o OUTPUT_FILE] audio_file\n"
            "       --input_folder DIR [--output_folder DIR] [--skip_existing_output]\n",
            argv[0]);
        return 1;
    }
    if (opt_batch_size < 1) {
        fprintf(stderr, "Error: --batch-size must be >= 1\n");
        return 1;
    }
    if (opt_output && opt_input_folder) {
        fprintf(stderr, "Error: -o/--output cannot be combined with --input_folder\n");
        return 1;
    }
    if (opt_watch && !opt_input_folder) {
        fprintf(stderr, "Error: --watch requires --input_folder\n");
        return 1;
    }
    if (opt_watch && !opt_output_folder) {
        fprintf(stderr, "Error: --watch requires --output_folder\n");
        return 1;
    }

    load_vocab();

    char lang_tok[16];
    snprintf(lang_tok, sizeof(lang_tok), "<|%s|>", lang);
    int lang_id = lookup_token(lang_tok);
    if (lang_id < 0) { fprintf(stderr, "Unknown language: %s\n", lang); return 1; }

    int prompt_ids[32], n_prompt = 0;
    prompt_ids[n_prompt++] = TOK_LEADING_SPACE;
    prompt_ids[n_prompt++] = TOK_STARTOFCONTEXT;
    prompt_ids[n_prompt++] = TOK_STARTOFTRANSCRIPT;
    prompt_ids[n_prompt++] = TOK_EMO_UNDEFINED;
    prompt_ids[n_prompt++] = lang_id;
    prompt_ids[n_prompt++] = lang_id;
    prompt_ids[n_prompt++] = opt_pnc        ? TOK_PNC       : TOK_NOPNC;
    prompt_ids[n_prompt++] = opt_itn        ? TOK_ITN       : TOK_NOITN;
    prompt_ids[n_prompt++] = opt_timestamps ? TOK_TIMESTAMP : TOK_NOTIMESTAMP;
    prompt_ids[n_prompt++] = opt_diarize    ? TOK_DIARIZE   : TOK_NODIARIZE;

    if (verbose) {
        fprintf(stderr, "Options: lang=%s pnc=%d itn=%d timestamps=%d diarize=%d chunk=%d",
                lang, opt_pnc, opt_itn, opt_timestamps, opt_diarize, opt_chunk);
        if (opt_chunk)
            fprintf(stderr, " silence_dur=%.2fs target=%.0fs%s",
                    opt_silence_dur, opt_target_sec,
                    isnan(opt_silence_db) ? " silence_db=auto" : "");
        fputc('\n', stderr);
    }

    if (verbose) fprintf(stderr, "Loading model weights ...\n");
    load_weights();
#if defined(BACKEND_CUDA) && !defined(EMBEDDED_MODEL)
    backend_weights_upload_blob(sf_data, sf_data_bytes);
#endif
    backend_init(0);

    init_hann();
    init_mel_filterbank();

    int ret = 0;

    if (opt_watch) {
        run_watch(opt_input_folder, opt_output_folder,
                  prompt_ids, n_prompt,
                  opt_chunk, opt_batch_size,
                  opt_silence_db, opt_silence_dur, opt_target_sec);
    } else if (opt_input_folder) {
        /* --input_folder mode: iterate over audio files in the directory */
        DIR *dir = opendir(opt_input_folder);
        if (!dir) {
            perror(opt_input_folder);
            backend_destroy();
            return 1;
        }

        /* Collect and sort entries for deterministic ordering */
        char **entries = NULL;
        int n_entries = 0, cap_entries = 0;
        struct dirent *de;
        while ((de = readdir(dir)) != NULL) {
            if (de->d_name[0] == '.') continue;
            if (n_entries == cap_entries) {
                cap_entries = cap_entries ? cap_entries * 2 : 64;
                char **tmp = realloc(entries, (size_t)cap_entries * sizeof(char *));
                if (!tmp) { fputs("OOM\n", stderr); closedir(dir); backend_destroy(); return 1; }
                entries = tmp;
            }
            entries[n_entries++] = strdup(de->d_name);
        }
        closedir(dir);

        /* Sort alphabetically */
        int cmp_str(const void *a, const void *b) {
            return strcmp(*(const char **)a, *(const char **)b);
        }
        qsort(entries, (size_t)n_entries, sizeof(char *), cmp_str);

        /* Build full paths and probe them all in parallel */
        char **in_paths = malloc((size_t)n_entries * sizeof(char *));
        int  *is_audio  = malloc((size_t)n_entries * sizeof(int));
        if (!in_paths || !is_audio) { fputs("OOM\n", stderr); backend_destroy(); return 1; }
        for (int f = 0; f < n_entries; f++) {
            in_paths[f] = malloc(4096);
            if (!in_paths[f]) { fputs("OOM\n", stderr); backend_destroy(); return 1; }
            snprintf(in_paths[f], 4096, "%s/%s", opt_input_folder, entries[f]);
        }
        if (n_entries > 0)
            ffprobe_batch(in_paths, n_entries, is_audio);

        for (int f = 0; f < n_entries; f++) {
            const char *in_path = in_paths[f];

            if (!is_audio[f]) {
                if (verbose) fprintf(stderr, "Skipping %s (not recognized as audio by ffprobe)\n", entries[f]);
                free(entries[f]);
                free(in_paths[f]);
                continue;
            }

            FILE *out = stdout;
            char out_path[4096];
            out_path[0] = '\0';

            if (opt_output_folder) {
                /* Strip extension, append .txt */
                char base[1024];
                snprintf(base, sizeof(base), "%s", entries[f]);
                char *dot = strrchr(base, '.');
                if (dot) *dot = '\0';
                snprintf(out_path, sizeof(out_path), "%s/%s.txt",
                         opt_output_folder, base);

                if (opt_skip_existing) {
                    struct stat st;
                    if (stat(out_path, &st) == 0) {
                        if (verbose) fprintf(stderr, "Skipping %s (output exists)\n", entries[f]);
                        free(entries[f]);
                        continue;
                    }
                }

                out = fopen(out_path, "w");
                if (!out) {
                    fprintf(stderr, "Cannot open output file: %s\n", out_path);
                    free(entries[f]);
                    free(in_paths[f]);
                    ret = 1;
                    continue;
                }
            }

            if (do_transcribe_file(in_path, out, prompt_ids, n_prompt,
                                   opt_chunk, opt_batch_size,
                                   opt_silence_db, opt_silence_dur,
                                   opt_target_sec) != 0)
                ret = 1;

            if (out != stdout) fclose(out);
            free(entries[f]);
            free(in_paths[f]);
        }
        free(entries);
        free(in_paths);
        free(is_audio);

    } else {
        /* Single-file mode */
        const char *audio_file = argv[i];

        FILE *out = stdout;
        if (opt_output) {
            out = fopen(opt_output, "w");
            if (!out) {
                perror(opt_output);
                backend_destroy();
                return 1;
            }
        }

        ret = do_transcribe_file(audio_file, out, prompt_ids, n_prompt,
                                 opt_chunk, opt_batch_size,
                                 opt_silence_db, opt_silence_dur,
                                 opt_target_sec);

        if (out != stdout) fclose(out);
    }

    backend_destroy();
    return ret;
}
