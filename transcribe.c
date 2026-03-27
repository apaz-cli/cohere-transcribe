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
#include <pthread.h>
#include <time.h>

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
                              + 2*n_cakv + n_hdec + n_mask + work_size;

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
    precompute_ca_kv(enc_h, T_enc, ca_k, ca_v);

    int tokens[MAX_SEQ], S = 0;
    for (int i = 0; i < n_prompt; i++) tokens[S++] = prompt_ids[i];
    int prompt_end = S;

    for (int step = 0; step < MAX_SEQ - S; step++) {
        int next_id = decoder_step(tokens, S, T_enc, ca_k, ca_v, h_dec, mask, work);
        tokens[S++] = next_id;
        if (next_id == TOK_EOS) break;
    }

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
} ChunkWork;

typedef struct {
    ChunkWork      *jobs;
    int             n_jobs;
    int             next;      /* index of next unclaimed job */
    int             completed;
    pthread_mutex_t mu;
} Queue;

static void *worker(void *arg) {
    Queue *q = arg;
    for (;;) {
        pthread_mutex_lock(&q->mu);
        int i = q->next++;
        pthread_mutex_unlock(&q->mu);
        if (i >= q->n_jobs) break;
        ChunkWork *w = &q->jobs[i];
        FILE *f = open_memstream(&w->text, &w->text_len);
        transcribe_chunk(w->samples, w->n_samples, w->prompt_ids, w->n_prompt, f, &w->n_tokens);
        fclose(f);
        pthread_mutex_lock(&q->mu);
        int done = ++q->completed;
        pthread_mutex_unlock(&q->mu);
        if (verbose) fprintf(stderr, "[%d/%d chunk=%.1fs tokens=%d]\n",
                             done, q->n_jobs, (float)w->n_samples / SR, w->n_tokens);
    }
    return NULL;
}

static void run_chunks_parallel(const float *samples,
                                const int *splits, int n_splits, int n_samples,
                                const int *prompt_ids, int n_prompt) {
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
            n_jobs++;
        }
        start = end;
    }

    long nthreads = sysconf(_SC_NPROCESSORS_ONLN);
    if (nthreads < 1) nthreads = 1;
    if (nthreads > n_jobs) nthreads = n_jobs;
    if (verbose) fprintf(stderr, "  running %d chunks on %ld thread(s)\n", n_jobs, nthreads);
    pthread_t *tids = calloc((size_t)nthreads, sizeof(pthread_t));
    if (!tids) { fputs("OOM\n", stderr); exit(1); }

    Queue q = { jobs, n_jobs, 0, 0, PTHREAD_MUTEX_INITIALIZER };
    for (int i = 0; i < (int)nthreads; i++)
        pthread_create(&tids[i], NULL, worker, &q);
    for (int i = 0; i < (int)nthreads; i++)
        pthread_join(tids[i], NULL);

    for (int i = 0; i < n_jobs; i++) {
        fwrite(jobs[i].text, 1, jobs[i].text_len, stdout);
        free(jobs[i].text);
    }

    pthread_mutex_destroy(&q.mu);
    free(tids);
    free(jobs);
}

/* ============================================================
 * main
 * ============================================================ */
int main(int argc, char **argv) {
    const char *lang         = "en";
    int opt_pnc              = 1;
    int opt_itn              = 0;
    int opt_timestamps       = 0;
    int opt_diarize          = 0;
    int opt_chunk            = 0;
    float opt_silence_db     = NAN;   /* NAN = auto-estimate from recording */
    float opt_silence_dur    = 0.2f;
    float opt_target_sec     = 20.0f;

    int i;
    for (i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "--lang")        == 0 && i+1 < argc) lang = argv[++i];
        else if (strcmp(argv[i], "--no-pnc")      == 0) opt_pnc = 0;
        else if (strcmp(argv[i], "--itn")         == 0) opt_itn = 1;
        else if (strcmp(argv[i], "--timestamps")  == 0) opt_timestamps = 1;
        else if (strcmp(argv[i], "--diarize")     == 0) opt_diarize = 1;
        else if (strcmp(argv[i], "--chunk")       == 0) opt_chunk = 1;
        else if (strcmp(argv[i], "--silence-db")  == 0 && i+1 < argc) opt_silence_db  = atof(argv[++i]);
        else if (strcmp(argv[i], "--silence-dur") == 0 && i+1 < argc) opt_silence_dur = atof(argv[++i]);
        else if (strcmp(argv[i], "--target-sec")  == 0 && i+1 < argc) opt_target_sec  = atof(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) verbose = 1;
        else if (argv[i][0] == '-') {
            fprintf(stderr, "Unknown flag: %s\n", argv[i]);
            fprintf(stderr, "Usage: %s [-v] [--lang CODE] [--no-pnc] [--itn] [--timestamps] [--diarize]\n"
                            "       [--chunk [--silence-db DB] [--silence-dur SEC] [--target-sec SEC]] audio_file\n", argv[0]);
            return 1;
        } else break;
    }
    if (i >= argc) {
        fprintf(stderr, "Usage: %s [-v] [--lang CODE] [--no-pnc] [--itn] [--timestamps] [--diarize]\n"
                        "       [--chunk [--silence-db DB] [--silence-dur SEC] [--target-sec SEC]] audio_file\n", argv[0]);
        return 1;
    }
    const char *audio_file = argv[i];

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
    backend_init(0);
#if defined(BACKEND_CUDA) && !defined(EMBEDDED_MODEL)
    backend_weights_upload_blob(sf_data, sf_data_bytes);
    patch_weights_to_device();
#endif

    int n_samples;
    if (verbose) fprintf(stderr, "Loading %s ...\n", audio_file);
    float *samples = load_audio(audio_file, &n_samples);
    if (verbose) fprintf(stderr, "  Loaded %.2fs of audio\n", (float)n_samples / SR);

    /* Single-pass MHSA is O(T²) in memory; beyond ~2 min the arena exceeds tens of GB. */
    if (!opt_chunk && n_samples > SR * 120) {
        fprintf(stderr,
            "Error: %.0fs of audio is too long for single-pass mode.\n"
            "Use --chunk to split on silence and transcribe in segments.\n",
            (float)n_samples / SR);
        free(samples);
        backend_destroy();
        return 1;
    }

    init_hann();
    init_mel_filterbank();

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    if (!opt_chunk) {
        if (verbose) fprintf(stderr, "Transcribing %.2fs single-pass ...\n", (float)n_samples / SR);
        int n_tokens = 0;
        transcribe_chunk(samples, n_samples, prompt_ids, n_prompt, stdout, &n_tokens);
        if (verbose) fprintf(stderr, "  %d tokens\n", n_tokens);
    } else {
        int n_frames = n_samples / HOP_LEN;
        float *frame_rms = malloc(n_frames * sizeof(float));
        if (!frame_rms) { fputs("OOM\n", stderr); return 1; }
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

        run_chunks_parallel(samples, splits, n_splits, n_samples, prompt_ids, n_prompt);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (verbose) {
        double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        fprintf(stderr, "Done in %.2fs (%.1fx realtime)\n",
                elapsed, (float)n_samples / SR / elapsed);
    }

    free(samples);
    backend_destroy();
    return 0;
}
