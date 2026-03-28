/* tests/compare_floats.c
 * Element-wise comparison of two binary float32 files.
 * Usage: compare_floats <ref.bin> <cmp.bin> <abs_tol>
 * Exit 0 on pass, 1 on fail, 2 on error.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static float *read_bin(const char *path, size_t *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(2); }
    if (fseek(f, 0, SEEK_END) != 0) { perror("fseek"); exit(2); }
    long bytes = ftell(f);
    if (bytes < 0) { perror("ftell"); exit(2); }
    rewind(f);
    *n_out = (size_t)bytes / sizeof(float);
    float *buf = (float *)malloc((size_t)bytes + 1);
    if (!buf) { fputs("OOM\n", stderr); exit(2); }
    if (fread(buf, sizeof(float), *n_out, f) != *n_out) { perror("fread"); exit(2); }
    fclose(f);
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: compare_floats <ref.bin> <cmp.bin> <abs_tol>\n");
        return 2;
    }
    const char *ref_path = argv[1];
    const char *cmp_path = argv[2];
    float abs_tol = (float)atof(argv[3]);

    size_t n_ref, n_cmp;
    float *ref = read_bin(ref_path, &n_ref);
    float *cmp = read_bin(cmp_path, &n_cmp);

    if (n_ref != n_cmp) {
        fprintf(stderr, "FAIL: element count mismatch: %s=%zu  %s=%zu\n",
                ref_path, n_ref, cmp_path, n_cmp);
        return 1;
    }

    float max_abs = 0.0f, max_rel = 0.0f, sum_abs = 0.0f;
    size_t n_fail = 0;
    size_t first_fail = (size_t)-1;

    for (size_t i = 0; i < n_ref; i++) {
        float a = ref[i], b = cmp[i];
        float diff = fabsf(a - b);
        float denom = fabsf(a);
        float rel = (denom > 1e-6f) ? diff / denom : diff;

        if (diff > max_abs) max_abs = diff;
        if (rel  > max_rel) max_rel = rel;
        sum_abs += diff;

        if (diff > abs_tol) {
            if (first_fail == (size_t)-1) first_fail = i;
            n_fail++;
        }
    }

    printf("Elements   : %zu\n", n_ref);
    printf("Tolerance  : %.3e (abs)\n", (double)abs_tol);
    printf("Max abs    : %.6e\n", (double)max_abs);
    printf("Max rel    : %.6e\n", (double)max_rel);
    printf("Mean abs   : %.6e\n", (double)(sum_abs / (float)n_ref));

    if (n_fail > 0) {
        printf("Failures   : %zu / %zu\n", n_fail, n_ref);
        printf("First fail : index %zu  ref=%.8g  cmp=%.8g  diff=%.3e\n",
               first_fail,
               (double)ref[first_fail],
               (double)cmp[first_fail],
               (double)fabsf(ref[first_fail] - cmp[first_fail]));
        puts("FAIL");
        return 1;
    }
    puts("PASS");
    return 0;
}
