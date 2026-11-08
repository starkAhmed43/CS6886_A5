/*
 * attn_tiled.c -- Q3b: cache and register tiling.
 *
 * Register micro-tiles (MR x NR accumulators, k loop inside) inside JB-wide cache blocks.
 * Driver (main) unchanged; see attn_naive.c for the input/output format.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------------- */
/* attention()                                                                */
/* ------------------------------------------------------------------------- */
#ifndef MR
#define MR 4   /* rows per register tile */
#endif
#ifndef NR
#define NR 16  /* columns per register tile (NR/4 NEON vectors) */
#endif
#ifndef JB
#define JB 256 /* S columns / V rows per cache block */
#endif

/* C[MR x NR] (+)= A[MR x kn] * B[kn x NR]; A has stride lda, B and C have stride ldb, ldc. */
static inline void micro(const float *A, int lda, const float *B, int ldb, float *C, int ldc,
                         int kn, float alpha, int accumulate)
{
    float acc[MR][NR] = {{0}};
    for (int k = 0; k < kn; k++)
        for (int r = 0; r < MR; r++) {
            const float a = A[(size_t)r * lda + k];
            for (int c = 0; c < NR; c++)
                acc[r][c] += a * B[(size_t)k * ldb + c];
        }
    for (int r = 0; r < MR; r++)
        for (int c = 0; c < NR; c++)
            C[(size_t)r * ldc + c] = alpha * acc[r][c] + (accumulate ? C[(size_t)r * ldc + c] : 0.0f);
}

void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    const float scale = 1.0f / sqrtf((float)d);
    float *S = malloc((size_t)N * N * sizeof(float));
    float *Kt = malloc((size_t)N * d * sizeof(float));

    for (int j = 0; j < N; j++)
        for (int k = 0; k < d; k++)
            Kt[(size_t)k * N + j] = K[(size_t)j * d + k];

    /* 1. S = Q K^T * scale: the d x JB panel of Kt stays in cache across all row blocks */
    for (int jb = 0; jb < N; jb += JB)
        for (int i = 0; i < N; i += MR)
            for (int j = jb; j < jb + JB; j += NR)
                micro(Q + (size_t)i * d, d, Kt + j, N, S + (size_t)i * N + j, N, d, scale, 0);

    /* 2. softmax of each row of S */
    for (int i = 0; i < N; i++) {
        float *row = S + (size_t)i * N;
        float max = row[0];
        for (int j = 1; j < N; j++)
            if (row[j] > max)
                max = row[j];
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            row[j] = expf(row[j] - max);
            sum += row[j];
        }
        for (int j = 0; j < N; j++)
            row[j] = row[j] / sum;
    }

    /* 3. O = S V: the JB x d block of V stays in cache across all row blocks */
    for (int jb = 0; jb < N; jb += JB)
        for (int i = 0; i < N; i += MR)
            for (int k = 0; k < d; k += NR)
                micro(S + (size_t)i * N + jb, N, V + (size_t)jb * d + k, d, O + (size_t)i * d + k, d,
                      JB, 1.0f, jb > 0);

    free(Kt);
    free(S);
}

/* ------------------------------------------------------------------------- */
/* Driver: do not change below this line.                                     */
/* ------------------------------------------------------------------------- */
#define REPS 5

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }
    FILE *in = fopen(argv[1], "rb");
    if (!in) {
        perror(argv[1]);
        return 1;
    }
    int32_t hdr[2];
    if (fread(hdr, sizeof hdr, 1, in) != 1) {
        fprintf(stderr, "%s: bad header\n", argv[1]);
        return 1;
    }
    int N = hdr[0], d = hdr[1];
    size_t n = (size_t)N * d;
    float *Q = malloc(n * sizeof(float)), *K = malloc(n * sizeof(float));
    float *V = malloc(n * sizeof(float)), *O = malloc(n * sizeof(float));
    if (!Q || !K || !V || !O || fread(Q, sizeof(float), n, in) != n ||
        fread(K, sizeof(float), n, in) != n || fread(V, sizeof(float), n, in) != n) {
        fprintf(stderr, "%s: could not read Q, K, V\n", argv[1]);
        return 1;
    }
    fclose(in);

    attention(Q, K, V, O, N, d); /* warm-up, not timed */
    double t[REPS];
    for (int r = 0; r < REPS; r++) {
        double t0 = now();
        attention(Q, K, V, O, N, d);
        t[r] = now() - t0;
    }
    qsort(t, REPS, sizeof t[0], cmp_double);
    double median = t[REPS / 2];

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    const char *name = strrchr(argv[0], '/') ? strrchr(argv[0], '/') + 1 : argv[0];
    printf("%-18s N=%-5d d=%-4d threads=%-3d time_ms=%10.3f  GFLOP/s=%8.2f\n", name, N, d,
           threads, median * 1e3, 4.0 * N * (double)N * d / median * 1e-9);

    FILE *out = fopen(argv[2], "wb");
    if (!out || fwrite(hdr, sizeof hdr, 1, out) != 1 || fwrite(O, sizeof(float), n, out) != n) {
        perror(argv[2]);
        return 1;
    }
    fclose(out);
    free(Q);
    free(K);
    free(V);
    free(O);
    return 0;
}
