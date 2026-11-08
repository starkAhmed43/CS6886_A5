/*
 * attn_best.c -- Q3d: your fastest single-threaded version. Use any technique you like.
 *
 * Keep the driver (main) below unchanged. See attn_naive.c for the input and
 * output format.
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
/* attention(): flash-style fused attention with NEON micro-kernels.          */
/* ------------------------------------------------------------------------- */
#include <arm_neon.h>

/* RB query rows per block, KB keys per block (chosen by sweep). The RB x KB
 * score tile (16 KB) stays in L1 and K blocks stream from L2; the N x N
 * matrix is never stored. */
#define RB 16
#define KB 256
#define DMAX 128

/* 2^x for x <= 0 on 4 lanes. Round x = n + f with f in [-0.5, 0.5], degree-5
 * polynomial for 2^f (rel. error ~2e-6), and add n to the exponent bits.
 * The clamp keeps 2^n finite; results below 2^-126 flush to ~0. */
static inline float32x4_t exp2_neon(float32x4_t x)
{
    x = vmaxq_f32(x, vdupq_n_f32(-126.0f));
    float32x4_t n = vrndnq_f32(x);
    float32x4_t f = vsubq_f32(x, n);
    float32x4_t p = vdupq_n_f32(0.00133336f);
    p = vfmaq_f32(vdupq_n_f32(0.00961813f), p, f);
    p = vfmaq_f32(vdupq_n_f32(0.0555041f), p, f);
    p = vfmaq_f32(vdupq_n_f32(0.2402265f), p, f);
    p = vfmaq_f32(vdupq_n_f32(0.6931472f), p, f);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, f);
    int32x4_t e = vshlq_n_s32(vcvtq_s32_f32(n), 23);
    return vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(p), e));
}

/* S[4 x 16] = Qp^T Kb: Qp is d x 4 (k-major, so one load gives 4 rows'
 * values for lane-broadcast FMAs), Kb is a packed d x 16 panel of K^T. 16 accumulators; 16 FMAs per 5 loads. */
static inline void qk_tile(const float *Qp, const float *Kb, float *S, int d)
{
    float32x4_t a[4][4];
    for (int i = 0; i < 4; i++)
        for (int v = 0; v < 4; v++)
            a[i][v] = vdupq_n_f32(0.0f);
    for (int k = 0; k < d; k++) {
        float32x4_t q = vld1q_f32(Qp + 4 * k);
        const float *kr = Kb + 16 * k;
        for (int v = 0; v < 4; v++) {
            float32x4_t b = vld1q_f32(kr + 4 * v);
            a[0][v] = vfmaq_laneq_f32(a[0][v], b, q, 0);
            a[1][v] = vfmaq_laneq_f32(a[1][v], b, q, 1);
            a[2][v] = vfmaq_laneq_f32(a[2][v], b, q, 2);
            a[3][v] = vfmaq_laneq_f32(a[3][v], b, q, 3);
        }
    }
    for (int i = 0; i < 4; i++)
        for (int v = 0; v < 4; v++)
            vst1q_f32(S + i * KB + 4 * v, a[i][v]);
}

/* Ob[4 x 16] += P[4 x KB] Vb[KB x 16] (Vb has row stride d). Accumulators
 * stay in registers over the whole key block; P values come by lane. */
static inline void pv_tile(const float *P, const float *Vb, float *Ob, int d)
{
    float32x4_t a[4][4];
    for (int i = 0; i < 4; i++)
        for (int v = 0; v < 4; v++)
            a[i][v] = vld1q_f32(Ob + (size_t)i * d + 4 * v);
    for (int j = 0; j < KB; j += 4) {
        float32x4_t p0 = vld1q_f32(P + j), p1 = vld1q_f32(P + KB + j);
        float32x4_t p2 = vld1q_f32(P + 2 * KB + j), p3 = vld1q_f32(P + 3 * KB + j);
#define STEP(L)                                                      \
    for (int v = 0; v < 4; v++) {                                    \
        float32x4_t b = vld1q_f32(Vb + (size_t)(j + L) * d + 4 * v); \
        a[0][v] = vfmaq_laneq_f32(a[0][v], b, p0, L);                \
        a[1][v] = vfmaq_laneq_f32(a[1][v], b, p1, L);                \
        a[2][v] = vfmaq_laneq_f32(a[2][v], b, p2, L);                \
        a[3][v] = vfmaq_laneq_f32(a[3][v], b, p3, L);                \
    }
        STEP(0) STEP(1) STEP(2) STEP(3)
#undef STEP
    }
    for (int i = 0; i < 4; i++)
        for (int v = 0; v < 4; v++)
            vst1q_f32(Ob + (size_t)i * d + 4 * v, a[i][v]);
}

/* Rows [r0, r0+RB) of O. All scratch is local, so calls for different row
 * blocks are independent (thread-safe; Kp and V are read-only). */
static void attn_rows(const float *Q, const float *Kp, const float *V, float *O, int N, int d,
                      int r0)
{
    float Qp[(RB / 4) * DMAX * 4] __attribute__((aligned(64)));
    float S[RB * KB] __attribute__((aligned(64)));
    float Ob[RB * DMAX] __attribute__((aligned(64)));
    float m[RB], l[RB];

    /* Fold 1/sqrt(d) * log2(e) into Q so softmax uses exp2 with no extra multiply. */
    const float c = 1.4426950408889634f / sqrtf((float)d);
    for (int g = 0; g < RB / 4; g++)
        for (int i = 0; i < 4; i++)
            for (int k = 0; k < d; k++)
                Qp[(g * d + k) * 4 + i] = Q[(size_t)(r0 + 4 * g + i) * d + k] * c;
    for (int i = 0; i < RB; i++) {
        m[i] = -1e30f;
        l[i] = 0.0f;
    }
    memset(Ob, 0, sizeof(float) * RB * d);

    for (int kb = 0; kb < N / KB; kb++) {
        const float *Kb = Kp + (size_t)kb * d * KB;
        for (int g = 0; g < RB / 4; g++)
            for (int j = 0; j < KB; j += 16)
                qk_tile(Qp + g * d * 4, Kb + (size_t)j * d, S + 4 * g * KB + j, d);

        /* Online softmax: new running max, rescale old O and l, exponentiate
         * the tile in place and accumulate the row sums. */
        for (int i = 0; i < RB; i++) {
            float *s = S + i * KB;
            float32x4_t mx = vld1q_f32(s);
            for (int j = 4; j < KB; j += 4)
                mx = vmaxq_f32(mx, vld1q_f32(s + j));
            float mn = fmaxf(m[i], vmaxvq_f32(mx));
            float alpha = vgetq_lane_f32(exp2_neon(vdupq_n_f32(m[i] - mn)), 0);
            m[i] = mn;
            float32x4_t vm = vdupq_n_f32(mn), va = vdupq_n_f32(alpha);
            float32x4_t sum = vdupq_n_f32(0.0f);
            for (int j = 0; j < KB; j += 4) {
                float32x4_t e = exp2_neon(vsubq_f32(vld1q_f32(s + j), vm));
                vst1q_f32(s + j, e);
                sum = vaddq_f32(sum, e);
            }
            l[i] = l[i] * alpha + vaddvq_f32(sum);
            for (int k = 0; k < d; k += 4)
                vst1q_f32(Ob + i * d + k, vmulq_f32(vld1q_f32(Ob + i * d + k), va));
        }

        const float *Vb = V + (size_t)kb * KB * d;
        for (int g = 0; g < RB / 4; g++)
            for (int c0 = 0; c0 < d; c0 += 16)
                pv_tile(S + 4 * g * KB, Vb + c0, Ob + 4 * g * d + c0, d);
    }

    for (int i = 0; i < RB; i++) {
        float32x4_t inv = vdupq_n_f32(1.0f / l[i]); /* one division per row */
        for (int k = 0; k < d; k += 4)
            vst1q_f32(O + (size_t)(r0 + i) * d + k, vmulq_f32(vld1q_f32(Ob + i * d + k), inv));
    }
}

void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    /* Pack K once as [N/16][d][16]: K^T in panels of 16 keys, so each QK tile
     * streams one contiguous d x 16 panel. Shared and read-only afterwards. */
    float *Kp = aligned_alloc(64, (size_t)N * d * sizeof(float));
    for (int j = 0; j < N; j++)
        for (int k = 0; k < d; k++)
            Kp[((size_t)(j / 16) * d + k) * 16 + j % 16] = K[(size_t)j * d + k];

    for (int r0 = 0; r0 < N; r0 += RB) /* independent row blocks: parallelise this loop */
        attn_rows(Q, Kp, V, O, N, d, r0);
    free(Kp);
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
