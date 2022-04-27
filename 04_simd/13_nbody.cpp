#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

void debug(char *s, __m256 x)
{
    float y[8];
    _mm256_store_ps(y, x);
    printf("%s ", s);
    for (int i = 0; i < 8; i++)
    {
        printf("%.18f ", y[i]);
    }
    printf("\n");
}

int main()
{
    const int N = 8;
    float x[N], y[N], m[N], fx[N], fy[N];
    for (int i = 0; i < N; i++)
    {
        x[i] = drand48();
        y[i] = drand48();
        m[i] = drand48();
        fx[i] = fy[i] = 0;
    }
    for (int i = 0; i < N; i++)
    {

        __m256 js = _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0);
        __m256 is = _mm256_set1_ps(i);

        __m256 xs = _mm256_load_ps(x);
        __m256 ys = _mm256_load_ps(y);
        __m256 ms = _mm256_load_ps(m);

        __m256 rx = _mm256_set1_ps(x[i]);
        __m256 ry = _mm256_set1_ps(y[i]);

        __m256 mask = _mm256_cmp_ps(is, js, _CMP_NEQ_OQ);

        // ---------------------- calc invr ----------------------
        rx = _mm256_sub_ps(rx, xs);
        ry = _mm256_sub_ps(ry, ys);
        __m256 invr = _mm256_rsqrt_ps(_mm256_add_ps(_mm256_mul_ps(rx, rx), _mm256_mul_ps(ry, ry)));

        // ---------------------- calc fx ----------------------
        // rx*m[i]/(r*r*r)
        __m256 tmp1 = _mm256_mul_ps(rx, ms);
        tmp1 = _mm256_mul_ps(tmp1, invr);
        tmp1 = _mm256_mul_ps(tmp1, invr);
        tmp1 = _mm256_mul_ps(tmp1, invr);
        tmp1 = _mm256_blendv_ps(_mm256_set1_ps(0), tmp1, mask);

        // reduction
        __m256 tmp2 = _mm256_permute2f128_ps(tmp1, tmp1, 1);
        tmp2 = _mm256_add_ps(tmp2, tmp1);
        tmp2 = _mm256_hadd_ps(tmp2, tmp2);
        tmp2 = _mm256_hadd_ps(tmp2, tmp2);

        float df[8];
        _mm256_store_ps(df, tmp2);
        fx[i] -= df[0];

        // ---------------------- calc fy ----------------------
        // ry*m[i]/(r*r*r)
        tmp1 = _mm256_mul_ps(ry, ms);
        tmp1 = _mm256_mul_ps(tmp1, invr);
        tmp1 = _mm256_mul_ps(tmp1, invr);
        tmp1 = _mm256_mul_ps(tmp1, invr);
        tmp1 = _mm256_blendv_ps(_mm256_set1_ps(0), tmp1, mask);

        // reduction
        tmp2 = _mm256_permute2f128_ps(tmp1, tmp1, 1);
        tmp2 = _mm256_add_ps(tmp2, tmp1);
        tmp2 = _mm256_hadd_ps(tmp2, tmp2);
        tmp2 = _mm256_hadd_ps(tmp2, tmp2);

        df[8];
        _mm256_store_ps(df, tmp2);
        fy[i] -= df[0];

        // ------------------------------------------------------

        printf("%d %g %g\n", i, fx[i], fy[i]);
    }
}

