#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
using std::cout, std::cin, std::cerr, std::endl;
using std::ostream;
using std::pow;
using std::vector, std::tuple;
const size_t BS = 16;
template <class T>
ostream &operator<<(ostream &st, const std::vector<T> &vec)
{
    st << "[ ";
    for (auto x : vec)
    {
        cout << x << ", ";
    }
    st << ']';
    return st;
}

// C++ implementation of some numpy functions
namespace np
{
    vector<float> linspace(float start, float stop, int num)
    {
        vector<float> ret;
        float step = (stop - start) / (float)(num - 1);
        for (int i = 0; i < num; i++)
        {
            ret.push_back(start + step * i);
        }
        return ret;
    }
    vector<vector<float>> zeros(tuple<int, int> dim)
    {
        auto [height, width] = dim;
        return vector<vector<float>>(height, vector<float>(width, 0));
    }
}

void linspace(float *a, float start, float stop, int num)
{
    float step = (stop - start) / (float)(num - 1);
    for (int i = 0; i < num; i++)
    {
        a[i] = start + step * i;
    }
}
void zeros(float *a, tuple<int, int> dim)
{
    auto [height, width] = dim;
    for (int i = 0; i < height * width; i++)
    {
        a[i] = 0.0;
    }
}

#define SQUARE(x) ((x) * (x))
__global__ void calc_b(float *b, float *u, float *v, int ny, int nx, float dx, float dy, float dt, float rho)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i == 0 || j == 0 || i >= nx - 1 || j >= ny - 1)
    {
        return;
    }
    b[j * nx + i] = rho * (1 / dt *
                               ((u[j * nx + i + 1] - u[j * nx + i - 1]) / (2 * dx) + (v[(j + 1) * nx + i] - v[(j - 1) * nx + i]) / (2 * dy)) -
                           SQUARE((u[j * nx + i + 1] - u[j * nx + i - 1]) / (2 * dx)) - 2 * ((u[(j + 1) * nx + i] - u[(j - 1) * nx + i]) / (2 * dy) * (v[j * nx + i + 1] - v[j * nx + i - 1]) / (2 * dx)) - SQUARE((v[(j + 1) * nx + i] - v[(j - 1) * nx + i]) / (2 * dy)));
}

__global__ void calc_p(float *p, float *pn, float *b, int ny, int nx, float dx, float dy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i == 0 || j == 0 || i >= nx - 1 || j >= ny - 1)
    {
        return;
    }
    p[j * nx + i] = (SQUARE(dy) * (pn[j * nx + i + 1] + pn[j * nx + i - 1]) +
                     SQUARE(dx) * (pn[(j + 1) * nx + i] + pn[(j - 1) * nx + i]) -
                     b[j * nx + i] * SQUARE(dx) * SQUARE(dy)) /
                    (2 * (SQUARE(dx) + SQUARE(dy)));
}

__global__ void calc_u_v(float *u, float *v, float *un, float *vn, float *p, int ny, int nx, float dx, float dy, float dt, float rho, float nu)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i == 0 || j == 0 || i >= nx - 1 || j >= ny - 1)
    {
        return;
    }
    u[j * nx + i] = un[j * nx + i] - un[j * nx + i] * dt / dx * (un[j * nx + i] - un[j * nx + i - 1]) - un[j * nx + i] * dt / dy * (un[j * nx + i] - un[(j - 1) * nx + i]) - dt / (2 * rho * dx) * (p[j * nx + i + 1] - p[j * nx + i - 1]) + nu * dt / SQUARE(dx) * (un[j * nx + i + 1] - 2 * un[j * nx + i] + un[j * nx + i - 1]) + nu * dt / SQUARE(dy) * (un[(j + 1) * nx + i] - 2 * un[j * nx + i] + un[(j - 1) * nx + i]);
    v[j * nx + i] = vn[j * nx + i] - vn[j * nx + i] * dt / dx * (vn[j * nx + i] - vn[j * nx + i - 1]) - vn[j * nx + i] * dt / dy * (vn[j * nx + i] - vn[(j - 1) * nx + i]) - dt / (2 * rho * dx) * (p[(j + 1) * nx + i] - p[(j - 1) * nx + i]) + nu * dt / SQUARE(dx) * (vn[j * nx + i + 1] - 2 * vn[j * nx + i] + vn[j * nx + i - 1]) + nu * dt / SQUARE(dy) * (vn[(j + 1) * nx + i] - 2 * vn[j * nx + i] + vn[(j - 1) * nx + i]);
}

__global__ void copy(float *dst, float *src, int ny, int nx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    dst[j * nx + i] = src[j * nx + i];
}

void print(float *a, int ny, int nx)
{
    cout << "[ ";
    for (int j = 0; j < ny; j++)
    {
        cout << "[ ";
        for (int i = 0; i < nx; i++)
        {
            cout << a[j * nx + i] << ", ";
        }
        cout << "], ";
    }
    cout << "]" << endl;
}

int main()
{
    int nx = 41;
    int ny = 41;
    int nt = 500;
    int nit = 50;
    float dx = 2.0 / (float)(nx - 1);
    float dy = 2.0 / (float)(ny - 1);
    float dt = .01;
    float rho = 1;
    float nu = .02;
    float *x;
    float *y;
    cudaMallocManaged(&x, nx * sizeof(float));
    cudaMallocManaged(&y, ny * sizeof(float));
    linspace(x, 0, 2, nx);
    linspace(y, 0, 2, ny);

    float *u, *v, *p, *b, *pn, *vn, *un;
    cudaMallocManaged(&u, ny * nx * sizeof(float));
    cudaMallocManaged(&v, ny * nx * sizeof(float));
    cudaMallocManaged(&p, ny * nx * sizeof(float));
    cudaMallocManaged(&b, ny * nx * sizeof(float));
    cudaMallocManaged(&un, ny * nx * sizeof(float));
    cudaMallocManaged(&vn, ny * nx * sizeof(float));
    cudaMallocManaged(&pn, ny * nx * sizeof(float));

    zeros(u, {ny, nx});
    zeros(v, {ny, nx});
    zeros(p, {ny, nx});
    zeros(b, {ny, nx});
    dim3 grid = dim3((ny + BS - 1) / BS, ((nx + BS - 1) / BS), 1);
    dim3 block = dim3(BS, BS, 1);

    for (int n = 0; n < nt; n++)
    {
        calc_b<<<grid, block>>>(b, u, v, ny, nx, dx, dy, dt, rho);
        cudaDeviceSynchronize();
        for (int it = 0; it < nit; it++)
        {
            memcpy(pn, p, ny * nx * sizeof(float));
            calc_p<<<grid, block>>>(p, pn, b, ny, nx, dx, dy);
            cudaDeviceSynchronize();

            for (int j = 0; j < ny; j++)
            {
                p[j * nx + nx - 1] = p[j * nx + nx - 2];
                p[j * nx + 0] = p[j * nx + 1];
            }
            for (int i = 0; i < nx; i++)
            {
                p[ny - 1 * nx + i] = 0;
                p[0 * nx + i] = p[1 * nx + i];
            }
        }
        memcpy(un, u, ny * nx * sizeof(float));
        memcpy(vn, v, ny * nx * sizeof(float));
        cudaDeviceSynchronize();
        calc_u_v<<<grid, block>>>(u, v, un, vn, p, ny, nx, dx, dy, dt, rho, nu);
        cudaDeviceSynchronize();
        for (int j = 0; j < ny; j++)
        {
            u[j * nx + 0] = 0;
            u[j * nx + nx - 1] = 0;
            v[j * nx + 0] = 0;
            v[j * nx + nx - 1] = 0;
        }
        for (int i = 0; i < nx; i++)
        {
            u[0 * nx + i] = 0;
            u[(ny - 1) * nx + i] = 1;
            v[0 * nx + i] = 0;
            v[(ny - 1) * nx + i] = 0;
        }
        print(p, ny, nx);
        print(u, ny, nx);
        print(v, ny, nx);
    }
    cudaFree(x);
    cudaFree(y);
    cudaFree(u);
    cudaFree(v);
    cudaFree(p);
    cudaFree(b);
    cudaFree(un);
    cudaFree(vn);
    cudaFree(pn);
}
