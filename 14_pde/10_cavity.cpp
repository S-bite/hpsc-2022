#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
using std::cout, std::cin, std::cerr, std::endl;
using std::ostream;
using std::pow;
using std::vector, std::tuple;

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
    vector<double> linspace(double start, double stop, int num)
    {
        vector<double> ret;
        double step = (stop - start) / (double)(num - 1);
        for (int i = 0; i < num; i++)
        {
            ret.push_back(start + step * i);
        }
        return ret;
    }
    vector<vector<double>> zeros(tuple<int, int> dim)
    {
        auto [height, width] = dim;
        return vector<vector<double>>(height, vector<double>(width, 0));
    }
}

int main()
{
    int nx = 41;
    int ny = 41;
    int nt = 500;
    int nit = 50;
    double dx = 2.0 / (double)(nx - 1);
    double dy = 2.0 / (double)(ny - 1);
    double dt = .01;
    double rho = 1;
    double nu = .02;
    vector<double> x = np::linspace(0, 2, nx);
    vector<double> y = np::linspace(0, 2, nx);
    auto u = np::zeros({ny, nx});
    auto v = np::zeros({ny, nx});
    auto p = np::zeros({ny, nx});
    auto b = np::zeros({ny, nx});

    for (int n = 0; n < nt; n++)
    {
        for (int j = 1; j < ny - 1; j++)
        {
            for (int i = 1; i < nx - 1; i++)
            {
                b[j][i] = rho * (1 / dt *
                                     ((u[j][i + 1] - u[j][i - 1]) / (2 * dx) + (v[j + 1][i] - v[j - 1][i]) / (2 * dy)) -
                                 pow((u[j][i + 1] - u[j][i - 1]) / (2 * dx), 2) - 2 * ((u[j + 1][i] - u[j - 1][i]) / (2 * dy) * (v[j][i + 1] - v[j][i - 1]) / (2 * dx)) - pow((v[j + 1][i] - v[j - 1][i]) / (2 * dy), 2));
            }
        }
        for (int it = 0; it < nit; it++)
        {
            auto pn = p;
            for (int j = 1; j < ny - 1; j++)
            {
                for (int i = 1; i < nx - 1; i++)
                {
                    p[j][i] = (pow(dy, 2) * (pn[j][i + 1] + pn[j][i - 1]) +
                               pow(dx, 2) * (pn[j + 1][i] + pn[j - 1][i]) -
                               b[j][i] * pow(dx, 2) * pow(dy, 2)) /
                              (2 * (pow(dx, 2) + pow(dy, 2)));
                }
            }
            for (int j = 0; j < ny; j++)
            {
                p[j][nx - 1] = p[j][nx - 2];
                p[j][0] = p[j][1];
            }
            for (int i = 0; i < nx; i++)
            {
                p[ny - 1][i] = 0;
                p[0][i] = p[1][i];
            }
        }
        auto un = u;
        auto vn = v;
        for (int j = 1; j < ny - 1; j++)
        {
            for (int i = 1; i < nx - 1; i++)
            {
                u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1]) - un[j][i] * dt / dy * (un[j][i] - un[j - 1][i]) - dt / (2 * rho * dx) * (p[j][i + 1] - p[j][i - 1]) + nu * dt / pow(dx, 2) * (un[j][i + 1] - 2 * un[j][i] + un[j][i - 1]) + nu * dt / pow(dy, 2) * (un[j + 1][i] - 2 * un[j][i] + un[j - 1][i]);
                v[j][i] = vn[j][i] - vn[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1]) - vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i]) - dt / (2 * rho * dx) * (p[j + 1][i] - p[j - 1][i]) + nu * dt / pow(dx, 2) * (vn[j][i + 1] - 2 * vn[j][i] + vn[j][i - 1]) + nu * dt / pow(dy, 2) * (vn[j + 1][i] - 2 * vn[j][i] + vn[j - 1][i]);
            }
        }
        for (int j = 0; j < ny; j++)
        {
            u[j][0] = 0;
            u[j][nx - 1] = 0;
            v[j][0] = 0;
            v[j][nx - 1] = 0;
        }
        for (int i = 0; i < nx; i++)
        {
            u[0][i] = 0;
            u[ny - 1][i] = 1;
            v[0][i] = 0;
            v[ny - 1][i] = 0;
        }
        cout << p << endl;
        cout << u << endl;
        cout << v << endl;
    }
}
