import numpy as np
import matplotlib.pyplot as plt


nx = 41
ny = 41
nt = 500
nit = 50
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
dt = .01
rho = 1
nu = .02

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))
X, Y = np.meshgrid(x, y)

for n in range(nt):
    p = np.array(eval(input()))
    u = np.array(eval(input()))
    v = np.array(eval(input()))
    plt.contourf(X, Y, p, alpha=0.5, cmap=plt.cm.coolwarm)
    plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
    plt.pause(.01)
    plt.clf()
print('complete')
plt.show()
