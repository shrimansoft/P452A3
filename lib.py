from cProfile import label
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

from math import exp, log
from re import I


def laplace(L, n, xb0, xb1, yb0, yb1, maxiter=50):
    dx = L / n
    dy = L / n

    # Solution done for a 2D grid
    u0 = [[0 for i in range(n + 1)] for j in range(n + 1)]  # Initialization
    x = [i * dx for i in range(n + 1)]
    y = [j * dy for j in range(n + 1)]

    # Constant boundary conditions
    for i in range(n + 1):
        for j in range(n + 1):
            if i == 0:
                u0[i][j] = xb0
            elif i == n:
                u0[i][j] = xb1
            elif j == 0:
                u0[i][j] = yb0
            elif j == n:
                u0[i][j] = yb1

    # Jacobi method
    u = u0.copy()
    iter = 0
    while iter <= maxiter:
        for i in range(1, n):
            for j in range(1, n):
                u[i][j] = (
                    1 / 4 * (u0[i][j + 1] + u0[i][j - 1] + u0[i + 1][j] + u0[i - 1][j])
                )
                u0[i][j] = u[i][j]

        iter += 1

    return u, x, y


# --------------------* Pseudorandom number generator *--------------------------


def mlcg(
    seed,  # Starting seed for reproducibility
    a,  #  parameters for the generator
    m,  #  parameters for the generator
    num,  # Number of random values
):
    x = seed
    arrayOfRandomNumber = []
    for i in range(num):
        x = (a * x) % m
        arrayOfRandomNumber.append(x)

    return arrayOfRandomNumber


# ---------------* Generate list of N random points between lims *---------------------
def monteCarlo(funtion, number):
    xRandomNumber = mlcg(234.34, 65, 1, number)

    summation = 0
    for i in xRandomNumber:
        summation += funtion(i)

    total = 1 / float(number) * summation

    return total


def gaussFunction(x):
    return exp(-(x**2))


def p(x, alpha):
    return alpha * exp(-x)


# Without importance sampling
totalWithou = monteCarlo(gaussFunction, 10000)

# With importance sampling
def g(x, alpha=1):
    return gaussFunction(-log(1 - x / alpha)) / p(x, alpha)


# -------------------* function from lib  becouse some problem with lib *-----------------------
def Schroed(
    y,
    r,
    V,
    E,
):
    (psi, phi) = y
    dphidx = [phi, (V - E) * psi]
    return np.array(dphidx)


# --------------------------*  Runge-Kutta RK4 *---------------------


def rk4(
    function,
    psi0,
    x,
    V,
    E,
):
    n = len(x)
    psi = np.array([psi0] * n)
    for i in range(n - 1):
        h = x[i + 1] - x[i]
        k1 = h * function(psi[i], x[i], V[i], E)
        k2 = h * function(psi[i] + 0.5 * k1, x[i] + 0.5 * h, V[i], E)
        k3 = h * function(psi[i] + 0.5 * k2, x[i] + 0.5 * h, V[i], E)
        k4 = h * function(psi[i] + k3, x[i + 1], V[i], E)
        psi[i + 1] = psi[i] + (k1 + 2.0 * (k2 + k3) + k4) / 6.0
    return psi


# --------------------------*  To count the number of nodes *---------------------
def countNodes(waveFunction):
    maxArray = argrelextrema(waveFunction, np.greater)[0]
    minArray = argrelextrema(waveFunction, np.less)[0]
    nodecounter = len(maxArray) + len(minArray)
    return nodecounter


def RefineEnergy(
    Ebot,
    Etop,
    Nodes,
    psi0,
    x,
    V,
):
    tolerance = 1e-12
    ET = Etop
    EB = Ebot
    psi = [1]
    while abs(EB - ET) > tolerance or abs(psi[-1]) > 1e-3:
        initE = (ET + EB) / 2.0
        psi = rk4(Schroed, psi0, x, V, initE)[:, 0]
        nodesIst = len(np.where(np.diff(np.signbit(psi)))[0]) - 1
        if nodesIst > Nodes + 1:
            ET = initE
            continue
        if nodesIst < Nodes - 1:
            EB = initE
            continue
        if nodesIst % 2 == 0:
            if psi[len(psi) - 1] <= 0.0:
                ET = initE
            else:
                EB = initE
        elif nodesIst > 0:
            if psi[len(psi) - 1] <= 0.0:
                EB = initE
            else:
                ET = initE
        elif nodesIst < 0:
            EB = initE
    return (EB, ET)


def shotInfiniPotenWell(EInterval, nodes):
    psi_0 = 0.0
    phi_0 = 1.0
    psiInit = np.array([psi_0, phi_0])
    hMesh = 1.0 / 100.0
    xArrayIpw = np.arange(0.0, 1.0 + hMesh, hMesh)
    VIpw = np.zeros(len(xArrayIpw))
    (EBref, ETref) = RefineEnergy(
        EInterval[0],
        EInterval[1],
        nodes,
        psiInit,
        xArrayIpw,
        VIpw,
    )
    psi = rk4(Schroed, psiInit, xArrayIpw, VIpw, EBref)[:, 0]
    normal = max(psi)
    N = psi * (1 / normal)
    return (EBref, N, xArrayIpw)
