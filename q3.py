import matplotlib.pyplot as plt


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


# Solve for given initial and boundary conditions
u, x, y = laplace(1.0, 50, 1.0, 0.0, 0.0, 0.0, 1000)
plt.imshow(u)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.show()
