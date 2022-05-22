from lib import *

# Without importance sampling
totalWithou = monteCarlo(gaussFunction, 10000)

# With importance sampling
def g(x, alpha=1):
    return gaussFunction(-log(1 - x / alpha)) / p(x, alpha)


totalWith = monteCarlo(g, 10000)

print("Without importance sampling: ", totalWithou)
print("With importance sampling: ", totalWith)
