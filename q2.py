from lib import *

# ----------------------* Start *-------------------------

EIpw = [1.0, 500.0]
nodes_arr = [0, 1]
L = 0.0
N = 1.0


figure = plt.figure()
plt.xlabel("x")
plt.ylabel(r"$\psi$")

(Energy, pisIpw, xIpw) = shotInfiniPotenWell(EIpw, 0)
print("Ground state's energy : ", Energy)
plt.plot(
    xIpw,
    pisIpw,
    "g",
    label="Eigenstate : % s" % (0,) + " \nGround state energy" + str(Energy),
)
(Energy, pisIpw, xIpw) = shotInfiniPotenWell(EIpw, 2)
print("First excited state's energy: ", Energy)
plt.plot(
    xIpw,
    pisIpw,
    "orange",
    label="Eigenstate : % s" % (1,) + " \nfirst exited state energy" + str(Energy),
)
plt.legend()
plt.show()
