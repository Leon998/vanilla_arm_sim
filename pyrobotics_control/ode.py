import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

plt.style.use('seaborn-poster')


F = lambda t, y: 5 - 2*y*y*y


sol = solve_ivp(F, [0, 5], [1])

plt.figure(figsize = (12, 8))
plt.plot(sol.t, sol.y[0])
plt.xlabel('t')
plt.ylabel('y')
plt.show()