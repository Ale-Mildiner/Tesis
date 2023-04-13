#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
p = 1
r = 1
pr = p+r
q = 1
N = 10
u0 = N
v0 = 0

def derivadas(X, t, p, r, q):
    u, v = X
    der_u = -(p+r)*u
    der_v = -q*v + r*u
    return np.array([der_u, der_v])


t = np.linspace(0.1, 10, 1000)
inicial = [u0, v0]

res = integrate.odeint(derivadas, inicial, t, args=(p,r,q))
u,v = res.T

u_exacta = N*(np.e**(-(p+r)*t))
v_excta = N*r*(np.e**(-q*t) - np.e**(-(p+r)*t))/((p+r)-q)

# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 1000
plt.figure()
plt.plot(t, u_exacta, label = 'u exacta')
plt.plot(t, v_excta, label = 'v exacta')
plt.plot(t, u, label = 'u numerica')
plt.plot(t, v, label = 'v numerica')
plt.plot(t,u+v, label = 'Atencion S(t)', linestyle = 'dashed', color ='k')
plt.xlabel('tiempo')
plt.ylabel('Soluciones')
plt.legend()
plt.show()

#%%

K = 1
rp = 1
rc = 1
t_peak = 5

def L(t, K, rp, rc, t_peak):
    arr = 2 * K * rp * np.e**(-rp*(t-t_peak))
    aba = rc * ( 1 + np.e**(-rp*(t-t_peak)))**2
    return arr/aba

t = np.linspace(0,100,100)
plt.figure()
plt.plot(t, L(t, K, rp, rc, t_peak))
plt.show()

import numpy as np
import pandas as pd

print(np.array([1,2,3]))
print(pd.DataFrame({'a': [1, 2]}))
