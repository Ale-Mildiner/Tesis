import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
#from scipy.integrate import solve_dde
from ddeint import ddeint

def sophy_model(X, t, p, q, r, n, tau, k):
    u, v = X
    v_past = v(t-tau)
    m = np.heaviside(t-tau, 1)
    dot_u = -p*u + m*v
    dot_v = -q*v + r*u + k*v*v_past
    return np.array([dot_u, dot_v])


t = np.linspace(0.1, 10, 1000)
p = r = q = n = k = 1
tau = 10
u0 = 10
v0 = 0
inicial = [u0, v0]

#res = integrate.odeint(sophy_model, inicial, t, args=(p,q,r,n,tau, k))
#res = solve_dde(sophy_model, inicial, t, args=(p,q,r,n,tau, k))
#u, v = res.T

plt.figure()
plt.plot(t, u, label = 'u')
plt.plot(t, v, label = 'v', lable = v)
plt.legend()
plt.show()