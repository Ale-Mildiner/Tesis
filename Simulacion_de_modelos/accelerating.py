import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import random


def acceleratin(X, t, rck, c, a, r):
    x1, x2, y1, y2 = X
    
    der_x1 = r*x1 * (1 - rck*y1 - c*x2)
    der_y1 = x1 - a*y1
    der_x2 = r*x2 * (1 - rck*y2 - c*x1)
    der_y2 = x2 - a*y2

    return np.array([der_x1, der_x2, der_y1, der_y2])


def accelerating_grl(X, t, rck, c, a, r, N):
    '''
    X: array con N variables con x e y intercalados
    t: tiempo
    rck: rc/k
    c: c
    a: alpha
    r: array de ri correpsondiente a cada r
    N: número de variables debe ser par (se puede solucionar)
    '''
    x = []
    y = []
    for i in range(N):
        if i%2 == 0:
            x.append(X[i])
        else:
            y.append(X[i])

    x_y_punto = []
    for i in range(len(x)):
        x_punto_i = r*x[i] * (1 - rck*y[i] - c*(sum(x)-x[i]))
        y_punto_i = x[i] - a*y[i]
        
        x_y_punto.append(x_punto_i)
        x_y_punto.append(y_punto_i)

    return np.array(x_y_punto)

c = 2.4
rck =  1
a = 0.005
r = rck = 0.2

t = np.linspace(0.1, 100, 1000)
x1_i = y1_i = random.random()
x2_i = y2_i = random.random()
inicial = [x1_i, x2_i, y1_i, y2_i]
res = integrate.odeint(acceleratin, inicial, t, args=(rck, c, a, r))
x1, x2, y1, y2 = res.T

inicial_2 = [x1_i, y1_i, x2_i, y2_i]

N = 4
ini = []
for i in range(N):
    x_i = random.random()
    print(x_i)
    ini.append(x_i)
    ini.append(x_i)

res_grl = integrate.odeint(accelerating_grl, ini, t, args=(rck, c, a, r, N*2))
xy = res_grl.T


# plt.figure()
# plt.plot(t, x1, label = 'x1', linestyle = 'dashed')
# plt.plot(t, x2, label = 'x2', linestyle = 'dashed')
# plt.show()

for i in range(N*2):
    if i%2 == 0:
        plt.plot(t, xy[i], label = 'x'+str((i+1)//2))

plt.xlabel('tiempo')
plt.ylabel(r'$L_i (t)$')
plt.legend()
plt.show()