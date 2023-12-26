import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(100)
lamb , mu , Xzero = 2,1,1  #problem parameters
T = 1
N = 2**8
dt = 1/N
dW = np.sqrt(dt) * np.random.randn(1,N)
W = np.cumsum(dW)
deltt = np.arange(0, 1, dt)

xtrue = Xzero* np.exp((lamb - 1/2*mu**2)*deltt + mu*W)

R = 4
Dt = R*dt
L = int(N/R)

Xem = np.zeros(L)
Xtemp = Xzero
for j in range(L):
    Winc = np.sum(dW[R * (j - 1) +1 :R * j])

    Xtemp = Xtemp + (Dt * lamb * Xtemp) + (mu* Xtemp * Winc)
    Xem[j] = Xtemp

deltT = np.arange(0,1,Dt)
plt.figure(1)
plt.plot(deltt,xtrue)
plt.plot(deltT,Xem)

np.random.seed(100)

# Problem parameters
lamb, mu, Xzero = 2, 1, 1
T = 1
N = 2**8
dt = 1 / N

# Brownian increments
dW = np.sqrt(dt) * np.random.randn(N)
W = np.cumsum(dW)

# Exact solution
deltt = np.arange(0, T + dt, dt)
xtrue = Xzero * np.exp((lamb - 0.5 * mu**2) * deltt + mu * np.concatenate(([0], np.cumsum(dW))))

# Plotting the exact solution
plt.figure(2)
plt.plot(deltt, xtrue, 'm-')

# Euler–Maruyama method
R = 4
Dt = R * dt
L = int(N / R)

Xem = np.zeros(L)
Xtemp = Xzero
for j in range(L):
    Winc = np.sum(dW[R * (j - 1) :R * j])
    Xtemp = Xtemp + (Dt * lamb * Xtemp )+ (mu * Xtemp * Winc)
    Xem[j] = Xtemp

deltT = np.arange(0, T + Dt, Dt)
plt.plot(deltT, np.concatenate(([Xzero], Xem)), 'r--*')


plt.xlabel('t', fontsize=12)
plt.ylabel('X', fontsize=16, rotation=0, ha='right')

# Calculate the endpoint error
emerr = abs(Xem[-1] - xtrue[-1])
print("Endpoint Error:", emerr)

np.random.seed(100)
lamb , mu , Xzero = 2 , 1 , 1
T, N = 1, 2^9
dt = T/N

M  = 1000 #number of paths sampled

Xerr = np.zeros((M,5))

for s in range(M):
    dW = np.sqrt(dt)* np.random.randn(N)
    W = np.cumsum(dW)
    Xtrue = Xzero*np.exp((lamb-0.5*mu**2)+mu*W[-1])
    for p in range(1,6):
        R = 2**(p-1)
        Dt = R*dt
        L = int(N/R)
        Xtemp = Xzero
        for j in range(L):
            Winc = np.sum(dW[R * (j - 1) :R * j])
            Xtemp = Xtemp + Dt*lamb*Xtemp + mu*Xtemp*Winc
        Xerr[s,p -1 ] = abs(Xtemp - Xtrue)


dtvals = [dt * (2 ** i) for i in range(5)]

meanerr = np.mean(Xerr,axis = 0)

print(f'dt is {dt}')
print(f'dtvals are {dtvals}')
print(f'mean error is {meanerr}')
print(Xerr)

sec = [i ** .5 for i in dtvals]

# # dtvals will contain the values [dt, 2*dt, 4*dt, 8*dt, 16*dt]
plt.figure(3)
plt.loglog(dtvals, meanerr, 'b*-');
plt.loglog(dtvals,sec,'r--')

plt.show()

