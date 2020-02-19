import numpy as np
import random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt


'''
Things to implement:
    - call minimize
    - extract non-zero values
    - calculate the b value using equation
    - indicator function
'''

# Linear Kernel
def linear_k(x, y):
    return np.dot(np.transpose(x), y)

# Polynomial Kernel
def polynomial_k(x, y, p):
    return ((np.dot(np.transpose(x), y) + 1)**p)

# Radial Basis Kernel
def rbf_k(x, y, sigma):
    a = (np.linalg.norm(x-y))**2
    b = 2*(sigma**2)
    return math.exp(-a/b)

# Pre-compute
def pre_compute(x):
    N = len(x)
    P = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            P[i][j] = -1 * linear_k(x[i],x[j])
    mr_world_wide = P

# Objective
def objective(alpha):
    return (0.5*np.dot(alpha,alpha)*mr_world_wide)-np.sum(alpha)

# Zerofun
def zerofun(alpha):
    for i in alpha:
        if i < 0 and i > C:
            return False
    if np.dot(alpha,targets) == 0:
        return True
    else:
        return False

# Caculate b
def calc_b(alpha, si):
    for (i, a) in enumerate(alpha):
        if a > 0 and (a < C or C is None):
            return sum([a*t*linear_k(xs[i],x) for (a,t,x) in zip(alphas,ts,xs)]) - ts[i]
    return None

# Indicator
def ind(alpha, b, s):
    return sum([a*t*linear_k(s,x) for (a,t,x) in zip(alphas,ts,xs)]) - b



x = np.array([1,3,-5])
y = np.array([4,-2,-1])
C = 1

num_of_train = 10
mr_world_wide = np.zeros((1,1))
t = np.array([1,1,-1])


np.random.seed(100)
classA = np.concatenate(
        (np.random.randn(10,2) * 0.2 + [1.5, 0.5],
         np.random.randn(10,2) * 0.2 + [-1.5,0.5]))
classB = np.random.randn(20,2) * 0.2 + [0.0, -0.5]

inputs = np.concatenate((classA,classB))
targets = np.concatenate(
    (np.ones(classA.shape[0]),
    -np.ones(classB.shape[0])))
print(targets)
N = inputs.shape[0]
B = [(0,C) for b in range(N)]
XC = {'type':'eq','fun':zerofun}
start = np.zeros(N)
ret = minimize(objective, start, bounds = B, constraints = XC)
alpha = ret['x']

permute=list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]



plt.plot([p[0] for p in classA],
        [p[1] for p in classA],
        'b.')
plt.plot([p[0] for p in classB],
         [p[1] for p in classB],
         'r.')

plt.axis('equal') # Force same scale on both axes
xgrid=np.linspace(-5,5)
ygrid=np.linspace(-4,4)

grid=np.array([[indicator(x,y)
              for x in xgrid]
              for y in ygrid])
plt.contour(xgrid, ygrid, grid,
            (-1.0, 0.0, 1.0),
            colors=('red','black','blue'),
            linewidths=(1,3,1))
plt.savefig('svmplot.pdf') # Save a copy in a file
#plt.show() # Show the plot on the screen

