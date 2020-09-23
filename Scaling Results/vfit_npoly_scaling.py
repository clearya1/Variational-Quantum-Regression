import qiskit
from qiskit import QuantumCircuit
from qiskit import Aer, execute
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

"""=======================Necessary Functions========================="""

#Code for computing inner products on a real quantum computer
def inner_prod(vec1, vec2):
    #first check lengths are equal
    if len(vec1) != len(vec2):
        raise ValueError('Lengths of states are not equal')
        
    circ = QuantumCircuit(nqubits+1,1)
    vec = np.concatenate((vec1,vec2))/np.sqrt(2)
    
    #print(np.linalg.norm(vec1))
    #print(np.linalg.norm(vec2))
    
    circ.initialize(vec, range(nqubits+1))
    circ.h(nqubits)
    circ.measure(nqubits,0)

    #print(circ.draw())

    backend = Aer.get_backend('qasm_simulator')
    job = execute(circ, backend, shots=10000)

    result = job.result()
    outputstate = result.get_counts(circ)
    #print(o)

    if ('0' in outputstate.keys()):
        m_sum = float(outputstate["0"])/10000
    else:
        m_sum = 0
     
    #print(2*m_sum-1)
    return 2*m_sum-1

#Returns the cost functions, based on n parameters for an n-th order polynomial
def calculate_cost_function_n(parameters):

    yphi = parameters[0]*np.sqrt(N)/ynorm*inner_prods[0]

    for i in range(1,len(parameters)):

        xpow = x**i
        xnormpow = np.linalg.norm(xpow)
        xpow = xpow/xnormpow
        
        yphi += parameters[i]*xnormpow/ynorm*inner_prods[i]
    
    #print((1-yphi)**2)
    return (1-yphi)**2

def return_fits(xfit):
    c_fit = np.zeros(100)
    q_fit = np.zeros(100)
    for i in range(order+1):
        q_fit += xfit**i*out['x'][i]
        c_fit += xfit**i*class_fit[i]

    return q_fit, c_fit

"""======================Looping for Scaling============================="""

it = 0
ctimes = np.zeros(it)
qtimes = np.zeros(it)
Ns = np.zeros(it)

for j in range(it):

    """============Input Data Sets Here and Polynomial Order================="""

    x = np.arange(0,2**(j+2),1)
    y = (2*x-1)**3 + [random.uniform(-5,5) for p in range(2**(j+2))]
    Ns[j] = 2**(j+2)
    order = 3

    """===============Normalising y data set and \ket{1}====================="""

    start = time.time()
    N = len(x)
    nqubits = math.ceil(np.log2(N))
    ones = np.ones(N)/np.sqrt(N)

    ynorm = np.linalg.norm(y)
    y = y/ynorm

    """===========Computing inner products once off to save time============="""

    inner_prods = np.zeros(order+1)
    inner_prods[0] = inner_prod(y,ones)

    for i in range(1,order+1):

        xpow = x**i
        xnormpow = np.linalg.norm(xpow)
        xpow = xpow/xnormpow
        
        inner_prods[i] = inner_prod(y,xpow)
        
    """=================Running Classical Optimiser========================="""
        
    #Random initial guess for parameters
    x0 = [random.uniform(-10,10) for p in range(order+1)]
    #Can change the optimiser method here
    out = minimize(calculate_cost_function_n, x0=x0, method="BFGS", options={'maxiter':200}, tol=1e-6)

    end=time.time()
    qtime = end-start
    print("Quantum Time: ", qtime)
    qtimes[j] = qtime

    #Classical fit
    start=time.time()
    class_fit = np.polyfit(x,y*ynorm,order)
    class_fit = class_fit[::-1]
    end=time.time()
    ctime = end-start
    print("Classical Time: ", ctime)
    ctimes[j] = ctime

    print("Variational Quantum Fit: ", out['x'])
    print("Classical Fit: ", class_fit)

"""==========================Plotting===================================="""

print("Quantum Times: ", qtimes)
print("Classical Times: ", ctimes)

"""plt.scatter(Ns, qtimes, label="Quantum")
plt.scatter(Ns, ctimes, label="Classical")
plt.legend()
plt.xlabel("N")
plt.ylabel("Time [s]")
plt.show()"""

#Results from running on Kay @ ICHEC follow:

kay_ctimes = [8.63814354e-03, 2.29811668e-03, 7.08341599e-04, 7.10725784e-04, 2.19583511e-04, 1.88350677e-04, 3.77080441e-02, 2.46553421e-02, 1.86512470e-02, 1.67284012e-02, 3.41582298e-03, 4.32085991e-03, 2.68661976e-02, 1.88193321e-02, 3.45296860e-02, 1.58960819e-02, 3.14261913e-02, 5.88543415e-02, 1.00360394e-01, 2.04956055e-01]
kay_qtimes = [1.32984161e-01, 1.04918718e-01, 9.29152966e-02, 1.00656033e-01, 1.13978148e-01, 1.39630556e-01, 1.83788538e-01, 2.71724701e-01, 4.47867155e-01, 7.00398207e-01, 1.01298451e+00, 1.40795231e+00, 2.39123535e+00, 4.40328908e+00, 8.23174143e+00, 1.64347804e+01, 3.59081967e+01, 7.53697002e+01, 1.59989884e+02, 3.28928713e+02]

Ns = np.zeros(len(kay_ctimes))
for i in range(len(kay_ctimes)):
    Ns[i] = 2**(i+2)
    
kay_qtimesfit = np.polyfit(Ns, kay_qtimes, 1)
kay_ctimesfit = np.polyfit(Ns, kay_ctimes, 1)

Ntimesfit = np.linspace(min(Ns), max(Ns), 100)

plt.scatter(Ns, kay_qtimes, color='r', label="Quantum $\propto$"+str(np.round(kay_qtimes[0],2))+"$\\times$N")
plt.scatter(Ns, kay_ctimes, color='b', label="Classical $\propto$"+str(np.round(kay_ctimes[0],2))+"$\\times$N")
plt.plot(Ntimesfit, Ntimesfit*kay_qtimesfit[0]+kay_qtimesfit[1], 'r')
plt.plot(Ntimesfit, Ntimesfit*kay_ctimesfit[0]+kay_ctimesfit[1], 'b')
plt.xlabel("N")
plt.ylabel("Time [s]")
plt.legend()
plt.show()
