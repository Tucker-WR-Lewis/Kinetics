# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:16:08 2024

@author: Tucker Lewis
"""

import numpy as np
import scipy as sp
import time

# def obj(ind):
#     # return (ackley(ind[0],ind[1]) - ackley(0,0))**2
#     return ackley(ind[0], ind[1])

def ackley(x,y):
    return -20*np.exp(-0.2*np.sqrt(0.5*(x**2+y**2)))-np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)))+np.e+20

def obj(x):
    #rastrigin
    to_sum = []
    for i in range(x.shape[1]):
        to_sum.append(x[:,i]**2 - 10*np.cos(2*np.pi*x[:,i]))
    return 10*x.shape[1]+np.sum(to_sum, axis = 0)

#how to vectorize the reaction system for the kinetics script
    # k2 = [sym.symbols('k{}'.format(int(str(k[i])[1:])+len(k))) for i in range(len(k))]
    # ydot2 = []
    # for equation in ydot:
    #     for i in range(len(k)):
    #         equation = equation.subs(k[i],k2[i])
    #     ydot2.append(equation)
    
    # ydot_full = ydot + ydot2
    # k_full = k + tuple(k2)
    
    # t = sym.symbols('t')
    # f_lamb2 = sym.lambdify((t, y) + k_full, ydot_full, "numpy")
    # f_jit2 = nb.njit(f_lamb2)
    # initial_cons_stacked = np.concatenate([initial_cons[0],initial_cons[1]],axis = 1)
    # f_jit2(0,initial_cons_stacked,*np.repeat(fake_ks,2).shape*np.repeat(k_l_bounds,2))


start = time.time()
#not vectorized this takes approximately 30s for 25 dimensions, with crossover at 0.05 and best1 strategy
dimensions = 30
bounds = np.array([[-5.12,-5.12, -5.12, -5.12, -5.12],[5.12,5.12, 5.12, 5.12, 5.12]])
bounds = np.array([np.repeat(-5.12,dimensions),np.repeat(5.12,dimensions)])
crossover = 0.05

#bounds is a m x n where m = 2 and n = number of parameters. m = 0 is lower bound, m = 1 is upper bound
num_params = bounds.shape[1]
#initialize population of candidate solutions with a Sobol distribution
sampler = sp.stats.qmc.Sobol(num_params) #not sure which dimension to use yet
pop_size = int(2 ** np.ceil(np.log2(num_params*10)))
pop = bounds[0, :] + ((sampler.random(pop_size)) * (bounds[1, :] - bounds[0, :]))
#evaluate initial population of candidate solutions
obj_all = obj(pop)
# find the best performing vector of initial population
best_vector = pop[np.argmin(obj_all)]
best_vectors = np.repeat(best_vector,pop_size).reshape(num_params,pop_size).transpose()
best_obj = min(obj_all)
prev_obj = best_obj
#run iterations of the algorithm
iterations = 1500
for i in range(iterations):
    F = np.random.uniform(low = 0.01, high = 1.99)
#perform mutation
    #rand1 strategy
    candidates = np.random.choice([i for i in range(pop_size)],(3,pop_size))
    mutated = np.clip(pop[candidates[0]] + F * (pop[candidates[1]] - pop[candidates[2]]),bounds[0],bounds[1])
    #best1 strategy
    # candidates = np.random.choice([i for i in range(pop_size)],(2,pop_size))
    # mutated = np.clip(best_vectors + F * (pop[candidates[0]] - pop[candidates[1]]),bounds[0],bounds[1])
#perform crossover
    #generate a uniform random value for every parameter
    p = np.random.uniform(low = 0, high = 1, size = (pop_size, num_params))
    #generate trial vector by binomial crossover
    trials = np.copy(pop)
    trials[np.where(p < crossover)] = mutated[np.where(p < crossover)]
# compute objective function value for target vector
    obj_target = obj(pop)
# compute objective function value for trial vector
    obj_trial = obj(trials)
# perform selection
    to_replace = np.where(obj_trial < obj_target)[0]
    # replace the target vector with the trial vector
    pop[to_replace] = trials[to_replace]
    # store the new objective function value
    obj_all[to_replace] = obj_trial[to_replace]
    # find the best performing vector at each iteration
    best_obj = min(obj_all)
    # store the lowest objective function value
    if best_obj < prev_obj:
        best_vector = pop[np.argmin(obj_all)]
        prev_obj = best_obj
        # report progress at each iteration
        # print('Iteration: %d f([%s]) = %.5f' % (i, np.around(best_vector, decimals=2), best_obj))
        print('Iteration: %d: %.5f' % (i,best_obj))
    if best_obj < 0.0001:
        break
    if i == iterations - 1:
        print('Did not Converge')
        
print('Total Time:{}'.format(time.time()-start))