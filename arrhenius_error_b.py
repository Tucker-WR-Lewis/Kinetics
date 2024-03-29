# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 10:16:08 2024

@author: Tucker Lewis
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def exp_fun(inv_T, a1, Ea):
    global b
    kb = 8.314e-3
    return b*a1*np.exp(-Ea/kb*inv_T)

# def hs_fun(inv_T_temp, r_eff, Ea, frac):
#     global lan_rates, red_mass, T, fit_index, hs_rates, col_rate
#     hs_rates = np.sqrt(8*1.381e-23*T/np.pi/red_mass[fit_index])*100*np.pi*(r_eff*1e-8)**2
#     col_rate = lan_rates[fit_index] + hs_rates
#     # print([r_eff,Ea,frac])
#     return frac*col_rate*np.exp(-Ea/8.314e-3*inv_T_temp)*1e12

def hs_fun(inv_T_temp, r_eff, Ea, frac):
    global lan_rates, red_mass, T, fit_index, hs_rates, col_rate
    hs_rates = np.sqrt(8*1.381e-23*T/np.pi/red_mass[fit_index])*100*np.pi*(r_eff*1e-8)**2
    col_rate = lan_rates[fit_index] + hs_rates
    # print([r_eff,Ea,frac])
    return frac*np.exp(-Ea/8.314e-3*inv_T_temp)*1e10

def diff_fit(params, inv_T_temp, rates,red_mass_temp, fit_index, lan_rates_temp):
    r_eff = params[0]
    Ea = params[1]
    frac = params[2]
    hs_rates = np.sqrt(8*1.381e-23*T/np.pi/red_mass_temp[fit_index])*100*np.pi*(r_eff*1e-8)**2
    col_rate = lan_rates_temp[fit_index] + hs_rates
    rates_calc = frac*col_rate*np.exp(-Ea/8.314e-3*inv_T_temp)
    # print(rates_calc)
    return np.sum((rates - rates_calc)**2)

def diff_fit_plot(params, inv_T_temp, rates,red_mass_temp, fit_index, lan_rates_temp):
    r_eff = params[0]
    Ea = params[1]
    frac = params[2]
    hs_rates = np.sqrt(8*1.381e-23*T/np.pi/red_mass_temp[fit_index])*100*np.pi*(r_eff*1e-8)**2
    col_rate = lan_rates_temp[fit_index] + hs_rates
    return frac*col_rate*np.exp(-Ea/8.314e-3*inv_T_temp)

T = np.array([300, 400, 500, 600])
# T = np.array([298,348,373,398,423,448,473,498,523,548,573,598,623])
inv_T_main = 1/T

La = np.array([0.9792, 1.1496, 1.1034, 1.0012])
Ce = np.array([0.98151, 0.99711, 1.0976, 1.0721])
Pr = np.array([0.837, 0.906, 0.946, 0.915])
Nd = np.array([0.74783, 0.83478, 0.90435, 0.94957])
Sm = np.array([0.65096, 0.71344, 0.75218, 0.78883])
Gd = np.array([0.89, 0.92878, 0.97782, 0.94804])
Tb = np.array([0.8586, 0.9, 0.93918, 0.96608])
Dy = np.array([0.58348, 0.6034, 0.62039, 0.64558])
Ho = np.array([0.60548, 0.68296, 0.75527, 0.78188])
Er = np.array([0.5358, 0.57629, 0.65023, 0.64965])
Lu = np.array([0.18428, 0.19317, 0.24441, 0.24794])

# La = np.array([0,0,0.43148,0.41771,0.4592,0.44621,0.40871,0.4979,0.48585,0,0,0,0])
# Ce = np.array([0,0.29185,0.2819,0.30632,0.29713,0.34122,0.30653,0.34853,0,0,0,0,0])
# Pr = np.array([0,0.16082,0.15821,0.16708,0.16747,0.21523,0.24267,0.24397,0.23078,0.22071,0.2228,0,0])
# Nd = np.array([0.096547,0.12806,0.1352,0.15873,0.17828,0.19423,0.19158,0.19169,0.21863,0.22546,0.23209,0.22718,0.24484])
# Sm = np.array([0,0,0.086297,0.10861,0.10535,0.12336,0.14305,0.14439,0.14576,0.1329,0.146210,0,0])
# Eu = np.array([0.012551,0.020847,0.019561,0.023113,0.027012,0.036747,0.035762,0.039832,0.043727,0.049838,0.055701,0.056796,0])
# Gd = np.array([0,0.14295,0.14383,0.15038,0.16747,0.17061,0.18903,0.20165,0.20163,0,0,0,0])
# Tb = np.array([0.05471,0.07743,0.094926,0.13088,0.11075,0.12336,0.13794,0.14439,0.17977,0.18036,0.18567,0.18629,0])
# Dy = np.array([0.025746,0.035737,0.040272,0.05291,0.054024,0.060369,0.063861,0.074685,0.087454,0.094929,0.10444,0.10905,0])
# Ho = np.array([0.016091,0.027398,0.028766,0.038986,0.043219,0.05512,0.058752,0.054769,0.075308,0.068823,0.074268,0.074971,0.080128])
# Er = np.array([0.013838,0.021442,0.024163,0.036202,0.048622,0.04987,0.04598,0.064727,0.055873,0.07357,0.074268,0.079514,0.08458])
# Tm = np.array([0.010298,0.017571,0.022437,0.027569,0.035116,0.036747,0.040871,0.04979,0.051015,0.064077,0.064984,0.068155,0.073451])

elems = np.array([La, Ce, Pr, Nd, Sm, Gd, Tb, Dy, Ho, Er, Lu])
elems_names = np.array(['La', 'Ce', 'Pr', 'Nd','Sm','Gd','Tb','Dy','Ho','Er','Lu'])

# n2 = np.array([2.0307e-11, 3.1032e-11, 4.2087e-11, 6.1089e-11])
# n4 = np.array([3.0099e-12, 4.2746e-12, 5.0059e-12, 6.8553e-12])
# n5 = np.array([2.09e-12, 2.69e-12, 2.49e-12, 2.69e-12])
# n6 = np.array([1.38e-12, 3e-12, 2.24e-12, 3.58e-12])
# n8 = np.array([1.1e-11, 1.21e-11, 8.21e-12, 1.1e-11])

lan_rates = np.array([7.02e-10, 6.72e-10, 6.65e-10, 6.61e-10, 6.56e-10])

n = np.array([2,3,5,6,8])
r = 1.4
# total_r = r*np.sqrt(n)*np.sqrt((2*np.sqrt(3)/np.pi))
total_r = r * n**(1/3) + 1.15
red_mass = np.array([36.55, 39.93, 40.68, 41.20, 41.87])*1.66e-27

hs_rate = []
for index, mass in enumerate(red_mass):
    temp_list = []
    for temp in T:
        temp_list.append(np.sqrt(8*8.314e-3*temp/np.pi*mass)*np.pi*total_r[index]**2)
    hs_rate.append(temp_list)
hs_rate = np.array(hs_rate)

col_rates = lan_rates + hs_rate.transpose()
# col_rates = lan_rates

# elems = np.array([n2,n4,n5,n6,n8]).transpose()/col_rates
# elems = np.array([n2,n4,n5,n6,n8]).transpose()

# elems = elems.transpose()
# elems_names = np.array(['n2','n4','n5','n6','n8'])

error_per = 0.15/1.96

outputs = []
errors_out = []
fits_out = []

l_bounds = np.zeros(3)
l_bounds[-1] = 0
h_bounds = np.repeat(np.inf,3)
param_bounds = sp.optimize.Bounds(l_bounds,h_bounds)

b_vals = np.linspace(0.01,2,200)
b_vals = np.array([1.0])
# b_vals = np.array([])
gofs = []

if b_vals.size > 0:
    l_bounds = np.zeros(2)
    l_bounds[-1] = 0
    l_bounds = [1.02, -1]
    h_bounds = np.repeat(np.inf,2)
    h_bounds = [1.13, 10]
    param_bounds = sp.optimize.Bounds(l_bounds,h_bounds)
    for b in b_vals:
        gof = []
        for names, species in enumerate(elems):
            index_nonzero = np.nonzero(species)
            species = species[index_nonzero]
            inv_T = inv_T_main[index_nonzero]
            temp_list = []
            fig = plt.figure()
            for i in range(10000):
                # species_temp = np.random.uniform(species*(1-error_per),species*(1+error_per))
                species_temp = np.random.normal(species, error_per*species)
                # popt_temp, pcov_temp = sp.optimize.curve_fit(exp_fun, inv_T, species_temp, p0 = [1,1], bounds = param_bounds)
                popt_temp, pcov_temp = sp.optimize.curve_fit(exp_fun, inv_T, species_temp, p0 = [1.07,1], bounds = param_bounds)
                temp_list.append(popt_temp)
                if i%10 == 0:
                    plt.semilogy(inv_T,species_temp,'o', color = 'red')
            temp_list = np.array(temp_list)
            outputs.append(temp_list)
            errors = np.round(([np.percentile(temp_list[:,0],[2.5, 97.5]),np.percentile(temp_list[:,1],[2.5, 97.5])]),3)
        
            plt.semilogy(inv_T,exp_fun(inv_T,errors[:,0][0], errors[:,0][1]), color = 'black')
            plt.semilogy(inv_T,exp_fun(inv_T,errors[:,1][0], errors[:,1][1]), color = 'black')
        
            
            popt, pcov = sp.optimize.curve_fit(exp_fun, inv_T, species, p0 = [1.07, 1], bounds = param_bounds)
            # gof.append((exp_fun(inv_T, popt[0], popt[1]) - species)**2)
        
            popt = np.round(popt,3)
            fits_out.append(popt)
            
            errors_out.append(np.array([[errors[0][0], popt[0], errors[0][1]], [errors[1][0], popt[1], errors[1][1]]]))
            
            plt.semilogy(inv_T,species,'o')
            plt.semilogy(inv_T, exp_fun(inv_T, popt[0], popt[1]))
            plt.title(elems_names[names])
            fig_string_1 = 'A: {} {} {}'.format(errors[0][0], popt[0], errors[0][1])
            fig_string_2 = 'Ea (kJ/mol): {} {} {}'.format(errors[1][0], popt[1], errors[1][1])
            plt.annotate(fig_string_1, (0.55,0.8), xycoords = 'figure fraction')
            plt.annotate(fig_string_2, (0.55,0.75), xycoords = 'figure fraction')
            # gofs.append(np.sum(gof))

hs_fun_go = 0
if hs_fun_go == 1:
    l_bounds = np.repeat(-np.inf,3)
    l_bounds[0] = 0
    l_bounds[1] = -10
    l_bounds[2] = 0.01
    h_bounds = np.repeat(np.inf,3)
    h_bounds[0] = 75
    h_bounds[1] = 10
    h_bounds[2] = 1.1
    param_bounds = sp.optimize.Bounds(l_bounds,h_bounds)
    p0s = [1, 1, 0.25]
    gof = []
    for fit_index, species in enumerate(elems):
        if fit_index == 0:
            index_nonzero = np.nonzero(species)
            species = species[index_nonzero]
            inv_T = inv_T_main[index_nonzero]
            temp_list = []
            fig = plt.figure()
            for i in range(10):
                species_temp = np.random.normal(species, error_per*species)
                popt_temp, pcov_temp = sp.optimize.curve_fit(hs_fun, inv_T, species_temp*1e10, p0 = p0s, bounds = param_bounds, maxfev = 5000)
                temp_list.append(popt_temp)
                if i%10 == 0:
                    plt.semilogy(inv_T,species_temp,'o', color = 'red')
            temp_list = np.array(temp_list)
            outputs.append(temp_list)
            errors = np.round(([np.percentile(temp_list[:,0],[2.5, 97.5]),np.percentile(temp_list[:,1],[2.5, 97.5]), np.percentile(temp_list[:,2],[2.5, 97.5])]),3)
        
            plt.semilogy(inv_T,hs_fun(inv_T,errors[:,0][0], errors[:,0][1], errors[:,0][2]), color = 'black')
            plt.semilogy(inv_T,hs_fun(inv_T,errors[:,1][0], errors[:,1][1], errors[:,1][2]), color = 'black')
        
            
            popt, pcov = sp.optimize.curve_fit(hs_fun, inv_T, species*1e12, p0 = p0s, bounds = param_bounds, maxfev = 5000)
        
            popt = np.round(popt,3)
            fits_out.append(popt)
            
            errors_out.append(np.array([[errors[0][0], popt[0], errors[0][1]], [errors[1][0], popt[1], errors[1][1]]]))
            
            plt.semilogy(inv_T,species,'o')
            plt.semilogy(inv_T, hs_fun(inv_T, popt[0], popt[1], popt[2])/1e10)
            plt.title(elems_names[fit_index])
            fig_string_1 = 'A: {} {} {}'.format(errors[0][0], popt[0], errors[0][1])
            fig_string_2 = 'Ea (kJ/mol): {} {} {}'.format(errors[1][0], popt[1], errors[1][1])
            plt.annotate(fig_string_1, (0.55,0.8), xycoords = 'figure fraction')
            plt.annotate(fig_string_2, (0.55,0.75), xycoords = 'figure fraction')
            gofs.append(np.sum(gof))

diff_go = 0
if diff_go == 1:
    res_list = []
    for fit_index, rates in enumerate(elems):
        if fit_index == 0:
            plt.figure()
            index_nonzero = np.nonzero(rates)
            rates = rates[index_nonzero]
            inv_T = inv_T_main[index_nonzero]
            l_bounds = [0,-100, 0]
            h_bounds = [100, 100, 1]
            param_bounds_diff = sp.optimize.Bounds(l_bounds,h_bounds)
            diff_args = (inv_T, rates,red_mass, fit_index, lan_rates)
            
            temp_list = []
            for i in range(1000):
                rates_temp = np.random.normal(rates, error_per*rates)
                diff_args_temp = (inv_T, rates_temp,red_mass, fit_index, lan_rates)
                res = sp.optimize.differential_evolution(diff_fit, param_bounds_diff, args = diff_args_temp, strategy='best2bin', 
                                                          maxiter=1000, popsize=20, tol=0.0001, mutation= (0.1,1.99), recombination=0.6,  
                                                          seed=None, callback=None, disp=False, polish=False, init='sobol', 
                                                          atol=0, updating='immediate', workers=1, x0=None, 
                                                          integrality=None, vectorized=False)  
                temp_list.append(res.x)
                if i%10 == 0:
                    plt.semilogy(inv_T,rates_temp,'o', color = 'red')
                    print(i)
            temp_list = np.array(temp_list)
            outputs.append(temp_list)
            errors = np.round(([np.percentile(temp_list[:,0],[2.5, 97.5]),np.percentile(temp_list[:,1],[2.5, 97.5]), np.percentile(temp_list[:,2],[2.5, 97.5])]),3)
            errors = np.round(([np.percentile(temp_list[:,0],[10, 90]),np.percentile(temp_list[:,1],[10, 90]), np.percentile(temp_list[:,2],[10, 90])]),3)    
            
            plt.semilogy(inv_T,diff_fit_plot([errors[:,0][0], errors[:,0][1], errors[:,0][2]], *diff_args), color = 'black')
            plt.semilogy(inv_T,diff_fit_plot([errors[:,1][0], errors[:,1][1], errors[:,1][2]], *diff_args), color = 'black')
            
            res = sp.optimize.differential_evolution(diff_fit, param_bounds_diff, args = diff_args, strategy='best2bin', 
                                                      maxiter=1000, popsize=20, tol=0.0001, mutation= (0.1,1.99), recombination=0.7,  
                                                      seed=None, callback=None, disp=False, polish=False, init='sobol', 
                                                      atol=0, updating='immediate', workers=1, x0=None, 
                                                      integrality=None, vectorized=False)   
            res_list.append(res)
        
            plt.semilogy(inv_T,rates,'o')
            plt.semilogy(inv_T, diff_fit_plot(res.x,*diff_args))
            plt.title(elems_names[fit_index])

# gofs = np.array(gofs)

# plt.figure()
# plt.semilogy(b_vals,gofs)
errors_out = np.array(errors_out)