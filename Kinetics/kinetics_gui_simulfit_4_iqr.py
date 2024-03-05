# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 10:29:27 2023

@author: Tucker Lewis
"""

import sympy as sym
import string
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
import numba as nb
import multiprocessing
import pathlib
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import threading
import queue
import datetime

def prod(seq):
    product = 1
    if seq:
        for i in seq:
            product = product * i
    return product if seq else 1

def mk_exprs_symbs(rxns, names):
    # create symbols for reactants
    symbs = sym.symbols(" ".join(names))
    t = sym.symbols('t')
    # map between reactant symbols and keys in r_stoich, net_stoich
    c = dict(zip(names, symbs))
    f = {n: 0 for n in names}
    k = []
    for coeff, r_stoich, net_stoich in rxns:
        k.append(sym.S(coeff))
        r = k[-1]*prod([sym.Function(str(c[rk]**p))(t) for rk, p in r_stoich.items()])  # EXERCISE: c[rk]**p
        for net_key, net_mult in net_stoich.items():
            f[net_key] += net_mult*r  # EXERCISE: net_mult*r
    return [f[n] for n in names], [sym.Function(str(i))(t) for i in symbs], tuple(k)

def getodes(kinin_temp):
    f = open(kinin_temp)
    text = f.read()
    text_out = text.split()
    f.close()
    
    r_start = [0]
    r_end = []
    p_start = []
    p_end = []
    
    k_out = []
    low_bound = []
    high_bound = []
    
    iso_index = len(text_out)
    text_iso = ''
    for index, iso in enumerate(text_out):
        if iso == 'Iso':
            text_iso = text_out[index+1:]
            iso_index = index
    
    for index, con in enumerate(text_out):
        if con == 'Conditions':
            text_out = text_out[0:index]
            break    
    
    for index, i in enumerate(text_out):      
        for j in i:
            if i[0] == 'k' and i[1].isnumeric():
                k_out.append(i)
                low_bound.append(float(text_out[index+1]))
                high_bound.append(float(text_out[index+2]))
                r_start.append(index + 3)
                p_end.append(index)
            if j.isalpha():
                break
            if j == '>':
                r_end.append(index)
                p_start.append(index+1)
    
    low_bound = np.array(low_bound)
    high_bound = np.array(high_bound)
    reactants = []
    products = []
    
    for index, i in enumerate(r_end):
        reactants.append(text_out[r_start[index]:r_end[index]])
        products.append(text_out[p_start[index]:p_end[index]])
    
    reactants3 = []
    products3 = []
    
    for i in reactants:
        reactants2 = []
        for j in i:
            if j[0].isnumeric() or j[0].isalpha():
                reactants2.append(j)
        reactants3.append(reactants2)
    for i in products:
        products2 = []
        for j in i:
            if j[0].isnumeric() or j[0].isalpha():
                products2.append(j)
        products3.append(products2)
    
    species = [sub[item] for item in range(len(reactants3)) for sub in [reactants3, products3]]
    
    temp = []
    for i in species:
        for j in i:
            if j not in temp:
                temp.append(j)
    reactmap = []
    prodmap = []
    for i in reactants3:
        res = []
        for j in i:
            res.append(string.ascii_uppercase[temp.index(j)])
        reactmap.append(res)
    for i in products3:
        res = []
        for j in i:
            res.append(string.ascii_uppercase[temp.index(j)])
        prodmap.append(res)
    res = []
    for i in temp:
            res.append(string.ascii_uppercase[temp.index(i)])
    
    reactions = []
        
    for i in range(len(reactmap)):
        react_dict = {}
        net_dict = {}
        for j in reactmap[i]:
            react_dict[j] = 1
            net_dict[j] = -1
        for j in prodmap[i]:
            net_dict[j] = 1
        reactions.append([k_out[i], react_dict, net_dict])
    
    names = res

    sym.init_printing()
    ydot, y, k = mk_exprs_symbs(reactions, names)
    
    names = temp
    
    text_lines = np.array(text.split('\n'))
    con_index = np.where(text_lines == "Conditions")[0][0]
    iso_index = np.where(text_lines == 'Iso')[0][0]
    con_text = text_lines[con_index+1:iso_index-1]
    
    con_limits_low = []
    con_limits_high = []
    constraints = []
    for line in con_text:
        stuff = line[0:-2].split('<')
        con_limits_low.append(float(stuff[0]))
        con_limits_high.append(float(stuff[2]))
        constraints.append(stuff[1])
        
    constraints_new = ['']*len(constraints)
    for index_constraints, current_cons in enumerate(constraints):
        for index_split, split_cons in enumerate(current_cons.split()):
            for index, cur_k in enumerate(k):
                if sym.symbols(split_cons) == cur_k:
                    constraints_new[index_constraints] = constraints_new[index_constraints] +'params[%d]'%index
            if index_split < len(current_cons.split())-1:
                constraints_new[index_constraints] = constraints_new[index_constraints] +' + '
    
    spec = np.array([a[0] for a in species])
    indices = np.array(np.unique(spec,return_index = True)[1])
    indices.sort()
    specs = [str(spec[int(index)]) for index in indices]
    
    red_iso = []
    for index, stri in enumerate(text_iso):
        if stri != '=':
            red_iso.append(stri)
    iso_index = []
    temps = []
    for stri in red_iso:
        for index, spi in enumerate(specs):
            if spi == stri:
                temps.append(index)
        if stri == '|':
            iso_index.append(temps)
            temps = []
    
    return ydot, y, k, low_bound, high_bound, specs, constraints_new, con_limits_low, con_limits_high, names, reactants3, products3, iso_index

def getgof(params,numpoints_temp,numks,ydatas,neutral_con_temp, iso_temp): #ins is a tuple 
    global num_analyze, res_per_square
    final_res = []
    k_vals = params[0:numks]
    num_cons = int(len(params[numks:])/len(ydatas))
    for num_analyze in range(len(ydatas)):   
        ydata = ydatas[num_analyze]
        con0 = params[numks+num_analyze*num_cons:numks+num_analyze*num_cons+num_cons]
        in_cons = np.repeat(con0, numpoints_temp).reshape(ydata.shape[1],ydata.shape[0])
        in_cons[1] = neutral_con_temp[num_analyze]
        
        fit_ys = np.delete(solve(in_cons, k_vals).reshape(in_cons.shape[1],in_cons.shape[0]),1,axis=1)
        for indices in iso_temp:
            # fit_ys[:,indices[0]] = np.sum(data[:,iso_temp[0]], axis =1 )
            fit_ys[:,indices[0]] = np.sum(fit_ys[:,indices], axis =1 )
            fit_ys[:,indices[1:]] = np.zeros([ydata.shape[0],len(indices[1:])])
        ydata = np.delete(ydata,1,axis=1)
        
        res_abs = np.abs(fit_ys-ydata)
        res_fract = res_abs/(ydata+1)
        
        res_per = res_fract*100
        res_per_square = res_per
        max_vals = np.argmax(res_per_square, axis = 0)
        res_per_square[max_vals, range(len(max_vals))] = 0
        
        weighted = res_per_square*np.sqrt(np.abs(ydata+1))
        final_res.append(np.sum(weighted**2))
    return np.sum(final_res)

def solve(y_0,*ki):
    ############ runge kutta 4th order ODE solver ###############
    t_0 = 0
    t_final = rxntime[num_analyze]
    dt = t_final/25
    ts = [t_0]
    ys = [y_0]
    y = y_0
    t = t_0 
    if len(ki) == 1:
        ki = ki[0]
    ki = np.array(ki)/1e10
    while t < t_final:
        # Solving with Runge-Kutta
        k1 = np.array(f_jit(t, y, *ki))
        k2 = np.array(f_jit(t + dt/2, y + k1*dt/2,*ki))
        k3 = np.array(f_jit(t + dt/2, y + k2*dt/2,*ki))
        k4 = np.array(f_jit(t + dt, y + k3*dt,*ki))
        y = y + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    
        # Increasing t
        t = t + dt
    
        # Appending results
        ts.append(t)
        ys.append(y)

    return np.array(ys)[-1,:,:].transpose()
    
def mapval_toloop(param_vals, l_bound_temp2, h_bound_temp2,param_index_temp, gofargs_temp,best_fit, j):
    ############## performs a best fit with a specific parameter set to param_vals[j] ################
    
    l_bound_temp2[param_index_temp] = param_vals[j]*0.999
    h_bound_temp2[param_index_temp] = param_vals[j]*1.001
    
    k_bounds = sp.optimize.Bounds(l_bound_temp2,h_bound_temp2)
    best_fit[param_index_temp] = param_vals[j] 
    
    res_temp = sp.optimize.differential_evolution(getgof, k_bounds, args = gofargs_temp, strategy='best1bin', 
                                             maxiter=2000, popsize=1, tol=0.001, mutation= (0.1, 1.5), recombination=0.9, 
                                             seed=None, callback=None, disp=False, polish=False, init='sobol', 
                                             atol=0, updating='immediate', workers=1, constraints=nonlincon, x0=None, 
                                             integrality=None, vectorized=False)
    return res_temp.x, getgof(res_temp.x, *gofargs_temp)

def getparambounds(a):
    ############# generates a region of parameter space to search that is +/- an order of magnitude of the optimal value, with spacing clustered around the optimal value and then gradually increasing ############
    low = []
    high = []
    i = 1
    b = a
    while b < a*10:
        b = b*(1.005**i)
        high.append(b)
        i = i + 1       
    i = 1
    b = a
    while b > a/10:
        b = b*0.995**i
        low.append(b)
        i = i + 1   
    c = low + [a] + high
    c = np.array(c)
    lim_h = 20
    lim_l = 0.001
    c = np.clip(c, lim_l, lim_h)
    c.sort()
    return np.unique(c)

def search_param(best_fit, param_index, time_start,l_bounds_temp,h_bounds_temp, gofargs_temp):
    ####### parallelized mapping of the parameter space for a given parameter ##############
    param_bounds = getparambounds(best_fit[param_index])
    numsteps = len(param_bounds)
    outputs = np.zeros((len(best_fit),numsteps))
    gofs = np.zeros(numsteps)
    ares = []
    p = multiprocessing.Pool(processes = 6)
    for i in range(len(param_bounds)):
        ares.append(p.apply_async(mapval_toloop,args = (param_bounds,l_bounds_temp,h_bounds_temp,param_index,gofargs_temp, best_fit, i)))
    for i in range(len(param_bounds)):
        outputs[:,i], gofs[i] = ares[i].get()
        print(i, "is done. Evaluated", param_bounds[i]/1e10, "GOF is", gofs[i])
    p.close()
    p.join()
    plt.figure()
    plt.semilogy(param_bounds,gofs, 'o')
    print("Param", param_index, "took", time.time() - time_start)
    return gofs, outputs

def sim_monte(fit_stdev, fit_data, best_fit, fake_data_temp, neutral_con_temp, numpoints_temp, numk_temp, param_bounds_temp, iso_temp, nonlincon_temp, files_temp, kinin_temp, rois_temp, fit_params_temp):
    global constraints_new, rxntime, f_jit
    
    ydot, y, k, k_l_bounds, k_h_bounds, species_0, constraints_new, con_limits_low, con_limits_high, names, reactmap, prodmap, iso_index = getodes(kinin_temp)  
    t = sym.symbols('t')
    f_lamb = sym.lambdify((t, y) + k, ydot, "numpy")
    f_jit = nb.njit(f_lamb)
    
    rxntime = []
    for input_file in files_temp:
        if '.BatchExp' in input_file:
            rxntimes, neutral_reactants, datas, neutral_cons, initial_conss = batch_import(species_0, input_file, iso_index)
        if '.TOFs' in input_file:
            rxntimes, neutral_cons, datas, num_tofss, initial_conss = tof_import(input_file, rois_temp, names)
        rxntime.append(rxntimes)
        
    ################# generates random data from a normal distribution around the real data, then fits it and returns the data and the fit parameters ############
    sim_data = []
    for nums in range(len(fit_stdev)):
        sim_datas = np.abs(np.random.normal(loc = fit_data[nums], scale = fit_stdev[nums]*(np.abs(fake_data_temp[nums])+1), size = fake_data_temp[nums].shape))
        sim_data.append(sim_datas)
    sim_data = np.array(sim_data)
    # sim_data[:,1] = neutral_con_temp
    sim_gofargs = (numpoints_temp, numk_temp, sim_data, neutral_con_temp, iso_temp)
    sim_res = sp.optimize.differential_evolution(getgof, param_bounds_temp, args = sim_gofargs, strategy='best2bin', 
                                              maxiter=1000, popsize=fit_params_temp[0], tol=0.0001, mutation= fit_params_temp[1], recombination=fit_params_temp[2],  
                                              seed=None, callback=None, disp=False, polish=False, init='sobol', 
                                              atol=0, updating='immediate', workers=1, constraints=nonlincon_temp, x0=None, 
                                              integrality=None, vectorized=False)    
    
    return sim_res.x, sim_data

def error_analysis(best_fit, fake_data_temp, neutral_con_temp, numpoints_temp, numk_temp, ax_temp, param_bounds_temp, numsims_temp, species_0_temp, iso_temp, nonlincon_temp, files_temp, kinin_temp, rois_temp, fit_params_temp):
    global ares, num_analyze
    ############ calculating the standard deviation in the scatter of the original data around the best fit ##############
    fit_stdev = []
    fit_data = []

    initial_cons_temp_full = get_fit_initial_cons(best_fit, (fake_data_temp.shape[2],fake_data_temp.shape[1]))
    for num_analyze in range(len(fake_data_temp)):
        fit_datas = solve(initial_cons_temp_full[num_analyze], best_fit[0:numk_temp])
        for indices in iso_temp:
            fit_datas[:,indices[0]] = np.sum(fit_datas[:,indices], axis =1 )
            fit_datas[:, indices[1:]] = np.zeros([fake_data_temp[num_analyze].shape[0],len(indices[1:])])
        residual = (fit_datas - fake_data_temp[num_analyze])/(fake_data_temp[num_analyze]+0.1)
        # max_vals = np.argmax(residual[0], axis = 0)
        # residual[0][max_vals, range(len(max_vals))] = 0
        fit_stdev.append(np.std(residual, axis = 0))
        fit_data.append(fit_datas)
    ############# parallelized monte carlo simulation of the error. ####################
    sim_params = []
    ares = []
    full_sim_data = []
    numcpus = multiprocessing.cpu_count()-2
    if numcpus > 60:
        numcpus = 60
    with multiprocessing.Pool(processes = numcpus) as p:
        for loops in range(numsims_temp):
            ares.append(p.apply_async(sim_monte, args = (fit_stdev, fit_data, best_fit, fake_data_temp, neutral_con_temp, numpoints_temp, numk_temp, param_bounds_temp, iso_temp, nonlincon_temp, files_temp, kinin_temp, rois_temp, fit_params_temp)))
        
        # goody = 0
        # loopmonitor(ares,numsims_temp, goody)
        
        for loops in range(numsims_temp):
            window.event_generate("<<event1>>", when = "tail", state = int(np.clip((loops+1)/numsims_temp*100,0,100)))
            sim_res, sim_data = ares[loops].get()
            sim_params.append(sim_res)
            full_sim_data.append(sim_data)    
    
    sim_params = np.array(sim_params)
    param_stdev = np.std(sim_params, axis = 0)
    full_sim_data = np.array(full_sim_data)
    
    quartiles = np.percentile(sim_params, [25,75], axis = 0)
    k_factor = 1.5
    iqr = (quartiles[1]-quartiles[0])*k_factor
    t_fences = np.clip(np.array([quartiles[0]-iqr,quartiles[1]+iqr]),0,100)
    
    fit_low = []
    fit_high = []
    params_trunc = []
    for trunc_index, to_trunc in enumerate(sim_params.transpose()):
        indices = np.where((to_trunc > t_fences[:,trunc_index][0]) & (to_trunc < t_fences[:,trunc_index][1]))
        params_trunc.append(to_trunc[indices])
        if len(to_trunc[indices]) > 0:
            fit_low.append(np.percentile(to_trunc[indices],2.5))
            fit_high.append(np.percentile(to_trunc[indices],97.5))
        if len(to_trunc[indices]) == 0:
            fit_low.append(0.01)
            fit_high.append(10)

    new_params = []
    for to_hist in params_trunc:
        # plt.figure()
        # plt.hist(to_hist,bins = 25)
        hist, hist_bins = np.histogram(to_hist,25)
        prob_index = np.argmax(hist)
        new_params.append(np.average([hist_bins[prob_index],hist_bins[prob_index+1]]))
    best_fit = np.array(new_params)

    fit_low = np.array(fit_low)
    fit_high = np.array(fit_high) 
    
    plt.figure()
    
    ############### plotting and saving the fits #################
    neutral_con_temp_full = neutral_con_temp
    fake_data_temp_full = fake_data_temp
    for num_analyze, files in enumerate(files_temp):
        initial_cons_temp = initial_cons_temp_full[num_analyze]
        neutral_con_temp = neutral_con_temp_full[num_analyze]
        sorting_index = np.argsort(neutral_con_temp)
        fake_data_temp = fake_data_temp_full[num_analyze]
        count = 0
        
        for plt_index_temp in range(sim_data.shape[2]-1):
            plt.figure(figsize = [15, 10])
            if iso_temp != []:
                for iso in iso_temp:
                    full_sim_data[:,:,iso[0]+1] = np.sum(full_sim_data[:,:,np.array(iso)+1],axis = 2)
                    full_sim_data[:,:,np.array(iso[1:])+1] = np.zeros([full_sim_data.shape[0],full_sim_data.shape[1],len(indices[1:])])
            for plts in full_sim_data[:, num_analyze]:
                plt.semilogy(neutral_con_temp,np.delete(plts, 1, axis = 1)[:,plt_index_temp], 'o', color = 'red')
            if iso_temp == []:
                temp_plot = np.delete(solve(initial_cons_temp,fit_low[0:numk_temp]),1,axis = 1)[sorting_index]
                plt.semilogy(np.sort(neutral_con_temp),temp_plot[:,plt_index_temp], color = 'black')
                temp_plot = np.delete(solve(initial_cons_temp,fit_high[0:numk_temp]),1,axis = 1)[sorting_index]
                plt.semilogy(np.sort(neutral_con_temp),temp_plot[:,plt_index_temp], color = 'black')
                temp_plot = np.delete(fake_data_temp,1,axis = 1)[sorting_index]
                plt.semilogy(np.sort(neutral_con_temp),temp_plot[:,plt_index_temp], "o", markersize = 15)
                best = np.delete(solve(initial_cons_temp, best_fit[0:numk_temp]),1,axis=1)[sorting_index]
                plt.semilogy(np.sort(neutral_con_temp), best[:,plt_index_temp], color = "green")
            else:
                temp_plot = np.delete(solve(initial_cons_temp,fit_low[0:numk_temp]),1,axis = 1)[sorting_index]
                for indices in iso_temp:
                    temp_plot[:,indices[0]] = np.sum(temp_plot[:,indices], axis =1 )
                    temp_plot[:, indices[1:]] = np.zeros([temp_plot.shape[0],len(indices[1:])])
                plt.semilogy(np.sort(neutral_con_temp),temp_plot[:,plt_index_temp], color = 'black')
                temp_plot = np.delete(solve(initial_cons_temp,fit_high[0:numk_temp]),1,axis = 1)[sorting_index]
                temp_plot[:,indices[0]] = np.sum(temp_plot[:,indices], axis =1 )
                temp_plot[:, indices[1:]] = np.zeros([temp_plot.shape[0],len(indices[1:])])
                plt.semilogy(np.sort(neutral_con_temp),temp_plot[:,plt_index_temp], color = 'black')
                temp_plot = np.delete(fake_data_temp,1,axis = 1)[sorting_index]
                for indices in iso_temp:
                    temp_plot[:,indices[0]] = np.sum(temp_plot[:,indices], axis =1 )
                    temp_plot[:, indices[1:]] = np.zeros([temp_plot.shape[0],len(indices[1:])])
                plt.semilogy(np.sort(neutral_con_temp),temp_plot[:,plt_index_temp], "o", markersize = 15)
                best = np.delete(solve(initial_cons_temp, best_fit[0:numk_temp]),1,axis=1)[sorting_index]
                for indices in iso_temp:
                    best[:,indices[0]] = np.sum(best[:,indices], axis =1 )
                    best[:, indices[1:]] = np.zeros([best.shape[0],len(indices[1:])])
                plt.semilogy(np.sort(neutral_con_temp), best[:,plt_index_temp], color = "green")
            
            newdir = files[0:files.rfind('.')] + '/'
            save_path_temp = pathlib.Path(newdir)
            for iso in iso_temp:
                tit = ''
                if iso[0] == plt_index_temp:
                    tit_arr = np.array(species_0_temp)[iso]
                    tit = tit + tit_arr[0]
                    for st in tit_arr[1:]:
                        tit = tit + 'and' + st
                    plt.title(tit)
                    save = save_path_temp / tit
                if plt_index_temp in iso[1:]:
                    count = count + 1
                if iso[0] != plt_index_temp:
                    plt.title(species_0_temp[plt_index_temp])
                    save = save_path_temp / species_0_temp[plt_index_temp]
            if iso_temp == []:
                save = save_path_temp / species_0_temp[plt_index_temp]
                plt.title(species_0_temp[plt_index_temp])
            if np.array(iso_index).size != 0:
                if plt_index_temp not in np.array(iso_index)[:,1:]:
                    plt.savefig(save)
            if np.array(iso_index).size == 0:
                plt.savefig(save)
            plt.close()                                         
    
    ############### plotting the error on the parameter maps #################
    for err_region in range(len(best_fit)):
        if err_region < 0:
            ylow, yhigh = ax_temp[err_region].get_ylim()
            ax_temp[err_region].fill_between(np.array([best_fit[err_region]-param_stdev[err_region]*2, best_fit[err_region]+param_stdev[err_region]*2]),ylow, yhigh, color = 'red')
            
    fit_low[0:numk_temp] = fit_low[0:numk_temp]/1e10
    fit_high[0:numk_temp] = fit_high[0:numk_temp]/1e10
    return param_stdev, fit_low, fit_high, full_sim_data, sim_params, best_fit

def con_fun(params):
    con_vals = []
    for cons in constraints_new:
        con_vals.append(eval(cons))
    return np.array(con_vals)

def total_k(kvals, fit_low_temp, fit_high_temp, reactmap_temp, prodmap_temp, names_temp): 
    ############ calculating kT, product branching fractions, and groups formation and loss rate constants by relevant species #################
        kT = []
        kt_index = []
        total_index = 0
        for reactant in names_temp:
            if reactant != neutral_reactant:
                k = 0
                for index, match in enumerate(reactmap_temp):
                    if reactant == match[0]:
                        k = k + kvals[index]
                        kt_index.append(total_index)
                kT.append(k)
                total_index = total_index + 1
        
        kT_err_low = []
        kT_err_high = []
        kt_index_err = []
        total_index_err = 0 
        err_low = kvals - fit_low_temp
        err_high = fit_high_temp - kvals
        for reactant in names_temp:
            if reactant != neutral_reactant:
                k = 0
                for index, match in enumerate(reactmap_temp):
                    if reactant == match[0]:
                        k_low = k + err_low[index]**2
                        k_high = k + err_high[index]**2
                        kt_index_err.append(total_index_err)
                kT_err_low.append(np.sqrt(k_low))
                kT_err_high.append(np.sqrt(k_high))
                total_index_err = total_index_err + 1
        
        ratios = []
        for index, i in enumerate(kt_index):
            ratios.append(kvals[index]/kT[i])
            
        name_loss_formed = []
        for chem in names_temp:
            if chem != neutral_reactant:
                loss = []
                formed = []
                for index, rs in enumerate(reactmap_temp):
                    if rs[0] == chem:
                        loss.append(kvals[index])
                for index, rs in enumerate(prodmap_temp):
                    if rs[0] == chem:
                        formed.append(kvals[index])
            name_loss_formed.append([chem, loss, formed])
            
            name_loss_formed_err = []
            for chem in names_temp:
                if chem != neutral_reactant:
                    loss_err = []
                    formed_err = []
                    for index, rs in enumerate(reactmap_temp):
                        if rs[0] == chem:
                            loss_err.append(err_low[index])
                    for index, rs in enumerate(prodmap_temp):
                        if rs[0] == chem:
                            formed_err.append(err_low[index])
                name_loss_formed_err.append([chem, loss_err, formed_err])
                
        return kT, kT_err_low, kT_err_high, ratios, name_loss_formed, name_loss_formed_err

def batch_import(species_temp, files_temp, iso_temp):
    ####### imports batch.exp files and generates a table of the data and other values ##########
    f = open(files_temp)
    text = f.read()
    f.close()
    
    text_split = text.split()
    string_list = []
    for strin in text_split:
        string_list.append(strin.split(';'))
         
    for start_index in range(len(string_list)):
        if len(string_list[start_index]) > 1:
            break
    neutral_reactant_temp = string_list[start_index-1]
    rxntime_temp = float(string_list[start_index-4][0])/1000
    
    neutral_con_temp = np.array([float(a[0]) for a in string_list[start_index+1:]])
    
    cons = []
    for i in range(len(species_temp)):
        cons_temp = []
        spec_tocheck = species_temp[i]
        for j in string_list[start_index+1:]:
            detect = 0
            for h in range(len(j)):
                if j[h] == spec_tocheck and h< len(j):
                    cons_temp.append(float(j[h+1]))
                    detect = 1
            if detect == 0:
                cons_temp.append(float(0))
        cons.append(cons_temp)
    cons = np.array(cons)
    data_temp = np.insert(cons,1,neutral_con_temp,axis=0)
    initial_cons_temp = np.repeat((data_temp[:,0] + data_temp[:,-1])/2,data_temp.shape[1]).reshape(data_temp.shape)
    initial_cons_temp[1,:] = neutral_con_temp
        
    return rxntime_temp, neutral_reactant_temp, data_temp.transpose(), neutral_con_temp, initial_cons_temp

def tof_import(tofs_temp, rois_temp, species_temp):
    with open(tofs_temp) as f:
        file = f.read()
    file = file.split('\n')

    to_array = []
    neutral_cons_temp = []

    rxntime_temp = float(file[0].split('\t')[2])/1000
    num_tofs = 0
    for strin in file:
        line = []
        if 'Flow' in strin.split():
            num_tofs = num_tofs + 1
        if len(strin) == len(file[2]):
            for to_float in strin.split('\t'):
                line.append(float(to_float))
        if len(strin) > 0:
            if strin.split()[0] == file[1].split()[0]:
                neutral_cons_temp.append(float(strin.split()[-1]))
        if line != []:
            to_array.append(line)
        
    array = np.array(to_array)
    neutral_cons_temp = np.array(neutral_cons_temp)
    
    with open(rois_temp) as f:
        rois = f.read()
    rois = rois.split('\n')
    rois = rois[1:-1]

    lows_temp = []
    highs_temp = []
    names_temp = []
    for species in rois:
        species = species.split()
        if species[6] != '0':
            lows_temp.append(float(species[2]))
            highs_temp.append(float(species[3]))
            names_temp.append(species[6])
    lows_temp = np.array(lows_temp)
    highs_temp = np.array(highs_temp)

    masses = array[:,1]
    ydata = []

    for i in range(len(species_temp)):
        for j, named in enumerate(names_temp):
            if named == species_temp[i]:
                mass_index = np.where(np.logical_and(lows_temp[j]< masses, masses < highs_temp[j]))
                mass_index = mass_index[0]
                
                intensities = array[mass_index][:,2]
                
                mass_index = mass_index.reshape(num_tofs,int(mass_index.size/num_tofs)).transpose()
                intensities = intensities.reshape(num_tofs,int(intensities.size/num_tofs)).transpose()
                
                ydata.append(np.sum(intensities,0))
            
    ydata = np.array(ydata)
    ydata = ydata.transpose()

    for i, named in enumerate(species_temp):
        if named not in names_temp:
            ydata = np.insert(ydata,i,np.zeros(num_tofs),axis = 1)
    
    ydata = np.delete(ydata,1,1)
    ydata = np.insert(ydata,1,neutral_cons_temp, axis = 1)
    initial_cons_temp = np.repeat((ydata[0] + ydata[-1])/2,ydata.shape[0]).reshape(ydata.shape[1],ydata.shape[0])
    
    return rxntime_temp, neutral_cons_temp, ydata, num_tofs, initial_cons_temp

def mainfun(q_current, q_output, window):
    global rxntime, f_jit, nonlincon, time0, constraints_new, neutral_reactant, fit_low, rois, monitor_current_string
    global fit_high,numpoints,numk,data,neutral_con,iso_index, initial_cons, names, species_0, numdiffsteps
    
    if __name__ == '__main__':
        start_time = time.time()

    with open(batchin) as f:
        file = f.read()
    file_split = file.split('\n')
    
    groups = []
    for files in file_split:
        groups.append(files.split()[0])
        
    files_grouped = []
    for unique in np.unique(groups):
        temp = []
        for files in file_split:
            if unique == files.split()[0]:
                temp.append(files[files.find('"')+1:files.rfind('"')])
        files_grouped.append(temp)
    
    for files_g in files_grouped:
        for files in files_g:
            newdir = files[0:files.rfind('.')] + '/'
            save_path_temp = pathlib.Path(newdir)
            save_path_temp.mkdir(parents = True, exist_ok = True)


    if __name__ == '__main__':
        time0 = time.time() 
        ################# initial fitting ########################
        fit_params = (int(fit_pop_entry.get()), tuple(float(j) for j in fit_mutation_entry.get().strip("()").split(',')), float(fit_recomb_entry.get()))
        
        if "rois" not in globals():
            rois = ''
        ares = []
        outputs = []
        numcpus = multiprocessing.cpu_count()-2
        if numcpus > 60:
            numcpus = 60
        num_fits_init = int(initialfits_entry.get())
        tot_fits = num_fits_init * len(files_grouped)
        temp_res = []
        window.event_generate("<<event1>>", when = "tail", state = 0)
        q_current.put("Starting Global Optimizations ({})".format(tot_fits))
        window.event_generate("<<event2>>", when = "tail", state = 0)
        q_output.put("Starting Global Optimizations ({})".format(tot_fits))
        window.event_generate("<<event3>>", when = "tail", state = 0)
        with multiprocessing.Pool(processes = numcpus) as p:
            outputss = []
            for loops_index, init_loops in enumerate(np.repeat(range(len(files_grouped)),num_fits_init)):
                ares.append(p.apply_async(parallel_diff, args = (files_grouped, init_loops, kinin, rois, fit_params)))
            
            # goody = 0
            # loopmonitor(ares, tot_fits, goody)
            
            for loops_index, init_loops in enumerate(np.repeat(range(len(files_grouped)),num_fits_init)):
                window.event_generate("<<event1>>", when = "tail", state = int(np.clip((loops_index+1)/len(np.repeat(range(len(files_grouped)),num_fits_init))*100,0,100)))
                res, t_elapsed = ares[loops_index].get()
                outputss.append(res)
                temp_res.append(res.fun)
                if (loops_index+1)%num_fits_init == 0 and loops_index != 0:
                    temp_res = np.array(temp_res)
                    small_index = np.argmin(temp_res)
                    outputs.append(outputss[small_index])
                    temp_res = []
                    outputss = []
                if tot_fits == 1:
                    outputs.append(outputss[0])
                    
        q_output.put('Global Fits: {} s'.format(round(time.time()-time0,2)))
        window.event_generate("<<event3>>", when = "tail", state = 0)
        
        ################ Input Handling for error_analysis ##############################
    ydot, y, k, k_l_bounds, k_h_bounds, species_0, constraints_new, con_limits_low, con_limits_high, names, reactmap, prodmap, iso_index = getodes(kinin)
    t = sym.symbols('t')
    f_lamb = sym.lambdify((t, y) + k, ydot, "numpy")
    f_jit = nb.njit(f_lamb)
    numk = len(k)
    q_output.put('Begin Error Analysis')
    window.event_generate("<<event3>>", when = "tail", state = 0)
    for filenum, input_files in enumerate(files_grouped): 
        numdiffsteps = 0
        rxntime = []
        neutral_reactant = []
        data = []
        neutral_con = []
        initial_cons = []
        num_tofs  = []
        for input_file in input_files:
            if 'rois' not in globals():
                rois = ''
            if '.BatchExp' in input_file:
                rxntimes, neutral_reactants, datas, neutral_cons, initial_conss = batch_import(species_0, input_file, iso_index)
            if '.TOFs' in input_file:
                rxntimes, neutral_cons, datas, num_tofss, initial_conss = tof_import(input_file, rois, names)
                a = np.unique(np.array(reactmap),return_counts = True)
                neutral_reactants = a[0][np.argmax(a[1])]
                num_tofs.append(num_tofss)
            initial_conss[1] = neutral_cons
            rxntime.append(rxntimes)
            neutral_reactant.append(neutral_reactants)
            data.append(datas)
            neutral_con.append(neutral_cons)
            initial_cons.append(initial_conss)
        data = np.array(data)
        
        initial_con_0 = []
        for initial_c in initial_cons:
            initial_con_0_temp = (initial_c[:,0] + initial_c[:,-1])/2
            initial_con_0_temp[1] = 0
            initial_con_0.append(initial_con_0_temp)
        numpoints = initial_c.shape[1]

        con_l_bounds = []
        con_h_bounds = []
        l_bounds = []
        h_bounds = []
        param_bounds = []
        for j, initial_con_0_loop in enumerate(initial_con_0):
            con_l_bound = []
            con_h_bound = []
            for i, con in enumerate(initial_con_0[j]):
                if con == 0:
                    initial_con_0[j][i] = 0.001
                    con_l_bound.append(0)
                    con_h_bound.append(0.002)
                else:
                    con_l_bound.append(con*0.5)
                    con_h_bound.append(con*2)
            con_l_bound = np.array(con_l_bound)
            con_h_bound = np.array(con_h_bound)
            con_l_bounds.append(con_l_bound)
            con_h_bounds.append(con_h_bound)
        
        l_bounds = np.concatenate((k_l_bounds*1e10,con_l_bounds[0]))
        h_bounds = np.concatenate((k_h_bounds*1e10,con_h_bounds[0]))
        if len(con_l_bounds) > 1:
            for i in range(len(con_l_bounds[1:])):
                l_bounds = np.concatenate((l_bounds,con_l_bounds[i+1]))
                h_bounds = np.concatenate((h_bounds,con_h_bounds[i+1]))
        param_bounds = sp.optimize.Bounds(l_bounds,h_bounds)
        
        lb = np.array(con_limits_low)*1e10
        ub = np.array(con_limits_high)*1e10
        nonlincon = sp.optimize.NonlinearConstraint(con_fun, lb, ub)
        
        if __name__ == '__main__':
            ################ parameter mapping ###################
            # gofargs = (numpoints, numk, data, neutral_con, iso_index)
            # full_gofs = []
            # full_outputs = []
            ax = []
            # for u in range(len(outputs[filenum].x)):
            #     if u < 0:
            #         temp1, temp2 = search_param(outputs[filenum].x,u,time0,l_bounds,h_bounds,gofargs)
            #         full_gofs.append(temp1)
            #         full_outputs.append(temp2)
            #         ax.append(plt.gca())
            #         print("Param", u, "is done.", time.time()-time0, 's have elapsed')
        
            ################ Error Analysis ########################
            window.event_generate("<<event1>>", when = "tail", state = 0)
            monitor_string = "{}/{}: Error Analysis".format(filenum+1, len(files_grouped)) #will need better formating
            q_current.put(monitor_string)
            window.event_generate("<<event2>>", when = "tail", state = 0)
            numsims = int(sims_entry.get())
            param_stdev, fit_low, fit_high, full_sim_data, sim_params, globalfit = error_analysis(outputs[filenum].x, data, neutral_con, numpoints, numk, ax, param_bounds, numsims, species_0, iso_index, nonlincon, input_files, kinin, rois, fit_params)
            
            # new_params = []
            # for to_hist in sim_params.transpose():
            #     hist, hist_bins = np.histogram(to_hist,25)
            #     prob_index = np.argmax(hist)
            #     new_params.append(np.average([hist_bins[prob_index],hist_bins[prob_index+1]]))
            # globalfit = np.array(new_params)
            
            
            q_output.put('Fit {} Error Analysis Time: {} s'.format(filenum+1, round(time.time()-time0,2)))
            window.event_generate("<<event3>>", when = "tail", state = 0)
            # globalfit = outputs[filenum].x 
            globalfit[0:numk] = globalfit[0:numk]/1e10
            combined_out = np.array([globalfit, fit_low, fit_high]).transpose()
            kT, kT_err_low, kT_err_high, ratios, nlf, nlf_err = total_k(globalfit, fit_low, fit_high, reactmap, prodmap, names)
            
            ####################### formating output file #########################
            for num_analyze, input_file in enumerate(input_files):
                outstr = 'parameter\tvalue\tlow\thigh'
                num_cons = int((combined_out.shape[0]-numk)/len(input_files))
                for index, co in enumerate(combined_out):
                    if index < numk:
                        outstr = outstr + '\n' + str(k[index])
                        for stri2 in co:
                            outstr = outstr + '\t' + str(stri2)
                    if index >= numk:
                        index = index - num_analyze*num_cons - numk
                        if index < len(names) and index >= 0:
                            outstr = outstr + '\n' + names[index]
                            for stri2 in co:
                                outstr = outstr + '\t' + str(stri2)
                        
                outstr = outstr + '\n' + '\n'
                
                outstr = outstr + 'species\tkT\tkT_err'
                for index, total in enumerate(kT):
                    outstr = outstr + '\n' + names[index] + '\t' + str(total) + '\t' + str(kT_err_low[index]) + '\t' + str(kT_err_high[index])
                    
                outstr = outstr + '\n' + '\n'
                
                outstr = outstr + 'Product\tFraction\n'
                for index, chem in enumerate(prodmap):
                    outstr = outstr + chem[0] + '\t' + str(ratios[index]) +'\n'
                
                outstr = outstr + '\n'
                
                outstr = outstr + 'species\tloss_rate\terr\n'
                for index, forstr in enumerate(nlf):
                    outstr = outstr + forstr[0]
                    for ind, vals in enumerate(forstr[1]):
                        outstr = outstr + '\t' + str(vals) + '\t' + str(nlf_err[index][1][ind])
                    outstr = outstr + '\n'
                
                outstr = outstr + '\n'
                
                outstr = outstr + 'species\tformed_rate\terr\n'
                for index, forstr in enumerate(nlf):
                    outstr = outstr + forstr[0]
                    for ind, vals in enumerate(forstr[2]):
                        outstr = outstr + '\t' + str(vals) + '\t' + str(nlf_err[index][2][ind])
                    outstr = outstr + '\n'
                
                outstr = outstr + '\n'
                
                outstr = outstr + 'Sim Params\n'
                for s_p in sim_params:
                    for item in s_p:
                        outstr = outstr + str(item) + '\t'
                    outstr = outstr + '\n'
                
                outputname = input_file[-17:-9] + '_output.txt'
                newdir = input_file[0:input_file.rfind('.')] + '/'
                save_path = pathlib.Path(newdir)
                save_path.mkdir(parents=True, exist_ok=True)
                save = save_path /  outputname
                f = open(save, 'w')
                f.write(outstr)
                f.close()
            
    if __name__ == '__main__':
        q_output.put('Total Time: {}'.format(round(time.time() - start_time,2)))
        window.event_generate("<<event3>>", when = "tail", state = 0)
        window.event_generate("<<event4>>", when = 'tail', state = 0)

def parallel_diff(files_grouped, filenum, kinin, rois, fit_params_temp):
    global rxntime, f_jit, nonlincon, time0, constraints_new, neutral_reactant, fit_low, monitor_current_string
    global fit_high,numpoints,numk,data,neutral_con,iso_index, initial_cons, names, species_0, numdiffsteps
    t_start = time.time()
    files = files_grouped[filenum]
    
    ydot, y, k, k_l_bounds, k_h_bounds, species_0, constraints_new, con_limits_low, con_limits_high, names, reactmap, prodmap, iso_index = getodes(kinin)
    t = sym.symbols('t')
    f_lamb = sym.lambdify((t, y) + k, ydot, "numpy")
    f_jit = nb.njit(f_lamb)
    numk = len(k)
    
    numdiffsteps = 0
    rxntime = []
    neutral_reactant = []
    data = []
    neutral_con = []
    initial_cons = []
    num_tofs  = []
    for input_file in files:
        if '.BatchExp' in input_file:
            rxntimes, neutral_reactants, datas, neutral_cons, initial_conss = batch_import(species_0, input_file, iso_index)
        if '.TOFs' in input_file:
            rxntimes, neutral_cons, datas, num_tofss, initial_conss = tof_import(input_file, rois, names)
            a = np.unique(np.array(reactmap),return_counts = True)
            neutral_reactants = a[0][np.argmax(a[1])]
            num_tofs.append(num_tofss)
        initial_conss[1] = neutral_cons
        rxntime.append(rxntimes)
        neutral_reactant.append(neutral_reactants)
        data.append(datas)
        neutral_con.append(neutral_cons)
        initial_cons.append(initial_conss)
    data = np.array(data)
    
    initial_con_0 = []
    for initial_c in initial_cons:
        initial_con_0_temp = (initial_c[:,0] + initial_c[:,-1])/2
        initial_con_0_temp[1] = 0
        initial_con_0.append(initial_con_0_temp)
    numpoints = initial_c.shape[1]

    con_l_bounds = []
    con_h_bounds = []
    l_bounds = []
    h_bounds = []
    param_bounds = []
    for j, initial_con_0_loop in enumerate(initial_con_0):
        con_l_bound = []
        con_h_bound = []
        for i, con in enumerate(initial_con_0[j]):
            if con == 0:
                initial_con_0[j][i] = 0.001
                con_l_bound.append(0)
                con_h_bound.append(0.002)
            else:
                con_l_bound.append(con*0.5)
                con_h_bound.append(con*2)
        con_l_bound = np.array(con_l_bound)
        con_h_bound = np.array(con_h_bound)
        con_l_bounds.append(con_l_bound)
        con_h_bounds.append(con_h_bound)
    
    l_bounds = np.concatenate((k_l_bounds*1e10,con_l_bounds[0]))
    h_bounds = np.concatenate((k_h_bounds*1e10,con_h_bounds[0]))
    if len(con_l_bounds) > 1:
        for i in range(len(con_l_bounds[1:])):
            l_bounds = np.concatenate((l_bounds,con_l_bounds[i+1]))
            h_bounds = np.concatenate((h_bounds,con_h_bounds[i+1]))
    param_bounds = sp.optimize.Bounds(l_bounds,h_bounds)
    gofargs = (numpoints, numk, data, neutral_con, iso_index)
    
    lb = np.array(con_limits_low)*1e10
    ub = np.array(con_limits_high)*1e10
    nonlincon = sp.optimize.NonlinearConstraint(con_fun, lb, ub)
    
    res = sp.optimize.differential_evolution(getgof, param_bounds, args = gofargs, strategy='best2bin', 
                                              maxiter=2000, popsize=fit_params_temp[0], tol=0.0001, mutation= fit_params_temp[1], recombination=fit_params_temp[2], 
                                              seed=None, callback=None, disp= False, polish=False, init='sobol', 
                                              atol=0, updating='immediate', workers=1, constraints=nonlincon, x0=None, 
                                              integrality=None, vectorized=False)
    
    fit_initial_cons = get_fit_initial_cons(res, (datas.shape[1], datas.shape[0]))
        
    for num_analyze in range(len(data)):
        plt.figure(figsize = [15, 10])
        plt.semilogy(neutral_con[num_analyze],np.delete(data[num_analyze],1,axis = 1), "o")
        plt.semilogy(neutral_con[num_analyze],np.delete(solve(fit_initial_cons[num_analyze],res.x[0:numk]),1,axis = 1))
        
        if type(files) == list:
            newdir = files[num_analyze][0:files[num_analyze].rfind('.')] + '/'
            save_path = pathlib.Path(newdir)
            save_path.mkdir(parents=True, exist_ok=True)
            save = save_path / "total_fig"
            plt.savefig(save)
            plt.close()
        if type(files) != list:
            newdir = files[0:files.rfind('.')] + '/'
            save_path = pathlib.Path(newdir)
            save_path.mkdir(parents=True, exist_ok=True)
            save = save_path / "total_fig"
            plt.savefig(save)
            plt.close()
    
    return res, time.time()-t_start
    
def getbatchin(state):
    global batchin, loadedfiles_b
    
    if state == 0:
        if 'batchin' in globals():
            del batchin
    if 'batchin' not in globals():
        batchin = filedialog.askopenfilename(title = 'Load .BATCHIN', filetypes = [('BATCHIN', '*.BATCHIN')])
    loadedfiles_b = loadedfiles_b +1
    
    with open(batchin) as f:
        text = f.read()
    text = text.split('\n')
    roi = 0
    for line in text:
        if '.TOFs' in line:
            roi = roi +1
    if roi != 0:
        rois_button['state'] = 'active'
        window.update()
    
    if roi == 0:
        kinin_button['state'] = 'active'
        window.update()
    
    now = datetime.datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    monitor_output.delete('1.0', 'end')
    monitor_output.insert('1.0', dt_string + '\n' + batchin + '\n')

def create_batchin():
    global batchin
    to_batchin = filedialog.askopenfilenames(title = 'Select Data Files', 
                                             filetypes = [('.BatchExp', '*.BatchExp'), ('.TOFS', '*.TOFS')])
    temp = to_batchin[0]
    for index, char in enumerate(temp):
        if char == '/':
            temp = temp[0:index] + '\\' + temp[index+1:]
    temp = '1' + ' ' '"' + temp + '"'
    for line_num, line in enumerate(to_batchin[1:]):
        for index, char in enumerate(line):
            if char == '/':
                line = line[0:index] + '\\' + line[index+1:]
        temp = temp + '\n' + str(line_num+2) + ' ' +'"' + line + '"'
    with filedialog.asksaveasfile(title = 'Save .BATCHIN', filetypes = [('.BATCHIN', '*.BATCHIN')], defaultextension = '.BATCHIN', mode = 'w') as file:
        file.truncate(0)
        file.write(temp)
        batchin = file.name
    getbatchin(1)

def getrois():
    global rois
    rois = filedialog.askopenfilename(title = 'Load .ROIs', filetypes = [('ROIS', '*.ROIs')])
    kinin_button['state'] = 'active'
    monitor_output.insert('end', rois + '\n')
    window.update()

def getkinin(state):
    global kinin, loadedfiles_k, reactions_entries, reactions_k_entries, reactions_low_entries, reactions_high_entries
    global con_low, con_high, con_con, iso
    global text
    
    if state == 0:
        kinin = filedialog.askopenfilename(title = 'Load .KININ', filetypes = [('KININ', '*.KININ')])
    loadedfiles_k = loadedfiles_k +1
    if loadedfiles_b >= 1 and loadedfiles_k >= 1:
        # start_button['state'] = 'active'
        start_button['state'] = 'active'
        window.update()
    with open(kinin) as f:
        text = f.read()
    text = text.split('\n')
    reactions_entries = []
    reactions_k_entries = []
    reactions_low_entries = []
    reactions_high_entries = []
    
    clear()
    
    for i, line in enumerate(text):
        if line == 'Conditions':
            break
        if line == '':
            break
        vert_index = line.find('|')
        temp_rxn = tk.Entry(master = reactions_frame, justify = 'center', width = 40)
        temp_rxn.insert(0,line[0:vert_index-1])
        reactions_entries.append(temp_rxn)
        temp_rxn.grid(column = 0, row = 2+i)
        reactions_frame.rowconfigure(i+2, weight = 1)
        
        split_line = line.split(' ')

        temp_k = tk.Entry(master = reactions_frame, justify = 'center', width = 5)
        temp_k.insert(0,split_line[6])
        temp_k.grid(column = 1, row = i+2)
        reactions_k_entries.append(temp_k)
        
        temp_low = tk.Entry(master = reactions_frame, justify = 'center', width = 10)
        temp_low.insert(0,split_line[7])
        temp_low.grid(column = 2, row = i+2)
        reactions_low_entries.append(temp_low)
        
        temp_high = tk.Entry(master = reactions_frame, justify = 'center', width = 10)
        temp_high.insert(0, split_line[8])
        temp_high.grid(column = 3, row = i+2)
        reactions_high_entries.append(temp_high)
                
    text = np.array(text)
    con_in = np.where(text == 'Conditions')[0][0]
    iso_in = np.where(text == 'Iso')[0][0]
    con_low = []
    con_high = []
    con_con = []
    for i,line in enumerate(text[con_in+1:iso_in-1]):        
        low = line.find('<')
        high = line.rfind('<')
        
        con_low_temp = tk.Entry(master = conditions_frame, justify = 'center', width = 10)
        con_low_temp.insert(0, line[0:low-1])
        con_low_temp.grid(column = 0, row = i+2)
        con_low.append(con_low_temp)
        
        con_high_temp = tk.Entry(master = conditions_frame, justify = 'center', width = 10)
        con_high_temp.insert(0, line[high+1:-2])
        con_high_temp.grid(column = 2, row = i+2)
        con_high.append(con_high_temp)
        
        con_con_temp = tk.Entry(master = conditions_frame, justify = 'center', width = 20)
        con_con_temp.insert(0, line[low+1:high-1])
        con_con_temp.grid(column = 1, row = i+2)
        con_con.append(con_con_temp)

        conditions_frame.rowconfigure(i+1, weight = 1)
        
    iso = []
    if text[-1] == '':
        last_line = len(text)-1
    else: 
        last_line = len(text)
    for i, line in enumerate(text[iso_in+1:last_line]):
        iso_temp = tk.Entry(master = iso_frame, justify = 'center', width = 50)
        iso_temp.insert(0,line[0:-2])
        iso_temp.grid(column = 0, row = i+2, columnspan = 2)
        iso.append(iso_temp)
        
        iso_frame.rowconfigure(i+2, weight = 1)
    monitor_output.insert('end', kinin + '\n')
    window.update()
        
def start_mainfun():
    global t1, q_current, q_output, window
    q_current = queue.Queue()
    q_output = queue.Queue()
    start_button['state'] = 'disabled'
    threading.Thread(target=mainfun, args=(q_current, q_output, window), daemon = True).start()
    
def add_remove(function, where):
    # function = 0 is add, function = 1 is remove. where = 0 is reactions, 1 is conditions, 2 is iso
    global reactions_entries, reactions_k_entries, reactions_low_entries, reactions_high_entries
    global reactions_frame, conditions_frame, iso_frame
    if function == 0 and where == 0:
        i = reactions_frame.grid_size()[1]
        temp_rxn = tk.Entry(master = reactions_frame, justify = 'center', width = 40)
        reactions_entries.append(temp_rxn)
        temp_rxn.grid(column = 0, row = i)

        temp_k = tk.Entry(master = reactions_frame, justify = 'center', width = 5)
        temp_k.grid(column = 1, row = i)
        reactions_k_entries.append(temp_k)
        
        temp_low = tk.Entry(master = reactions_frame, justify = 'center', width = 10)
        temp_low.grid(column = 2, row = i)
        reactions_low_entries.append(temp_low)
        
        temp_high = tk.Entry(master = reactions_frame, justify = 'center', width = 10)
        temp_high.grid(column = 3, row = i)
        reactions_high_entries.append(temp_high)
        
        reactions_frame.rowconfigure(i, weight = 1)
        
    if function == 0 and where == 1:
        i = conditions_frame.grid_size()[1]
        con_low_temp = tk.Entry(master = conditions_frame, justify = 'center', width = 10)
        con_low_temp.grid(column = 0, row = i)
        con_low.append(con_low_temp)
        
        con_con_temp = tk.Entry(master = conditions_frame, justify = 'center', width = 20)
        con_con_temp.grid(column = 1, row = i)
        con_con.append(con_con_temp)
        
        con_high_temp = tk.Entry(master = conditions_frame, justify = 'center', width = 10)
        con_high_temp.grid(column = 2, row = i)
        con_high.append(con_high_temp)

        conditions_frame.rowconfigure(i, weight = 1)
    
    if function == 0 and where == 2:
        i = iso_frame.grid_size()[1]
        iso_temp = tk.Entry(master = iso_frame, justify = 'center', width = 50)
        iso_temp.grid(column = 0, row = i, columnspan = 2)
        iso.append(iso_temp)
        
        iso_frame.rowconfigure(i, weight = 1)
    
    stop = 0
    if function == 1:
        xs = []
        if where == 0:
            frame = reactions_frame
            if len(frame.winfo_children()) == 6:
                stop = 1
        if where == 1:
            frame = conditions_frame
            if len(frame.winfo_children()) == 5:
                stop = 1
        if where == 2:
            frame = iso_frame
            if len(frame.winfo_children()) == 3:
                stop = 1
        if stop == 0:
            frame_shape = frame.grid_size()
            for rows in range(frame_shape[1]-2):
                button_x = tk.Button(master = frame, text = 'x', command = lambda rows = rows: remove_button(rows, where, xs))
                button_x.grid(column = frame_shape[0]+1, row = rows+2)
                xs.append(button_x)
            frame.columnconfigure(frame_shape[0]+1, weight = 1)
        
    window.update()
    
def remove_button(row_num, where, xs):
    global reactions_entries, reactions_k_entries, reactions_low_entries, reactions_high_entries, con_low, con_high, con_con, iso
    global reactions_frame, conditions_frame, iso_frame
    
    if where == 0:
        del reactions_entries[row_num]
        del reactions_k_entries[row_num]
        del reactions_low_entries[row_num]
        del reactions_high_entries[row_num]
        refill_reactions_frame()
        
    if where == 1:
        del con_low[row_num]
        del con_high[row_num]
        del con_con[row_num]
        refill_conditions_frame()
        
    if where == 2:
        del iso[row_num]
        refill_iso_frame()
        
    for i in xs:
        i.destroy()
    window.update()

def reorder(frame_num):
    global move, move_state
    if frame_num == 0:
        frame = reactions_frame
    if frame_num == 1:
        frame = conditions_frame
    if frame_num == 2:
        frame = iso_frame
    
    column_num = frame.grid_size()[0]
    move = []

    if move_state == 1:
        for widget in move:
            widget.destroy()
        move_state = 0          
    else:
        for i in range(2,frame.grid_size()[1]):
            move_frame = tk.Frame(master = frame)
            up_button = tk.Button(master = move_frame, text = '↑', command = lambda i = i: move_it(i,i-1,frame_num))
            down_button = tk.Button(master = move_frame, text = '↓', command = lambda i = i: move_it(i+1,i,frame_num))
            
            up_button.grid(column = 0, row = 0)
            down_button.grid(column = 1, row = 0)
            move_frame.grid(column = column_num, row = i)
            move.append(move_frame)
        move_state = 1
            
    window.update()
            
def move_it(row_a, row_b,frame_num):
    row_a = row_a - 2
    row_b = row_b - 2
    global reactions_entries, reactions_k_entries, reactions_low_entries, reactions_high_entries, con_low, con_high, con_con, iso
    global reactions_frame, conditions_frame, iso_frame, move_state #made move state a global to clear an error
    
    if frame_num == 0:     
        if row_a >= reactions_frame.grid_size()[1]-2:
            row_a = 0
        if row_b >= reactions_frame.grid_size()[1]-2:
            row_b = 0
        
        temp_react_a = reactions_entries[row_a]
        temp_react_b = reactions_entries[row_b]
        reactions_entries[row_a] = temp_react_b
        reactions_entries[row_b] = temp_react_a
        
        temp_k_a = reactions_k_entries[row_a]
        temp_k_b = reactions_k_entries[row_b]
        reactions_k_entries[row_a] = temp_k_b
        reactions_k_entries[row_b] = temp_k_a
        
        temp_h_a = reactions_high_entries[row_a]
        temp_h_b = reactions_high_entries[row_b]
        reactions_high_entries[row_a] = temp_h_b
        reactions_high_entries[row_b] = temp_h_a
        
        temp_l_a = reactions_low_entries[row_a]
        temp_l_b = reactions_low_entries[row_b]
        reactions_low_entries[row_a] = temp_l_b
        reactions_low_entries[row_b] = temp_l_a
        
        refill_reactions_frame()
        move_state = 0
        reorder(frame_num)
        
    if frame_num == 1: 
        if row_a >= conditions_frame.grid_size()[1]-2:
            row_a = 0
        if row_b >= conditions_frame.grid_size()[1]-2:
            row_b = 0
        temp_con_a = con_con[row_a]
        temp_con_b = con_con[row_b]
        con_con[row_a] = temp_con_b
        con_con[row_b] = temp_con_a
        
        temp_l_a = con_low[row_a]
        temp_l_b = con_low[row_b]
        con_low[row_a] = temp_l_b
        con_low[row_b] = temp_l_a
        
        temp_h_a = con_high[row_a]
        temp_h_b = con_high[row_b]
        con_high[row_a] = temp_h_b
        con_high[row_b] = temp_h_a
        
        refill_conditions_frame()
        move_state = 0
        reorder(frame_num)
        
    if frame_num == 2:
        if row_a >= iso_frame.grid_size()[1]-2:
            row_a = 0
        if row_b >= iso_frame.grid_size()[1]-2:
            row_b = 0
        temp_iso_a = iso[row_a]
        temp_iso_b = iso[row_b]
        iso[row_a] = temp_iso_b
        iso[row_b] = temp_iso_a
            
        refill_iso_frame()
        move_state = 0
        reorder(frame_num)
        
    window.update()
    
def clear():
    reactions_frame.destroy()
    conditions_frame.destroy()
    iso_frame.destroy()
    create_reactions_frame()
    create_conditions_frame()
    create_iso_frame()
     
def create_reactions_frame():
    global reactions_frame, reactions_add, reactions_remove
    reactions_frame = tk.Frame(master = window, highlightbackground="black", highlightthickness=1)
    
    reactions_reactions_label = tk.Label(master = reactions_frame, text = 'Reactions')
    reactions_k_label = tk.Label(master = reactions_frame, text = 'k')
    reactions_lowlimit_label = tk.Label(master = reactions_frame, text = 'Low')
    reactions_highlimit_label = tk.Label(master = reactions_frame, text = 'High')
    
    reactions_arr_frame = tk.Frame(master = reactions_frame)
    reactions_add = tk.Button(master = reactions_arr_frame, text = 'Add', command = lambda: add_remove(0,0))
    reactions_remove = tk.Button(master = reactions_arr_frame, text = 'Remove', command = lambda: add_remove(1,0))
    reactions_reorder = tk.Button(master = reactions_arr_frame, text = "Reorder", command = lambda: reorder(0))
    
    reactions_frame.grid(column = 0, row = 1, sticky = "NESW", rowspan = 2)
    
    reactions_frame.grid_rowconfigure(0, weight = 1)
    reactions_arr_frame.grid(column = 0, columnspan = 4, row = 0, sticky = "NESW")
    reactions_arr_frame.rowconfigure(0, weight = 1)
    reactions_add.grid(column = 0, row = 0)
    reactions_arr_frame.columnconfigure(0, weight = 1)
    reactions_remove.grid(column = 1, row = 0)
    reactions_arr_frame.columnconfigure(1, weight = 1)
    reactions_reorder.grid(column = 2, row = 0)
    reactions_arr_frame.columnconfigure(2, weight = 1)
    
    reactions_frame.rowconfigure(1, weight = 1)
    reactions_reactions_label.grid(column = 0, row = 1)
    reactions_frame.grid_columnconfigure(0, weight=4)
    reactions_k_label.grid(column = 1, row = 1)
    reactions_frame.grid_columnconfigure(1, weight=2)
    reactions_lowlimit_label.grid(column = 2, row = 1)
    reactions_frame.grid_columnconfigure(2, weight=2)
    reactions_highlimit_label.grid(column = 3, row = 1)
    reactions_frame.grid_columnconfigure(3, weight=2)
    
def create_conditions_frame():
    global conditions_frame, conditions_add, conditions_remove
    conditions_frame = tk.Frame(master = window, highlightbackground="black", highlightthickness=1)
    
    conditions_conditions_label = tk.Label(master = conditions_frame, text = 'Conditions')
    conditions_low_label = tk.Label(master = conditions_frame, text = 'Low')
    conditions_high_label = tk.Label(master = conditions_frame, text = 'High')
    
    conditions_arr_frame = tk.Frame(master = conditions_frame)
    conditions_add = tk.Button(master = conditions_arr_frame, text = 'Add', command = lambda: add_remove(0,1))
    conditions_remove = tk.Button(master = conditions_arr_frame, text = 'Remove', command = lambda: add_remove(1,1))
    conditions_reorder = tk.Button(master = conditions_arr_frame, text = "Reorder", command = lambda: reorder(1))
    
    conditions_frame.grid(column = 1, row = 1, sticky = "NESW")
    
    conditions_frame.grid_rowconfigure(0, weight = 1)
    conditions_arr_frame.grid(column = 0, columnspan = 3, row = 0, sticky = "NESW")
    conditions_add.grid(column = 0, row = 0)
    conditions_remove.grid(column = 1, row = 0)
    conditions_reorder.grid(column = 2, row = 0)
    conditions_arr_frame.rowconfigure(0, weight = 1)
    conditions_arr_frame.columnconfigure(0, weight = 1)
    conditions_arr_frame.columnconfigure(1, weight = 1)
    conditions_arr_frame.columnconfigure(2, weight = 1)
    
    conditions_frame.rowconfigure(1, weight = 1)
    conditions_low_label.grid(column = 0, row = 1)
    conditions_frame.columnconfigure(0, weight = 1)
    conditions_conditions_label.grid(column = 1, row = 1)
    conditions_frame.columnconfigure(1, weight = 2)
    conditions_high_label.grid(column = 2, row = 1)
    conditions_frame.columnconfigure(2, weight = 1)
    
def create_iso_frame():
    global iso_frame, iso_add, iso_remove
    iso_frame = tk.Frame(master= window, highlightbackground="black", highlightthickness=1)

    iso_iso_label = tk.Label(master = iso_frame, text = 'Isobaric Species')
    iso_arr_frame = tk.Frame(master = iso_frame)
    iso_add = tk.Button(master = iso_arr_frame, text = 'Add', command = lambda: add_remove(0,2))
    iso_remove = tk.Button(master = iso_arr_frame, text = 'Remove', command = lambda: add_remove(1,2))
    iso_reorder = tk.Button(master = iso_arr_frame, text = 'Reorder', command = lambda: reorder(2))
    
    iso_frame.grid(column = 1, row = 2, sticky = "NESW")
    
    iso_frame.rowconfigure(0, weight = 1)
    iso_arr_frame.grid(column = 0, row = 0, sticky = "NESW")
    iso_add.grid(column = 0, row = 0)
    iso_remove.grid(column = 1, row = 0)
    iso_reorder.grid(column = 2, row = 0)
    iso_arr_frame.rowconfigure(0 , weight = 1)
    iso_arr_frame.columnconfigure(0, weight = 1)
    iso_arr_frame.columnconfigure(1, weight = 1)
    iso_arr_frame.columnconfigure(2, weight = 1)
    
    iso_frame.rowconfigure(1, weight = 1)
    iso_iso_label.grid(column = 0, row = 1)
    iso_frame.columnconfigure(0, weight = 1)
    
def savekinin():
    global reactions_entries, reactions_k_entries, reactions_low_entries, reactions_high_entries, con_low, con_high, con_con, iso
    global reactions_frame, conditions_frame, iso_frame
    global kinin
    
    kin_str = ''
    for i in range(len(reactions_entries)):
        kin_str = kin_str + reactions_entries[i].get() + ' | ' + reactions_k_entries[i].get() + ' ' + reactions_low_entries[i].get() + ' ' + reactions_high_entries[i].get() +'\n'
    
    kin_str = kin_str + '\n' + 'Conditions\n'
    
    for i in range(len(con_con)):
        kin_str = kin_str + con_low[i].get() + ' < ' + con_con[i].get() + ' < ' + con_high[i].get() + ' |\n'
        
    kin_str = kin_str + '\n' + 'Iso\n'
    
    for widget in iso:
        kin_str = kin_str + widget.get() + ' |\n'
    with filedialog.asksaveasfile(title = 'Save .KININ', filetypes = [('KININ', '*.KININ')], defaultextension = '.KININ', mode = 'w') as file:
        file.truncate(0)
        file.write(kin_str)
        kinin = file.name
    getkinin(1)
      
def refill_reactions_frame():
    global reactions_entries, reactions_k_entries, reactions_low_entries, reactions_high_entries, con_low, con_high, con_con, iso
    global reactions_frame, conditions_frame, iso_frame
    reactions = []
    rk = []
    rhigh = []
    rlow = []
    num_entries = len(reactions_entries)
    for i in range(num_entries):
        reactions.append(reactions_entries[i].get())
        rk.append(reactions_k_entries[i].get())
        rhigh.append(reactions_high_entries[i].get())
        rlow.append(reactions_low_entries[i].get())
        
    reactions_frame.destroy()
    create_reactions_frame()
    
    reactions_entries = []
    reactions_k_entries = []
    reactions_high_entries = []
    reactions_low_entries = []
    
    for i in range(num_entries):
        temp_rxn = tk.Entry(master = reactions_frame, justify = 'center', width = 40)
        temp_rxn.insert(0,reactions[i])
        reactions_entries.append(temp_rxn)
        temp_rxn.grid(column = 0, row = 2+i)
        reactions_frame.rowconfigure(i+2, weight = 1)

        temp_k = tk.Entry(master = reactions_frame, justify = 'center', width = 5)
        temp_k.insert(0,rk[i])
        temp_k.grid(column = 1, row = i+2)
        reactions_k_entries.append(temp_k)
        
        temp_low = tk.Entry(master = reactions_frame, justify = 'center', width = 10)
        temp_low.insert(0,rlow[i])
        temp_low.grid(column = 2, row = i+2)
        reactions_low_entries.append(temp_low)
        
        temp_high = tk.Entry(master = reactions_frame, justify = 'center', width = 10)
        temp_high.insert(0, rhigh[i])
        temp_high.grid(column = 3, row = i+2)
        reactions_high_entries.append(temp_high)     

def refill_conditions_frame():
    global reactions_entries, reactions_k_entries, reactions_low_entries, reactions_high_entries, con_low, con_high, con_con, iso
    global reactions_frame, conditions_frame, iso_frame
    cc = []
    cl = []
    ch = []
    num_c = len(con_con)
    for i in range(num_c):
        cc.append(con_con[i].get())
        cl.append(con_low[i].get())
        ch.append(con_high[i].get())
        
    conditions_frame.destroy()
    create_conditions_frame()
    
    con_con = []
    con_low = []
    con_high = []
    
    for i in range(num_c):
        con_low_temp = tk.Entry(master = conditions_frame, justify = 'center', width = 10)
        con_low_temp.insert(0, cl[i])
        con_low_temp.grid(column = 0, row = i+2)
        con_low.append(con_low_temp)
        
        con_con_temp = tk.Entry(master = conditions_frame, justify = 'center', width = 20)
        con_con_temp.insert(0, cc[i])
        con_con_temp.grid(column = 1, row = i+2)
        con_con.append(con_con_temp)
        
        con_high_temp = tk.Entry(master = conditions_frame, justify = 'center', width = 10)
        con_high_temp.insert(0, ch[i])
        con_high_temp.grid(column = 2, row = i+2)
        con_high.append(con_high_temp)
        
        conditions_frame.rowconfigure(i+1, weight = 1)

def refill_iso_frame():
    global reactions_entries, reactions_k_entries, reactions_low_entries, reactions_high_entries, con_low, con_high, con_con, iso
    global reactions_frame, conditions_frame, iso_frame
    iso_t = []
    num_iso = len(iso)
    for i in range(num_iso):
        iso_t.append(iso[i].get())
    
    iso_frame.destroy()
    create_iso_frame()
    
    iso = []
    
    for i in range(num_iso):
        iso_temp = tk.Entry(master = iso_frame, justify = 'center', width = 50)
        iso_temp.insert(0,str(iso_t[i]))
        iso_temp.grid(column = 0, row = i+2, columnspan = 2)
        iso.append(iso_temp)
        
        iso_frame.rowconfigure(i+2, weight = 1)

def eventhandler(evt):
    global progress_var
    progress_var.set(evt.state)
    window.update()

def eventhandler2(evt):
    monitor_current_string.set(q_current.get())
    window.update()

def eventhandler3(evt):
    temp_text = monitor_output.get('1.0','end')
    temp_text_new = q_output.get()
    temp_text_update = temp_text + '\n' + temp_text_new
    monitor_output.delete('1.0', 'end')
    monitor_output.insert('1.0', temp_text_update)
    window.update()

def eventhandler4(evt):
    global batchin
    to_file = monitor_output.get('1.0','end')
    now = datetime.datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H-%M-%S")
    logname = batchin[0:batchin.rfind('.')] + ' ' + dt_string +'.kin_gap_log'
    with open(logname, 'w') as f:
        f.write(to_file)
    
def get_fit_initial_cons(res,data_shape):
    if type(res) == sp.optimize._optimize.OptimizeResult:
        res = res.x
    fit_initial_cons = []
    num_cons = int(len(res[numk:])/len(initial_cons))
    for i in range(len(initial_cons)):
        con0 = res[numk+i*num_cons:numk+i*num_cons+num_cons]
        in_cons = np.repeat(con0, numpoints).reshape(data_shape)
        in_cons[1] = neutral_con[i]
        fit_initial_cons.append(in_cons)
    return fit_initial_cons

def loopmonitor(ares_temp, total, goody_temp):
    if goody_temp < total:
        goody_temp = 0
        for aress in ares_temp:
            if aress.ready():
                goody_temp = goody_temp + 1
                window.event_generate("<<event1>>", when = "tail", state = int((goody_temp)/total*100))
        window.after(15000, loopmonitor,ares_temp,total, goody_temp)


#########################################################################################
#program begins here!

if __name__ == '__main__':
    multiprocessing.freeze_support()
    loadedfiles_b = 0
    loadedfiles_k = 0
    
    reactions_entries = []
    reactions_k_entries = []
    reactions_low_entries = []
    reactions_high_entries = []
    
    con_con = []
    con_low = []
    con_high = []
    
    iso = []
    
    move_state = 0
    
    window = tk.Tk()
    window.title("Kin_GAP_2b")
    window.geometry('1000x600')
    
    progress_var = tk.DoubleVar()
    monitor_current_string = tk.StringVar()
    monitor_output_string = tk.StringVar()
    
    create_reactions_frame()
    create_conditions_frame()
    create_iso_frame()
    
    buttons_frame = tk.Frame(master = window, highlightbackground="black", highlightthickness=1)
    monitor_frame = tk.Frame(master = window, highlightbackground='black', highlightthickness=1)
    # data_frame = tk.Frame(master = window, highlightbackground="black", highlightthickness=2)
    
    batchin_button = tk.Button(master = buttons_frame, text = 'Import .BATCHIN', command = lambda: getbatchin(0))
    create_batchin_button = tk.Button(master = buttons_frame, text = 'Create .BATCHIN', command = create_batchin)
    rois_button = tk.Button(master = buttons_frame, text = 'Import .ROIs', command = getrois, state = 'disabled')
    kinin_button = tk.Button(master = buttons_frame, text = 'Import .KININ', command = lambda: getkinin(0), state = 'disabled')
    start_button = tk.Button(master = buttons_frame, text = 'Start Run', command = start_mainfun, state = 'disabled')
    clear_button = tk.Button(master = buttons_frame, text = "Clear All", command = clear)
    save_kinin_button = tk.Button(master = buttons_frame, text = "Save .KININ", command = savekinin)
    
    sims_entry = tk.Entry(master = monitor_frame, width = 5, justify = 'center')
    sims_entry.insert(0,'500')
    initialfits_entry = tk.Entry(master = monitor_frame, width = 5, justify = 'center')
    initialfits_entry.insert(0,'50')
    sims_label = tk.Label(master = monitor_frame, text = 'Number of Initial Fits/ Number of Error Sims: ')
    
    fit_frame = tk.Frame(master = monitor_frame)
    fit_label = tk.Label(master = fit_frame, text = 'Fit Pop/Mutation/Recomb: ')
    fit_pop_entry = tk.Entry(master = fit_frame, width = 5, justify = 'center')
    fit_pop_entry.insert(0,'1')
    fit_mutation_entry = tk.Entry(master = fit_frame, width = 10, justify = 'center')
    fit_mutation_entry.insert(0,'(0.1,1.50)')
    fit_recomb_entry = tk.Entry(master = fit_frame, width = 5, justify = 'center')
    fit_recomb_entry.insert(0,'0.8')
    
    monitor_current_label = tk.Label(master = monitor_frame, textvariable = monitor_current_string)
    monitor_current_string.set("Nothing Yet")
    monitor_progressbar = ttk.Progressbar(master = monitor_frame, orient = "horizontal", variable = progress_var)
    monitor_output_frame = tk.Frame(master = monitor_frame)
    monitor_output = tk.Text(master = monitor_output_frame, width= 40)
    monitor_scroll = tk.Scrollbar(master = monitor_output_frame, orient = 'vertical', command = monitor_output.yview)
    monitor_output['yscrollcommand'] = monitor_scroll.set
    # data_data_label = tk.Label(master = data_frame, text = 'Data')    

    buttons_frame.grid(column = 0, row = 0, sticky = "NESW", columnspan = 3)
    monitor_frame.grid(column = 2, row = 1, rowspan = 2, sticky = "NESW")
    # data_frame.grid(column = 2, row = 1, sticky = "NESW", rowspan = 2)
    
    window.grid_rowconfigure(0, weight = 1)
    window.grid_rowconfigure(1, weight = 15)
    window.grid_rowconfigure(2, weight = 15)
    window.grid_columnconfigure(0, weight = 2)
    window.grid_columnconfigure(1, weight = 1)
    window.columnconfigure(2, weight = 1)
    # window.grid_columnconfigure(2, weight = 2)
    
    
    buttons_frame.grid_rowconfigure(0, weight=1)
    batchin_button.grid(column = 0, row = 0)
    buttons_frame.grid_columnconfigure(0, weight=1)
    create_batchin_button.grid(column = 1, row = 0)
    buttons_frame.grid_columnconfigure(1, weight=1)
    rois_button.grid(column = 2, row = 0)
    buttons_frame.grid_columnconfigure(2, weight=1)
    kinin_button.grid(column = 3, row = 0)
    buttons_frame.columnconfigure(3, weight = 1)
    
    buttons_frame.rowconfigure(1, weight = 1)
    save_kinin_button.grid(column = 0, row = 1)
    clear_button.grid(column = 1, row = 1)
    start_button.grid(column = 2, row = 1)

    monitor_frame.rowconfigure(0, weight = 1)
    monitor_frame.columnconfigure(0, weight = 1)
    monitor_frame.columnconfigure(1, weight = 1)
    monitor_frame.columnconfigure(2, weight = 1)
    sims_label.grid(row = 0, column = 0)
    initialfits_entry.grid(row = 0, column = 1)
    sims_entry.grid(row = 0, column = 2)
    monitor_frame.rowconfigure(1, weight = 1)
    fit_frame.grid(row = 1, column = 0, columnspan = 3, sticky = 'EW')
    fit_frame.columnconfigure(0, weight = 1)
    fit_frame.columnconfigure(1, weight = 1)
    fit_frame.columnconfigure(2, weight = 1)
    fit_frame.columnconfigure(3, weight = 1)
    fit_label.grid(row = 0, column = 0)
    fit_pop_entry.grid(row = 0, column = 1)
    fit_mutation_entry.grid(row = 0, column = 2)
    fit_recomb_entry.grid(row = 0 , column = 3)    
    monitor_frame.rowconfigure(2, weight = 1)
    monitor_current_label.grid(row = 2, column = 0, columnspan = 3)
    monitor_frame.rowconfigure(3, weight = 1)
    monitor_progressbar.grid(row = 2, column = 0, columnspan = 3, sticky = 'EW')
    monitor_frame.rowconfigure(4, weight = 10)
    monitor_output_frame.grid(row = 4, column = 0, columnspan = 3, sticky = "NESW")
    monitor_output_frame.rowconfigure(0, weight = 1)
    monitor_output_frame.columnconfigure(0, weight = 10)
    monitor_output_frame.columnconfigure(1, weight = 1)
    monitor_output.grid(row = 0, column = 0, sticky = "EW")
    monitor_scroll.grid(row = 0, column = 1, sticky = 'NS')
    
    # data_frame.rowconfigure(0, weight = 1)
    # data_data_label.grid(column = 0, row = 0)
    # data_frame.columnconfigure(0, weight = 1)
    
    window.bind("<<event1>>", eventhandler)
    window.bind("<<event2>>", eventhandler2)
    window.bind("<<event3>>", eventhandler3)
    window.bind("<<event4>>", eventhandler4)
    window.mainloop()