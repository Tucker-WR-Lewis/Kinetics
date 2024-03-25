# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 10:56:11 2023

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
import sys

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
            # text_con = text_out[index+1:iso_index]
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
                # if index > 0:
                #     temps.append(index+1)
                # if index == 0:
                #     temps.append(index)
                temps.append(index)
        if stri == '|':
            iso_index.append(temps)
            temps = []
    
    return ydot, y, k, low_bound, high_bound, specs, constraints_new, con_limits_low, con_limits_high, names, reactants3, products3, iso_index

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
                # neutral_cons.append(string)
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

def con_fun(params):
    con_vals = []
    for cons in constraints_new:
        con_vals.append(eval(cons))
    return np.array(con_vals)

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
            fit_ys[:,indices[0]] = np.sum(fit_ys[:,indices], axis =1 )
            fit_ys[:,indices[1:]] = np.zeros([ydata.shape[0],len(indices[1:])])
        ydata = np.delete(ydata,1,axis=1)
        
        res_abs = np.abs(fit_ys-ydata)
        res_fract = res_abs/(ydata+1) #old method, new gof_3
        
        res_per = res_fract*100
        res_per_square = res_per #added the square back in for new gof_2 and new gof_3, removed for new gof_4
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
   
def error_analysis(best_fit, fake_data_temp, neutral_con_temp, numpoints_temp, numk_temp, param_bounds_temp, numsims_temp, species_0_temp, iso_temp, nonlincon_temp, kinin_temp,numcpus_temp):
    global ares, num_analyze, max_vals, residual, ylims, sim_gofs, params_trunc, sim_params, initial_list
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
        max_vals = np.argmax(residual, axis = 0)
        residual[max_vals, range(len(max_vals))] = 0
        fit_stdev.append(np.std(residual, axis = 0))
        fit_data.append(fit_datas)
    ############# parallelized monte carlo simulation of the error. ####################
    sim_params = []
    ares = []
    full_sim_data = []
    sim_gofs = []
    with multiprocessing.Pool(processes = 11) as p:
        for loops in range(numsims_temp):
            ares.append(p.apply_async(sim_monte, args = (fit_stdev, fit_data, best_fit, fake_data_temp, neutral_con_temp, numpoints_temp, numk_temp, param_bounds_temp, iso_temp, nonlincon_temp, kinin_temp)))
            
        for loops in range(numsims_temp):
            if loops%10 == 0:
                print('{} loop done'.format(loops))
            sim_res, sim_data, sim_gof = ares[loops].get()
            sim_params.append(sim_res)
            full_sim_data.append(sim_data)   
            sim_gofs.append(sim_gof)
    
    sim_params = np.array(sim_params)
    param_stdev = np.std(sim_params, axis = 0)
    full_sim_data = np.array(full_sim_data)
    sim_gofs = np.array(sim_gofs)
    
    quartiles = np.percentile(sim_gofs, [25,75], axis = 0)
    k_factor = 1.5
    iqr = (quartiles[1]-quartiles[0])*k_factor
    t_fences = np.array([quartiles[0]-iqr,quartiles[1]+iqr])
       
    fit_low = []
    fit_high = []
    params_trunc = []
    indices = np.where(sim_gofs < t_fences[1])
    gofs_iqr = sim_gofs[indices]
    gofs_high_95 = np.percentile(gofs_iqr,95)
    indices_95 = np.where(gofs_iqr < gofs_high_95)
    gofs_iqr_95 = sim_gofs[indices_95]
    ommiteds = []
    for omitted_index, ommited in enumerate(sim_gofs):
        if ommited not in gofs_iqr_95:
            ommiteds.append(omitted_index)

    for trunc_index, to_trunc in enumerate(sim_params.transpose()):
        params_trunc.append(to_trunc[indices][indices_95])
        if len(to_trunc[indices]) > 0:
            fit_low.append(np.percentile(to_trunc[indices][indices_95],2.5))
            fit_high.append(np.percentile(to_trunc[indices][indices_95],97.5))
        if len(to_trunc[indices]) == 0:
            fit_low.append(0.01)
            fit_high.append(10)
            
    new_params = []
    for to_hist in params_trunc:
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
    neutral_con_temp_full = neutral_con_temp
    num_analyze = 0
    initial_cons_temp = initial_cons_temp_full[num_analyze]
    neutral_con_temp = neutral_con_temp_full[num_analyze]
    sorting_index = np.argsort(neutral_con_temp)
    count = 0
    
    num_species = initial_cons_temp.shape[0]
    num_neutral = initial_cons_temp.shape[1]
    
    initial_list = []
    ylims = []
    # for replot in range(2):
    #     for plt_index_temp in range(sim_data.shape[2]-1):
    #         plt.figure(figsize = [15, 10])
    #         if iso_temp != []:
    #             for iso in iso_temp:
    #                 full_sim_data[:,:,iso[0]+1] = np.sum(full_sim_data[:,:,np.array(iso)+1],axis = 2)
    #                 full_sim_data[:,:,np.array(iso[1:])+1] = np.zeros([full_sim_data.shape[0],full_sim_data.shape[1],len(indices[1:])])
            
    #         for omit_index in ommiteds:
    #             num_cons = int(len(sim_params[0][numk_temp:])/len(fake_data_temp))
    #             sim_index = [numk_temp+num_analyze*num_cons,numk_temp+num_analyze*num_cons+num_cons]
    #             initial_cons_temp = np.reshape(np.repeat(sim_params[omit_index][sim_index[0]:sim_index[1]],num_neutral),(num_species,num_neutral))
    #             initial_cons_temp[1] = neutral_con_temp
    #             temp_plot = np.delete(solve(initial_cons_temp,sim_params[omit_index][0:numk_temp]),1,axis = 1)[sorting_index][:,plt_index_temp]
    #             plt.semilogy(np.sort(neutral_con_temp),temp_plot, color = 'black', alpha = 0.5)
            
    #         for plts_index, plts in enumerate(np.array(params_trunc).transpose()):
    #             num_cons = int(len(sim_params[0][numk_temp:])/len(fake_data_temp))
    #             sim_index = [numk_temp+num_analyze*num_cons,numk_temp+num_analyze*num_cons+num_cons]
    #             initial_cons_temp = np.reshape(np.repeat(plts[sim_index[0]:sim_index[1]],num_neutral),(num_species,num_neutral))
    #             initial_cons_temp[1] = neutral_con_temp
    #             temp_plot = np.delete(solve(initial_cons_temp,plts[0:numk_temp]),1,axis = 1)[sorting_index][:,plt_index_temp]
    #             plt.semilogy(np.sort(neutral_con_temp),temp_plot, color = 'red', alpha = 0.1)
                
    #         if iso_temp == []:
    #             for num_analyze_2 in range(len(fake_data_temp)):
    #                 neutral_con_temp_2 = neutral_con_temp_full[num_analyze_2]
    #                 fake_data_temp_2 = fake_data_temp_full[num_analyze_2]
    #                 temp_plot = np.delete(fake_data_temp_2,1,axis = 1)[sorting_index]
    #                 plt.semilogy(np.sort(neutral_con_temp_2),temp_plot[:,plt_index_temp], "o", markersize = 15)

    #         else:
    #             for num_analyze_2 in range(len(fake_data_temp)):
    #                 neutral_con_temp_2 = neutral_con_temp_full[num_analyze_2]
    #                 fake_data_temp_2 = fake_data_temp_full[num_analyze_2]
    #                 temp_plot = np.delete(fake_data_temp_2,1,axis = 1)[sorting_index]
    #                 for indices in iso_temp:
    #                     temp_plot[:,indices[0]] = np.sum(temp_plot[:,indices], axis =1 )
    #                     temp_plot[:, indices[1:]] = np.zeros([temp_plot.shape[0],len(indices[1:])])
    #                 plt.semilogy(np.sort(neutral_con_temp_2),temp_plot[:,plt_index_temp], "o", markersize = 15)

    #         for iso in iso_temp:
    #             tit = ''
    #             if iso[0] == plt_index_temp:
    #                 tit_arr = np.array(species_0_temp)[iso]
    #                 tit = tit + tit_arr[0]
    #                 for st in tit_arr[1:]:
    #                     tit = tit + 'and' + st
    #                 plt.title(tit)
    #             if plt_index_temp in iso[1:]:
    #                 count = count + 1
    #             if iso[0] != plt_index_temp:
    #                 plt.title(species_0_temp[plt_index_temp])
    #         if iso_temp == []:
    #             plt.title(species_0_temp[plt_index_temp])
    #         if replot != 0:
    #             ax = plt.gca()
    #             ax.set_ylim(ylims[0],ylims[1])
    #         else:
    #             ylims.append(plt.gca().get_ylim())
    #         initial_list.append(initial_cons_temp)
    #     if replot == 0:
    #         ylims = np.array(ylims)
    #         ylims = (np.min(ylims),np.max(ylims))
        
    return param_stdev, fit_low, fit_high, full_sim_data, sim_params, fit_stdev, sim_gofs

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

def sim_monte(fit_stdev, fit_data, best_fit, fake_data_temp, neutral_con_temp, numpoints_temp, numk_temp, param_bounds_temp, iso_temp, nonlincon_temp, kinin_temp):
    global constraints_new, rxntime, f_jit
    
    ydot, y, k, k_l_bounds, k_h_bounds, species_0, constraints_new, con_limits_low, con_limits_high, names, reactmap, prodmap, iso_index = getodes(kinin_temp)  
    t = sym.symbols('t')
    f_lamb = sym.lambdify((t, y) + k, ydot, "numpy")
    f_jit = nb.njit(f_lamb)
    
    rxntime = [0.0024807320000000002, 0.0024807320000000002]
    neutral_reactant = [np.array('CH4')]
    
    numdiffsteps = 0
    num_tofs  = [11]
        
    ################# generates random data from a normal distribution around the real data, then fits it and returns the data and the fit parameters ############
    sim_data = []
    for nums in range(len(fit_stdev)):
        # sim_datas = np.abs(np.random.normal(loc = fit_data, scale = fit_stdev*(np.abs(fit_data)+1), size = fake_data_temp.shape))
        sim_datas = np.abs(np.random.normal(loc = fit_data[nums], scale = fit_stdev[nums]*(np.abs(fake_data_temp[nums])+1), size = fake_data_temp[nums].shape))
        sim_data.append(sim_datas)
    sim_data = np.array(sim_data)
    # sim_data[:,1] = neutral_con_temp
    sim_gofargs = (numpoints_temp, numk_temp, sim_data, neutral_con_temp, iso_temp)
    sim_res = sp.optimize.differential_evolution(getgof, param_bounds_temp, args = sim_gofargs, strategy='best2bin', 
                                              maxiter=1000, popsize=1, tol = 0.001, mutation= (0.1, 1.5), recombination=0.8, 
                                              seed=None, callback=None, disp=False, polish=False, init='sobol', 
                                              atol=0, updating='immediate', workers=1, constraints=nonlincon_temp, x0=None, 
                                              integrality=None, vectorized=False)
    sim_gof = sim_res.fun
    return sim_res.x, sim_data, sim_gof

def initial_fitting(params_bounds_temp, gofargs_temp, nonlincon, kinin_temp):
    global constraints_new, rxntime, f_jit
    ydot, y, k, k_l_bounds, k_h_bounds, species_0, constraints_new, con_limits_low, con_limits_high, names, reactmap, prodmap, iso_index = getodes(kinin_temp)
    t = sym.symbols('t')
    f_lamb = sym.lambdify((t, y) + k, ydot, "numpy")
    f_jit = nb.njit(f_lamb)
    rxntime = [0.0024807320000000002, 0.0024807320000000002]
    res = sp.optimize.differential_evolution(getgof, params_bounds_temp, args = gofargs_temp, strategy='best2bin', 
                                              maxiter=2000, popsize=1, tol=0.01, mutation= (0.1, 1.5), recombination=0.8, 
                                              seed=None, callback=None, disp= False, polish=False, init='sobol', 
                                              atol=0, updating='immediate', workers=1, constraints=nonlincon, x0=None, 
                                              integrality=None, vectorized=False)
    return res
    

################ Start of Program #####################
if __name__ == '__main__':
    global rxntime
    global numpoints
    global num_analyze
global initial_cons
good = []
zzzz = []
zzzz_init = []
data_std = []
res_std = []
res_std_max = []
for loopz in range(31):
    print("Fit number:", str(loopz+1) )
    num_analyze = 0
    kinin = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4.KININ"
    
    ydot, y, k, k_l_bounds, k_h_bounds, species_0, constraints_new, con_limits_low, con_limits_high, names, reactmap, prodmap, iso_index = getodes(kinin)
    t = sym.symbols('t')
    f_lamb = sym.lambdify((t, y) + k, ydot, "numpy")
    f_jit = nb.njit(f_lamb)
    numk = len(k)
    
    # fake_params = np.array([3.1e+00, 1.1e-02, 1.0e-01, 2.1e+00, 3.1e+00, 4.1e+00, 3.1e+00,
    #        1.0e-01, 4.1e+00, 1.0e-01, 5.1e+00, 1.1e+00, 2.1e+00, 9.1e+00,
    #        2.1e+00, 6.1e+00, 1.0e-01, 1.1e+00, 6.1e+00, 1.0e-01, 1.0e-01,
    #        9.1e+00, 1.0e-01, 1.0e-01, 4.1e+00, 1.0e-01, 5.1e+00, 1.0e-01,
    #        1.0e-01, 1.0e-01, 1.0e-01, 1.0e+03, 1.1e-03, 1.5e+04, 1.0e+01,
    #        1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
    #        1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
    #        1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, ])
            
    fake_params = np.concatenate([np.random.uniform(0.01,9.9,31),np.array([1.0e+03, 1.1e-03, 1.5e+04, 1.0e+01,
    1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
    1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
    1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+03, 1.1e-03, 1.5e+04, 1.0e+01,
    1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
    1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
    1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01])])
    
    # fake_params = np.array([4.49543586e+00, 9.20163833e-01, 9.80797728e+00, 3.58997886e+00,
    #         8.97618994e+00, 4.79132588e+00, 8.31532547e+00, 8.54678120e+00,
    #         7.43303540e+00, 3.28760649e+00, 4.24989589e+00, 2.43936604e+00,
    #         1.51969830e+00, 3.54149088e+00, 5.58714771e-01, 1.98317032e+00,
    #         3.55849434e+00, 8.97952460e+00, 6.48604321e+00, 3.90008384e+00,
    #         5.03908811e+00, 8.34408388e+00, 9.87267043e+00, 5.96947674e+00,
    #         6.23689137e+00, 8.99574211e+00, 3.05888585e+00, 7.38297768e+00,
    #         6.23032664e+00, 1.67019112e+00, 2.50345608e-01, 1.00000000e+04,
    #         1.10000000e-03, 1.50000000e+02, 1.00000000e+01, 1.00000000e+01,
    #         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
    #         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
    #         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
    #         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
    #         1.00000000e+01,]) #set A
    
    # fake_params = np.array([4.49543586e+00, 9.20163833e-01, 9.80797728e+00, 3.58997886e+00,
    #         8.97618994e+00, 4.79132588e+00, 8.31532547e+00, 8.54678120e+00,
    #         7.43303540e+00, 3.28760649e+00, 4.24989589e+00, 2.43936604e+00,
    #         1.51969830e+00, 3.54149088e+00, 5.58714771e-01, 1.98317032e+00,
    #         3.55849434e+00, 8.97952460e+00, 6.48604321e+00, 3.90008384e+00,
    #         5.03908811e+00, 8.34408388e+00, 9.87267043e+00, 5.96947674e+00,
    #         6.23689137e+00, 8.99574211e+00, 3.05888585e+00, 7.38297768e+00,
    #         6.23032664e+00, 1.67019112e+00, 2.50345608e-01, 1.00000000e+04,
    #         1.10000000e-03, 1.50000000e+02, 1.00000000e+01, 1.00000000e+01,
    #         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
    #         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
    #         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
    #         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01,
    #         1.00000000e+01, 1.00000000e+04, 1.10000000e-03, 1.50000000e+02, 
    #         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 
    #         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 
    #         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 
    #         1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 1.00000000e+01, 
    #         1.00000000e+01, 1.00000000e+01, 1.00000000e+01,])
    
    # fake_params = np.array([3.1e+00, 1.0e-01, 1.0e-01, 2.1e+00, 3.1e+00, 4.1e+00, 3.1e+00,
    #         1.0e-01, 4.1e+00, 1.0e-01, 5.1e+00, 1.1e+00, 2.1e+00, 9.1e+00,
    #         2.1e+00, 6.1e+00, 1.0e-01, 1.1e+00, 6.1e+00, 1.0e-01, 1.0e-01,
    #         9.1e+00, 1.0e-01, 1.0e-01, 4.1e+00, 1.0e-01, 5.1e+00, 1.0e-01,
    #         1.0e-01, 1.0e-01, 1.0e-01, 1.0e+03, 1.1e-03, 1.5e+04, 1.0e+01,
    #         1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
    #         1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
    #         1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01]) #set B
    
    
    # neutral_con = [np.array([0.1,  6.038061e+11,  1.198767e+12*0.6,  1.795443e+12,
    #          2.390645e+12*1.5,  2.689213e+12*1.5,  2.094381e+12*1.5,  1.498149e+12,
    #          9.021700e+11,  3.062484e+11,  4.549347e+09])]
    neutral_con = [np.array([0.1,  6.038061e+11,  1.198767e+12,  1.795443e+12,
             2.390645e+12,  2.689213e+12,  2.094381e+12,  1.498149e+12,
             9.021700e+11,  3.062484e+11,  4.549347e+09]),np.array([0.1,  6.038061e+11,  1.198767e+12,  1.795443e+12,
                      2.390645e+12,  2.689213e+12,  2.094381e+12,  1.498149e+12,
                      9.021700e+11,  3.062484e+11,  4.549347e+09])]
    
    
    numk = len(k)
    rxntime = [0.0024807320000000002, 0.0024807320000000002]
    neutral_reactant = [np.array('CH4')]
    numpoints = 11
    initial_cons = [np.array([[1.0e+03, 1.0e+03, 1.0e+03, 1.0e+03, 1.0e+03, 1.0e+03, 1.0e+03,
            1.0e+03, 1.0e+03, 1.0e+03, 1.0e+03],
           [1.1e-03, 1.1e-03, 1.1e-03, 1.1e-03, 1.1e-03, 1.1e-03, 1.1e-03,
            1.1e-03, 1.1e-03, 1.1e-03, 1.1e-03],
           [1.5e+04, 1.5e+04, 1.5e+04, 1.5e+04, 1.5e+04, 1.5e+04, 1.5e+04,
            1.5e+04, 1.5e+04, 1.5e+04, 1.5e+04],
           [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
            1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
           [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
            1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
           [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
            1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
           [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
            1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
           [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
            1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
           [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
            1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
           [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
            1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
           [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
            1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
           [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
            1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
           [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
            1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
           [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
            1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
           [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
            1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
           [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
            1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
           [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
            1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
           [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
            1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
           [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
            1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
           [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
            1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
           [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
            1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
           [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
            1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01]]),np.array([[1.0e+03, 1.0e+03, 1.0e+03, 1.0e+03, 1.0e+03, 1.0e+03, 1.0e+03,
                    1.0e+03, 1.0e+03, 1.0e+03, 1.0e+03],
                   [1.1e-03, 1.1e-03, 1.1e-03, 1.1e-03, 1.1e-03, 1.1e-03, 1.1e-03,
                    1.1e-03, 1.1e-03, 1.1e-03, 1.1e-03],
                   [1.5e+04, 1.5e+04, 1.5e+04, 1.5e+04, 1.5e+04, 1.5e+04, 1.5e+04,
                    1.5e+04, 1.5e+04, 1.5e+04, 1.5e+04],
                   [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
                    1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
                   [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
                    1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
                   [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
                    1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
                   [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
                    1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
                   [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
                    1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
                   [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
                    1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
                   [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
                    1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
                   [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
                    1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
                   [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
                    1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
                   [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
                    1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
                   [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
                    1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
                   [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
                    1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
                   [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
                    1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
                   [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
                    1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
                   [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
                    1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
                   [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
                    1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
                   [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
                    1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
                   [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
                    1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01],
                   [1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01,
                    1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01]])]
    fake_initial_cons = get_fit_initial_cons(fake_params, (len(species_0)+1,11))
    fake_initial_cons[0][1,0] = 0.1
    fake_data_perfect = solve(fake_initial_cons[0], fake_params[0:numk])
    data = []
    for i in range(2):
        data.append(np.random.normal(fake_data_perfect, 0.2*fake_data_perfect))
    # scatters = np.array([0.1,0.11,0.12,0.15,0.17,0.2,0.05,0.1,0.26,0.12,0.2])
    # fake_data = np.random.normal(fake_data_perfect.transpose(), scatters*fake_data_perfect.transpose()).transpose()
    data = np.array(data)
    
    neutral_con = [np.array([0.1,  6.038061e+11,  1.198767e+12,  1.795443e+12,
             2.390645e+12,  2.689213e+12,  2.094381e+12,  1.498149e+12,
             9.021700e+11,  3.062484e+11,  4.549347e+09]),np.array([0.1,  6.038061e+11,  1.198767e+12,  1.795443e+12,
                      2.390645e+12,  2.689213e+12,  2.094381e+12,  1.498149e+12,
                      9.021700e+11,  3.062484e+11,  4.549347e+09])]
    
    outputss = []
    numdiffsteps = 0
    # data = [fake_data]
    # initial_cons = [fake_initial_cons, fake_initial_cons]
    num_tofs  = [11]
    
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
                con_l_bound.append(con*0.25)
                con_h_bound.append(con*4)
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
    
    if __name__ == '__main__':
        start = time.time()
        quick_l = [0.001, -10, 0.001, -10, 0.001, -10]
        quick_h = [1e5, 10, 1e5, 10, 1e5, 10]
        quick_param_bounds = sp.optimize.Bounds(quick_l, quick_h)
        quick_gofs = []
        quick_ress = []
        
        p = multiprocessing.Pool(processes = 11)
        ares = []
        fit_x = []
        fit_fun = []
        num_fits_init = 50
        for i in range(num_fits_init):
            ares.append(p.apply_async(initial_fitting,args = (param_bounds,gofargs, nonlincon, kinin)))
        for i in range(num_fits_init):
            res = ares[i].get()
            outputss.append(res)
            fit_x.append(res.x)
            fit_fun.append(res.fun)
            print(i, 'is complete')
        p.close()
        p.join()
        
        small = []
        for fitnums in range(num_fits_init):
            small.append(outputss[fitnums].fun)
        res = outputss[np.argmin(small)]
        best_fit = res.x
        outputs = []
        outputs.append(res)
        
        print("Function Evauluated to: {:.2e}".format(res.fun))
        
        fit_initial_cons = []
        num_cons = int(len(res.x[numk:])/len(initial_cons))
        for i in range(len(initial_cons)):
            con0 = res.x[numk+i*num_cons:numk+i*num_cons+num_cons]
            in_cons = np.repeat(con0, numpoints).reshape(data[0].shape[1],data[0].shape[0])
            in_cons[1] = neutral_con[i]
            fit_initial_cons.append(in_cons)
            
        for i in range(len(data)):
            plt.figure(figsize = [15, 10])
            plt.semilogy(neutral_con[i],np.delete(data[i],1,axis = 1), "o")
            plt.semilogy(neutral_con[i],np.delete(solve(fit_initial_cons[i],res.x[0:numk]),1,axis = 1))
        print('Global Fit took: ',round(time.time()-start,2))
        if len(data[0].shape) == 3:
            for j in range(np.delete(data[i],1,axis = 1).shape[1]):
                plt.figure()
                for i in range(len(data)):
                    plt.semilogy(neutral_con[i],np.delete(data[i],1,axis = 1)[:,j], "o")
                    plt.semilogy(neutral_con[i],np.delete(solve(fit_initial_cons[i],res.x[0:numk]),1,axis = 1)[:,j])
    
    if __name__ == '__main__':
        time0 = time.time() 
        filenum = 0
        ############## Error Stuff ####################
        numsims = 200
        numcpus = multiprocessing.cpu_count()-2
        if numcpus > 60:
            numcpus = 60
        param_stdev, fit_low, fit_high, full_sim_data, sim_params, fit_stdev, sim_gofs = error_analysis(outputs[filenum].x, data, neutral_con, numpoints, numk, param_bounds, numsims, species_0, iso_index, nonlincon, kinin, numcpus)
    
        fail_index = []
        if (fake_params[0:numk] > fit_low[0:numk]).all() and (fake_params[0:numk] < fit_high[0:numk]).all():
            print(True)
            good.append(True)
        else:
            print(False)
            good.append(False)
            l = []
            for i, tf in enumerate(fake_params[0:numk] > fit_low[0:numk]):
                if tf == False:
                    l.append(i)
                    fail_index.append(i)
            for i, tf in enumerate(fake_params[0:numk] < fit_high[0:numk]):
                if tf == False:
                    l.append(i) 
                    fail_index.append(i)
            l = np.array(l)
            l = np.unique(l)
            fail_index = np.unique(fail_index)
            zzzz.append(np.array([fit_low[l], fake_params[l], fit_high[l]]))
        print("Error Time:", round(time.time()-start,2))
        
        ############### Useful Info ################################
        data_std.append(np.std(np.delete((data[0] - solve(fake_initial_cons[0], fake_params[0:numk]))/data[0], 1, axis = 1)))
        res_std.append(np.std(residual))
        res_std_max.append(fit_stdev)
        fit_best = np.average(sim_params, 0)
        zz = np.array([fake_params, best_fit, fit_best])
        print('Total Time: ',round(time.time()-start,2))