# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 09:48:42 2024

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
import pathlib
import matplotlib as mpl

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
        # r = k[-1]*prod([sym.Function(str(c[rk]**p))(t) for rk, p in r_stoich.items()])  # EXERCISE: c[rk]**p
        r = k[-1]*prod([sym.Function(str(c[rk]))(t)**p for rk, p in r_stoich.items()])  # EXERCISE: c[rk]**p
        for net_key, net_mult in net_stoich.items():
            f[net_key] += net_mult*r  # EXERCISE: net_mult*r
    return [f[n] for n in names], [sym.Function(str(i))(t) for i in symbs], tuple(k)

def getodes(kinin_temp):
    global reactions
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
            text_iso = text_out[index+1:text.split('\n').index('Mass Descrimination')]
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
            if j[0].isnumeric() or j[0].isalpha() or j[0] == '(':
                reactants2.append(j)
        reactants3.append(reactants2)
    for i in products:
        products2 = []
        for j in i:
            if j[0].isnumeric() or j[0].isalpha() or j[0] == '(':
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
        
    for i in range(len(reactants3)):
        react_dict = {}
        net_dict = {}
        for j in reactants3[i]:
            if j in react_dict:
                react_dict[j] += 1
                net_dict[j] += -1
            else:
                react_dict[j] = 1
                net_dict[j] = -1
        for j in products3[i]:
            if j in net_dict:
                net_dict[j] += 1
            else:
                net_dict[j] = 1
        reactions.append([k_out[i], react_dict, net_dict])
    
    names = res
    
    sym.init_printing()
    ydot, y, k = mk_exprs_symbs(reactions, temp)
    
    names = temp
    
    text_lines = np.array(text.split('\n'))
    con_index = np.where(text_lines == "Conditions")[0][0]
    iso_index = np.where(text_lines == 'Iso')[0][0]
    con_text = text_lines[con_index+1:iso_index-1]
    
    con_limits_low = []
    con_limits_high = []
    constraints = []
    constraints_new = []
    for line in con_text:
        stuff = line[0:-2].split('<')
        con_limits_low.append(float(stuff[0]))
        con_limits_high.append(float(stuff[2]))
        constraints.append(stuff[1])
    for cur_con in constraints:
        for rate in cur_con.split():
            if sym.symbols(rate) in list(k):
                k_index = list(k).index(sym.symbols(rate))
                cur_con = cur_con.replace(rate,'params[{}]*k_l_bounds[{}]'.format(k_index,k_index))
        constraints_new.append(cur_con)
    if constraints_new == []:
        constraints_new.append('params[0]*k_l_bounds[0]')
    
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
        for index, spi in enumerate(names):
            if spi == stri:
                # if index > 0:
                #     temps.append(index+1)
                # if index == 0:
                #     temps.append(index)
                temps.append(index)
        if stri == '|':
            iso_index.append(temps)
            temps = []
            
    mass_descrim = [1]*len(names)
    initial_cons_bounds_start_index = text.split('\n').index('Mass Descrimination')
    for line in text.split('\n')[initial_cons_bounds_start_index:]:
        if line.split()[0] in names:
            mass_descrim[names.index(line.split()[0])] = float(line.split()[1])
    
    return ydot, y, k, low_bound, high_bound, specs, constraints_new, con_limits_low, con_limits_high, names, reactants3, products3, iso_index, np.array(mass_descrim)

def batch_import(species_temp, files_temp, iso_temp):
    global initials, initial_cons_temp, x, y, string_list, start_index
    ####### imports batch.exp files and generates a table of the data and other values ##########
    f = open(files_temp)
    text = f.read()
    f.close()
    
    names = species_temp
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
    data_temp = cons
    # data_temp[1] = neutral_con_temp
    initial_cons_temp = np.repeat(np.zeros(data_temp.shape[0]),data_temp.shape[1]).reshape(data_temp.shape)
    initial_cons_temp[1,:] = neutral_con_temp
    initials = [string_list[start_index][0::2], string_list[start_index][1::2]]
    for x_index, x in enumerate(names):
        for y_index, y in enumerate(initials[0]):
            if x == y:
                if not initial_cons_temp.any(axis = 1)[x_index]:
                    initial_cons_temp[x_index] = np.repeat(initials[1][y_index], data_temp.shape[1])
    if text.split('\n')[1] in names:
        for name_index in range(len(names)):
            if names[name_index] == text.split('\n')[1]:
                buffer_index = name_index
        temperature = float(text.split('\n')[3])
        pressure = float(text.split('\n')[4])
        buffer_density = pressure/temperature/62.363/1000*6.022e23
        initial_cons_temp[buffer_index] = np.repeat(buffer_density, data_temp.shape[1])
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

def getgof(params,numpoints_temp,numks,ydatas,neutral_con_temp, iso_temp, k_l_bounds_temp):
    global num_analyze, res_per_square
    final_res = []
    k_vals = params[0:numks]*k_l_bounds_temp
    num_cons = int(len(params[numks:])/len(ydatas))
    for num_analyze in range(len(ydatas)): 
        ydata = np.copy(ydatas[num_analyze])
        con0 = params[numks+num_analyze*num_cons:numks+num_analyze*num_cons+num_cons] #need to take into account the different potential initial con conditions
        in_cons = np.repeat(con0, ydata.shape[0]).reshape(ydata.shape[1],ydata.shape[0])
        in_cons[1] = neutral_con_temp[num_analyze]
        fit_ys = solve(in_cons, k_vals).reshape(in_cons.shape[1],in_cons.shape[0])
        for indices in iso_temp:
            fit_ys[:,indices[0]] = np.sum(fit_ys[:,indices], axis =1 )
            fit_ys[:,indices[1:]] = np.zeros([ydata.shape[0],len(indices[1:])])
        map_of_zeros = np.copy(ydata)
        map_of_zeros[map_of_zeros!=0] /= map_of_zeros[map_of_zeros!=0]
        fit_ys = fit_ys*map_of_zeros 
        
        # res_abs = np.abs(fit_ys-ydata)
        # res_fract = res_abs/(ydata+1)
        
        # res_per = res_fract*100
        # res_per_square = res_per
        # max_vals = np.argmax(res_per_square, axis = 0)
        # res_per_square[max_vals, range(len(max_vals))] = 0
        
        # weighted = res_per_square*np.sqrt(np.abs(ydata))
        # final_res.append(np.sum(weighted)**2) 
        # final_res.append(np.sum(np.abs(np.log(res_per_square**2, where=(res_per_square!=0)))))
        
        # final_res = np.sum((np.log(fit_ys, where=(fit_ys!=0)) - np.log(ydata,where=(ydata!=0)))**2)
        final_res = np.sum(np.log(fit_ys/ydata, where=(ydata!=0))**2)

    return np.sum(final_res)

def solve2(y_0,*ki):
    ############ runge kutta 4th order ODE solver ###############
    t_0 = 0
    t_final = rxntime[num_analyze]
    dt = t_final/1000
    ts = [t_0]
    ys = [y_0]
    y = y_0
    t = t_0 
    if len(ki) == 1:
        ki = ki[0]
    ki = np.array(ki)
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

def solve(y_0,*ki):
    ################ runge kutte 45 ODE solver #################
    t_final = rxntime[num_analyze]
    y = y_0
    t = 0
    h = rxntime[num_analyze]/1000
    if len(ki) == 1:
        ki = ki[0]
    ki = np.array(ki)
    A = [0, (1/4), (3/32), (12/13), 1, (1/2)]
    B = np.array([[0,1/4,3/32,1932/2197,439/216,-8/27],[0,0,9/32,-7200/2197,-8,2],[0,0,0,7296/2197,3680/513,-3544/2565],[0,0,0,0,-845/4104,1859/4104],[0,0,0,0,0,-11/40]]).transpose()
    # C = [25/216,0,1408/2565,2197/4104,-1/5]
    CH = [16/135,0,6656/12825,28561/56430,-9/50,2/55]
    CT = [-1/360,0,128/4275,2197/75240,-1/50,-2/55]
    error = 1e2
    i = 0
    while t < t_final:
        i = i + 1
        if i > 10000:
            break
        k1 = h * np.array(f_jit(t+(A[0]*h),y,*ki))
        k2 = h * np.array(f_jit(t+(A[1]*h),y+(B[1,0]*k1),*ki))
        k3 = h * np.array(f_jit(t+(A[2]*h),y+(B[2,0]*k1)+(B[2,1]*k2),*ki))
        k4 = h * np.array(f_jit(t+(A[3]*h),y+(B[3,0]*k1)+(B[3,1]*k2)+(B[3,2]*k3),*ki))
        k5 = h * np.array(f_jit(t+(A[4]*h),y+(B[4,0]*k1)+(B[4,1]*k2)+(B[4,2]*k3)+(B[4,3]*k4),*ki))
        k6 = h * np.array(f_jit(t+(A[5]*h),y+(B[5,0]*k1)+(B[5,1]*k2)+(B[5,2]*k3)+(B[5,3]*k4)+(B[5,4]*k5),*ki))
        TE = np.abs(CT[0] * k1 + CT[1] * k2 + CT[2] * k3 + CT[3] * k4 + CT[4] * k5 + CT[5] * k6)
        y_next = y + CH[0]*k1 + CH[1] * k2 + CH[2] * k3 + CH[3] * k4 + CH[4] * k5 + CH[5] * k6
        if np.max(TE/np.clip(y_next,1,None)) > error:
            t = t
        else:
            y = y_next
            t = t + h
        h = 0.9*h*(error/np.max(TE))**(1/5)
        # input('continue')
    return np.array(y).transpose()

def initial_fitting(params_bounds_temp, gofargs_temp, nonlincon, kinin_temp, files_temp, rois_temp):
    global constraints_new, rxntime, f_jit
    ydot, y, k, k_l_bounds, k_h_bounds, species_0, constraints_new, con_limits_low, con_limits_high, names, reactmap, prodmap, iso_index, mass_descrim = getodes(kinin_temp)
    t = sym.symbols('t')
    f_lamb = sym.lambdify((t, y) + k, ydot, "numpy")
    f_jit = nb.njit(f_lamb)
    rxntime = []
    for input_file in files_temp:
        if '.BATCHEXP' or '.batchexp' in input_file:
            rxntimes, neutral_reactants, datas, neutral_cons, initial_conss = batch_import(names, input_file, iso_index)
        if '.TOFs' in input_file:
            rxntimes, neutral_cons, datas, num_tofss, initial_conss = tof_import(input_file, rois_temp, names)
        rxntime.append(rxntimes)
    res = sp.optimize.differential_evolution(getgof, params_bounds_temp, args = gofargs_temp, strategy='best2bin', 
                                              maxiter=2000, popsize=1, tol=1, mutation= (0.01, 1.99), recombination=0.7, 
                                              seed=None, callback=None, disp=True, polish=False, init='sobol', 
                                              atol=0, updating='immediate', workers=1, constraints=nonlincon, x0=None, 
                                              integrality=None, vectorized=False)
    return res

def error_analysis(best_fit, gofargs_temp, param_bounds_temp, numsims_temp, names_temp, nonlincon_temp, kinin_temp,numcpus_temp, files_temp, rois_temp):
    global ares, num_analyze, max_vals, residual, ylims, sim_gofs, params_trunc, sim_params, initial_list, to_trunc
    #unpacking gofargs
    fake_data_temp = np.copy(gofargs_temp[2])
    numk_temp = gofargs_temp[1]
    iso_temp = gofargs_temp[4]
    neutral_con_temp = gofargs_temp[3]
    k_l_bounds_temp = gofargs_temp[5]
    ############ calculating the standard deviation in the scatter of the original data around the best fit ##############
    fit_stdev = []
    fit_data = []
    initial_cons_temp_full = get_fit_initial_cons(best_fit, (fake_data_temp.shape[2],fake_data_temp.shape[1]))
    for num_analyze in range(len(fake_data_temp)):
        fit_datas = solve(initial_cons_temp_full[num_analyze], best_fit[0:numk_temp]*k_l_bounds_temp)
        for indices in iso_temp:
            fit_datas[:,indices[0]] = np.sum(fit_datas[:,indices], axis =1 )
            fit_datas[:, indices[1:]] = np.zeros([fake_data_temp[num_analyze].shape[0],len(indices[1:])])
        map_of_zeros = np.copy(fake_data_temp[num_analyze])
        map_of_zeros[map_of_zeros!=0] /= map_of_zeros[map_of_zeros!=0]
        fit_datas = fit_datas*map_of_zeros        
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
    with multiprocessing.Pool(processes = numcpus_temp) as p:
        for loops in range(numsims_temp):
            ares.append(p.apply_async(sim_monte, args = (fit_stdev, fit_data, best_fit, gofargs_temp, param_bounds_temp, nonlincon_temp, kinin_temp, files_temp, rois_temp)))
            
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
    
    q1 = np.percentile(sim_gofs,25)
    q2 = np.percentile(sim_gofs,75)
    q3 = np.percentile(sim_gofs,50)
    t_fences = [q1 - k_factor*(q3-q1),q2+k_factor*(q2-q3)]
       
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
            fit_low.append(np.percentile(to_trunc[indices][indices_95],0))
            fit_high.append(np.percentile(to_trunc[indices][indices_95],100))
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
    
    if plotting == 'yes':
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
        
        map_of_zeros = np.copy(fake_data_temp_full[0])
        map_of_zeros[map_of_zeros!=0] /= map_of_zeros[map_of_zeros!=0]
        
        for replot in range(2):
            for plt_index_temp in range(sim_data.shape[2]):
                if np.any(fake_data_temp[0][:,plt_index_temp]):
                    plt.figure(figsize = [15, 10])
                    if iso_temp != []:
                        for iso in iso_temp:
                            full_sim_data[:,:,iso[0]+1] = np.sum(full_sim_data[:,:,np.array(iso)+1],axis = 2)
                            full_sim_data[:,:,np.array(iso[1:])+1] = np.zeros([full_sim_data.shape[0],full_sim_data.shape[1],len(indices[1:])])
                    
                    for omit_index in ommiteds:
                        for num_analyze in range(sim_data.shape[0]):
                            initial_cons_temp = initial_cons_temp_full[num_analyze]
                            neutral_con_temp = neutral_con_temp_full[num_analyze]
                            sorting_index = np.argsort(neutral_con_temp)
                            num_cons = int(len(sim_params[0][numk_temp:])/len(fake_data_temp))
                            sim_index = [numk_temp+num_analyze*num_cons,numk_temp+num_analyze*num_cons+num_cons]
                            initial_cons_temp = np.reshape(np.repeat(sim_params[omit_index][sim_index[0]:sim_index[1]],num_neutral),(num_species,num_neutral))
                            initial_cons_temp[1] = neutral_con_temp
                            temp_plot = solve(initial_cons_temp,sim_params[omit_index][0:numk_temp]*k_l_bounds_temp)[sorting_index][:,plt_index_temp]
                            plt.semilogy(np.sort(neutral_con_temp),temp_plot, color = 'black', alpha = 0.5)
                    
                    for plts_index, plts in enumerate(np.array(params_trunc).transpose()):
                        for num_analyze in range(sim_data.shape[0]):
                            initial_cons_temp = initial_cons_temp_full[num_analyze]
                            neutral_con_temp = neutral_con_temp_full[num_analyze]
                            sorting_index = np.argsort(neutral_con_temp)
                            num_cons = int(len(sim_params[0][numk_temp:])/len(fake_data_temp))
                            sim_index = [numk_temp+num_analyze*num_cons,numk_temp+num_analyze*num_cons+num_cons]
                            initial_cons_temp = np.reshape(np.repeat(plts[sim_index[0]:sim_index[1]],num_neutral),(num_species,num_neutral))
                            initial_cons_temp[1] = neutral_con_temp
                            temp_plot = (solve(initial_cons_temp,plts[0:numk_temp]*k_l_bounds_temp)*map_of_zeros[0])[sorting_index][:,plt_index_temp]
                            plt.semilogy(np.sort(neutral_con_temp),temp_plot, color = 'red', alpha = 0.1)
                        
                    if iso_temp == []:
                        for num_analyze_2 in range(len(fake_data_temp)):
                            initial_cons_temp = initial_cons_temp_full[num_analyze_2]
                            neutral_con_temp = neutral_con_temp_full[num_analyze_2]
                            sorting_index = np.argsort(neutral_con_temp)
                            neutral_con_temp_2 = neutral_con_temp_full[num_analyze_2]
                            fake_data_temp_2 = fake_data_temp_full[num_analyze_2]
                            temp_plot = fake_data_temp_2[sorting_index]
                            plt.semilogy(np.sort(neutral_con_temp_2),temp_plot[:,plt_index_temp], "o", markersize = 15)
        
                    else:
                        for num_analyze_2 in range(len(fake_data_temp)):
                            initial_cons_temp = initial_cons_temp_full[num_analyze_2]
                            neutral_con_temp = neutral_con_temp_full[num_analyze_2]
                            sorting_index = np.argsort(neutral_con_temp)
                            neutral_con_temp_2 = neutral_con_temp_full[num_analyze_2]
                            fake_data_temp_2 = fake_data_temp_full[num_analyze_2]
                            temp_plot = fake_data_temp_2[sorting_index]
                            for indices in iso_temp:
                                temp_plot[:,indices[0]] = np.sum(temp_plot[:,indices], axis =1 )
                                temp_plot[:, indices[1:]] = np.zeros([temp_plot.shape[0],len(indices[1:])])
                            plt.semilogy(np.sort(neutral_con_temp_2),temp_plot[:,plt_index_temp], "o", markersize = 15)
                            
                    for iso in iso_temp:
                        tit = ''
                        if iso[0] == plt_index_temp:
                            tit_arr = np.array(names_temp)[iso]
                            tit = tit + tit_arr[0]
                            for st in tit_arr[1:]:
                                tit = tit + 'and' + st
                            plt.title(tit)
                        if plt_index_temp in iso[1:]:
                            count = count + 1
                        if iso[0] != plt_index_temp:
                            plt.title(names_temp[plt_index_temp])
                    if iso_temp == []:
                        plt.title(names_temp[plt_index_temp])
                    if replot != 0:
                        ax = plt.gca()
                        ax.set_ylim(np.min(fake_data_temp[fake_data_temp != 0])/10,np.max(fake_data_temp)*10)
                    else:
                        ylims.append(plt.gca().get_ylim())
                    initial_list.append(initial_cons_temp)
                    if replot == 0:
                        plt.close()
            if replot == 0:
                ylims = np.array(ylims)
                ylims = (np.min(ylims),np.max(ylims))
        
    return param_stdev, fit_low, fit_high, full_sim_data, sim_params, fit_stdev, sim_gofs

def get_fit_initial_cons(res,data_shape):
    if type(res) == sp.optimize._optimize.OptimizeResult:
        res = res.x
    fit_initial_cons = []
    num_cons = int(len(res[numk:])/len(initial_cons))
    for i in range(len(initial_cons)):
        con0 = res[numk+i*num_cons:numk+i*num_cons+num_cons]
        in_cons = np.repeat(con0, data_shape[1]).reshape(data_shape)
        in_cons[1] = neutral_con[i]
        fit_initial_cons.append(in_cons)
    return fit_initial_cons

def sim_monte(fit_stdev, fit_data, best_fit, sim_gofargs, param_bounds_temp, nonlincon_temp, kinin_temp, files_temp, rois_temp):
    global constraints_new, rxntime, f_jit
    #unpacking gofargs
    fake_data_temp = sim_gofargs[2]
    
    ydot, y, k, k_l_bounds, k_h_bounds, species_0, constraints_new, con_limits_low, con_limits_high, names, reactmap, prodmap, iso_index, mass_descrim = getodes(kinin)
    t = sym.symbols('t')
    f_lamb = sym.lambdify((t, y) + k, ydot, "numpy")
    f_jit = nb.njit(f_lamb)
    
    rxntime = []
    for input_file in files_temp:
        if '.BATCHEXP' or '.batchexp' in input_file:
            rxntimes, neutral_reactants, datas, neutral_cons, initial_conss = batch_import(names, input_file, iso_index)
        if '.TOFs' in input_file:
            rxntimes, neutral_cons, datas, num_tofss, initial_conss = tof_import(input_file, rois_temp, names)
        rxntime.append(rxntimes)
    # neutral_reactant = neutral_reactants
        
    ################# generates random data from a normal distribution around the real data, then fits it and returns the data and the fit parameters ############
    sim_data = []
    for nums in range(len(fit_stdev)):
        # sim_datas = np.abs(np.random.normal(loc = fit_data, scale = fit_stdev*(np.abs(fit_data)+1), size = fake_data_temp.shape))
        sim_datas = np.abs(np.random.normal(loc = fit_data[nums], scale = fit_stdev[nums]*(np.abs(fake_data_temp[nums])), size = fake_data_temp[nums].shape))
        sim_data.append(sim_datas)
    sim_data = np.array(sim_data)
    # sim_data[:,1] = neutral_con_temp
    sim_gofargs = list(sim_gofargs)
    sim_gofargs[2] = sim_data
    sim_gofargs = tuple(sim_gofargs)
    sim_res = sp.optimize.differential_evolution(getgof, param_bounds_temp, args = sim_gofargs, strategy='best2bin', 
                                              maxiter=2000, popsize=1, tol = 0.1, mutation= (0.01, 1.99), recombination=0.70, 
                                              seed=None, callback=None, disp=False, polish=False, init='sobol', 
                                              atol=0, updating='immediate', workers=1, constraints=nonlincon_temp, x0=None, 
                                              integrality=None, vectorized=False)
    sim_gof = sim_res.fun
    return sim_res.x, sim_data, sim_gof

def get_all_inputs(kinin_temp, batchin_temp, BLS_temp):
    global data, initial_con_0
    #get reactions system and other info from kinin
    ydot, y, k, k_l_bounds, k_h_bounds, species_0, constraints_new, con_limits_low, con_limits_high, names, reactmap, prodmap, iso_index, mass_descrim = getodes(kinin)
    t = sym.symbols('t')
    f_lamb = sym.lambdify((t, y) + k, ydot, "numpy")
    f_jit = nb.njit(f_lamb)
    numk = len(k)
    #find the data files and group them by simultaneous fit
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
    #import the data
    for filenum, input_files in enumerate(files_grouped): 
        rxntime = []
        neutral_reactant = []
        data = []
        neutral_con = []
        initial_cons = []
        num_tofs  = []
        numpoints = []
        for input_file in input_files:
            if 'rois' not in globals():
                rois = ''
            if '.BATCHEXP' or '.batchexp' in input_file:
                rxntimes, neutral_reactants, datas, neutral_cons, initial_conss = batch_import(names, input_file, iso_index)
            if '.TOFs' in input_file:
                rxntimes, neutral_cons, datas, num_tofss, initial_conss = tof_import(input_file, rois, names)
                a = np.unique(np.array(reactmap),return_counts = True)
                neutral_reactants = a[0][np.argmax(a[1])]
                num_tofs.append(num_tofss)
            initial_conss[1] = neutral_cons
            rxntime.append(rxntimes)
            neutral_reactant.append(neutral_reactants)
            data.append(datas*mass_descrim)
            neutral_con.append(neutral_cons)
            initial_cons.append(initial_conss)
        #the following section standardizes the size of the data by padding zeros into smaller data sets
        size = []
        newdata = []
        for items in data:
            size.append(items.shape[0])
        biggest = np.max(size)
        numcons = data[0].shape[1]
        for resize_index, items in enumerate(data):
            sizediff = biggest - items.shape[0]
            toadd = np.repeat(np.zeros(numcons),sizediff).reshape(sizediff,numcons)
            newdata.append(np.concatenate((items, toadd)))
            neutral_con[resize_index] = np.concatenate([neutral_con[resize_index],np.zeros(sizediff)])
            initial_cons[resize_index] = np.concatenate([initial_cons[resize_index],np.repeat(np.zeros(numcons),sizediff).reshape(numcons,sizediff)], axis = 1)
        data = np.clip(np.array(newdata)-BLS_temp,0,None)
        numpoints.append(datas.shape[0])
    numpoints = np.max(numpoints)
    #set the initial conditions
    initial_con_0 = []
    for initial_c in initial_cons:
        # initial_con_0_temp = (initial_c[:,0] + initial_c[:,-1])/2
        initial_con_0_temp = initial_c[:,0]
        initial_con_0.append(initial_con_0_temp)

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
                # initial_con_0[j][i] = 0.001
                initial_con_0[j][i] = 0
                con_l_bound.append(0)
                # con_h_bound.append(0.002)
                con_h_bound.append(0)
            else:
                con_l_bound.append(con*0.1)
                con_h_bound.append(con*10)
        con_l_bound = np.array(con_l_bound)
        con_h_bound = np.array(con_h_bound)
        con_l_bounds.append(con_l_bound)
        con_h_bounds.append(con_h_bound)
    #make parameter bounds for diff evolution
    l_bounds = np.concatenate((k_l_bounds/k_l_bounds,con_l_bounds[0]))
    h_bounds = np.concatenate((k_h_bounds/k_l_bounds,con_h_bounds[0]))
    if len(con_l_bounds) > 1:
        for i in range(len(con_l_bounds[1:])):
            l_bounds = np.concatenate((l_bounds,con_l_bounds[i+1]))
            h_bounds = np.concatenate((h_bounds,con_h_bounds[i+1]))
    param_bounds = sp.optimize.Bounds(l_bounds,h_bounds)
    #make rate constant constraints
    if con_limits_low == []:
        con_limits_low.append(k_l_bounds[0])
        con_limits_high.append(k_h_bounds[0])
    nonlincon = sp.optimize.NonlinearConstraint(con_fun, con_limits_low, con_limits_high)
    
    gofargs = (numpoints, numk, np.copy(data), neutral_con, iso_index, k_l_bounds)
    
    return [param_bounds, gofargs, nonlincon, files_grouped, initial_cons, species_0, f_jit, rxntime, k, names, ydot, constraints_new, k_h_bounds]

def outputfile(input_files, combined_out, k, names):
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
                
        outstr = outstr + '\n'
        
        outstr = outstr + 'Sim Params\n'
        for s_p in sim_params:
            for item in s_p:
                outstr = outstr + str(item) + '\t'
            outstr = outstr + '\n'
        
        outstr = outstr + '\n'
        
        outstr = outstr + 'Sim Gofs\n'
        for s_p in sim_gofs:
            outstr = outstr + str(s_p) + '\n'
        
        outputname = input_file[input_file.rfind('\\')+1:input_file.rfind('.')] + '_output.txt'
        newdir = input_file[0:input_file.rfind('.')] + '/'
        save_path = pathlib.Path(newdir)
        save_path.mkdir(parents=True, exist_ok=True)
        save = save_path /  outputname
        f = open(save, 'w')
        f.write(outstr)
        f.close()
        return newdir + outputname

def get_cmap(n):
    if n < 11:
        return mpl.colormaps['tab10']
    if n < 20 and n > 11:
        return mpl.colormaps['tab20']

def plots(kinin_temp, kvt_temp):
    ydot, y, k, k_l_bounds, k_h_bounds, species_0, constraints_new, con_limits_low, con_limits_high, names, reactmap, prodmap, iso_index = getodes(kinin_temp)
    
    trunc_params, ommited_fits, temps = get_truncated_params(kvt_temp)
    data, rxntime, neutral_reactant, neutral_con, initial_cons, names = get_data(batchin, reactmap)
    
    for plotting_temp in temps:
        to_plot, ommited_to_plot, plotting_indices = get_plot_data(temps, plotting_temp, trunc_params, ommited_fits, k_l_bounds)
        
        species = 'all'
        species_index = []
        if species != 'all':
            for spec in species:
                species_index.append(np.where(np.array(species_0) == spec)[0][0])
        else:
            species_index = [item for item in range(len(species_0))]
        
        num_plots = len(plotting_indices)
        cmap = get_cmap(num_plots)
        used_names = np.array(names)[plotting_indices]
        
        ommited_to_plot = []
        
        ymin = np.min(np.abs(np.array(data)[np.nonzero(data)]))/10
        ymax = np.max(np.delete(data,1,axis = 2))
        
        plt.rcParams['font.size'] = 15
        
        file_path = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta+ and Ta(CH2) simul\new_figures" + r'\{}'.format(plotting_temp)
        
        for species_plot in species_index:
            leg_handles = []
            plt.figure(figsize = (10.5,7.5))
            for plots in ommited_to_plot:
                for num_plot, vals_plot in enumerate(plots):
                    plt.semilogy(np.sort(neutral_con[plotting_indices[num_plot]]), vals_plot[:,species_plot], color = 'black', alpha = 0.1)   
            for plots in to_plot:
                for num_plot, vals_plot in enumerate(plots):
                    plt.semilogy(np.sort(neutral_con[plotting_indices[num_plot]]), vals_plot[:,species_plot], color = cmap(num_plot), alpha = 0.01)
            for num_plot, plots in enumerate(np.array(data)[plotting_indices]):
                plots = np.delete(plots, 1, axis = 1)
                sorting_index = np.argsort(neutral_con[plotting_indices[num_plot]])
                leg_handles.append(plt.semilogy(np.sort(neutral_con[plotting_indices[num_plot]]), plots[:,species_plot][sorting_index], 'o', color = cmap(num_plot), label = used_names[num_plot], 
                                                markeredgecolor = 'black', markersize = 10))
            title = species_0[species_plot] + ' {}K'.format(plotting_temp)
            plt.title(title)
            plt.legend(handles = [item[0] for item in leg_handles], loc = 'best', frameon = True, framealpha = 0.3, fontsize = 'small')
            xlabel = "[{}]".format(neutral_reactant[0][0]) + ' $cm^{-3}$'
            plt.xlabel(xlabel, fontsize = 'large')
            plt.ylabel('Counts', fontsize = 'large')
            ax = plt.gca()
            ax.set_ylim([ymin,ymax])
            save_path = file_path + '\{}'.format(title)
            plt.savefig(save_path, bbox_inches = 'tight', pad_inches = 0.1)
            plt.close()

def get_truncated_params(kvt_temp):
    global sim_gofs, t_fences, indices_95, sim_params, indices, indices_95, gofs_iqr, trunc_index
    temps = []
    sim_params = []
    blanks = []
    sim_gofs = []
    ommiteds = []
    ommiteds_in = []
    params_trunc = []
    with open(kvt_temp) as f:
        file_list = f.read()
    file_list = file_list.split('\n')
    
    for file in file_list:
        temps.append(int(file[0:3]))
        file = file[6:-1]
        
        with open(file) as f:
            text = f.read()
        text_split = text.split('\n')    
        
        stop = 0
        for index, strings in enumerate(text_split):
            if strings == '':
                blanks.append(index)
            if strings == 'Sim Params':
                sim_start = index
            if len(strings) > 0:
                if strings[0] != 'p' and strings[0] != 'k' and stop == 0:
                    stop = 1
            if strings == 'Sim Gofs':
                sim_gofs_start = index
        sim_params.append(np.genfromtxt(text_split[sim_start+1:sim_gofs_start-1], dtype = np.float64))
        sim_gofs.append(np.genfromtxt(text_split[sim_gofs_start+1:], dtype = np.float64))
    # sim_params = np.array(data)
    # sim_gofs = np.array(sim_gofs)
    indices = []
    indices_95 = []
    for gofs in sim_gofs:
        quartiles = np.percentile(gofs, [25,75], axis = 0)
        k_factor = 1.5
        iqr = (quartiles[1]-quartiles[0])*k_factor
        t_fences = np.array([quartiles[0]-iqr,quartiles[1]+iqr])
    
        indices.append(np.where(gofs < t_fences[1]))
        gofs_iqr = gofs[np.where(gofs < t_fences[1])]
        gofs_high_95 = np.percentile(gofs_iqr,95)
        indices_95.append(np.where(gofs_iqr < gofs_high_95)[-1])
        gofs_iqr_95 = gofs[np.where(gofs_iqr < gofs_high_95)[-1]]
        omits = []
        for omitted_index, ommited in enumerate(gofs):
            if ommited not in gofs_iqr_95:
                omits.append(omitted_index)
        ommiteds_in.append(np.array(omits))

    for trunc_index, to_trunc in enumerate(sim_params):
        params_trunc.append(to_trunc[indices[trunc_index]][indices_95[trunc_index]])
        ommiteds.append(to_trunc[ommiteds_in[trunc_index]])
    
    return params_trunc, ommiteds, np.array(temps)

def get_data(batchin_temp, reactmap):
    global file_list
    with open(batchin_temp) as f:
        text = f.read()
    text = text.split('\n')

    file_list = []
    rxntime = []
    neutral_reactant = []
    data = []
    neutral_con = []
    initial_cons = []
    num_tofs  = []
            
    for line in text:
        file_list.append(line[3:-1])
    names = []
    for file in file_list:
        names.append(file[file.rfind('\\')+1:file.rfind('.')])
    for file in file_list:
        if 'rois' not in globals():
            rois = ''
        if '.BATCHEXP' in file:
            rxntimes, neutral_reactants, datas, neutral_cons, initial_conss = batch_import(species_0, file, iso_index)
        if '.TOFs' in file:
            rxntimes, neutral_cons, datas, num_tofss, initial_conss = tof_import(file, rois, names)
            a = np.unique(np.array(reactmap),return_counts = True)
            neutral_reactants = a[0][np.argmax(a[1])]
            num_tofs.append(num_tofss)
        initial_conss[1] = neutral_cons
        rxntime.append(rxntimes)
        neutral_reactant.append(neutral_reactants)
        data.append(datas)
        neutral_con.append(neutral_cons)
        initial_cons.append(initial_conss)
    return data, rxntime, neutral_reactant, neutral_con, initial_cons, names
        
def get_plot_data(temps, plotting_temp, trunc_params, ommited_fits, k_l_bounds_temp):
    global num_analyze, initial_vals, rate_constants, plotting_indices
    plotting_indices = []
    for temp_index, T in enumerate(temps):
        if T == plotting_temp:
            plotting_indices.append(temp_index)

    num_cons = data[plotting_indices[0]].shape[1]
    num_species = data[plotting_indices[0]].shape[0]
    plot_data = []
    ommited_data = []
    plot_data = []
    for index, params in enumerate(trunc_params[plotting_indices[0]]):
        plot_initial_cons = []
        rate_constants = params[0:numk]*k_l_bounds_temp
        plot_data_temp = []
        for num_analyze in range(int((trunc_params[plotting_indices[0]].shape[1] - numk)/num_cons)):
            in_cons = params[numk+num_analyze*num_cons:numk+num_analyze*num_cons+num_cons]
            initial_vals = np.repeat(in_cons,num_species).reshape(num_cons,num_species)
            initial_vals[1] = neutral_con[plotting_indices[num_analyze]]
            plot_initial_cons.append(initial_vals)
            sorting_index = np.argsort(neutral_con[plotting_indices[num_analyze]])
            plot_data_temp.append(np.delete(solve(initial_vals,rate_constants),1,axis = 1)[sorting_index])
        plot_data.append(plot_data_temp)
    for index, params in enumerate(ommited_fits[plotting_indices[0]]):
        plot_initial_cons = []
        rate_constants = params[0:numk]
        plot_data_temp = []
        for num_analyze in range(int((trunc_params[plotting_indices[0]].shape[1] - numk)/num_cons)):
            in_cons = params[numk+num_analyze*num_cons:numk+num_analyze*num_cons+num_cons]
            initial_vals = np.repeat(in_cons,num_species).reshape(num_cons,num_species)
            initial_vals[1] = neutral_con[plotting_indices[num_analyze]]
            plot_initial_cons.append(initial_vals)
            sorting_index = np.argsort(neutral_con[plotting_indices[num_analyze]])
            plot_data_temp.append(np.delete(solve(initial_vals,rate_constants),1,axis = 1)[sorting_index])
        ommited_data.append(plot_data_temp)
    return plot_data, ommited_data, plotting_indices

res_full = []
#input files
kinin = r"C:\Users\Tucker Lewis\Documents\AFRL\N3+ N4+\35reactions_deleted.KININ"
batchin = r"C:\Users\Tucker Lewis\Documents\AFRL\N3+ N4+\N3+_simul.BATCHIN"

kinin = r"C:\Users\Tucker Lewis\Documents\AFRL\N3+ N4+\testing\N4+ testing_6.KININ"
batchin = r"C:\Users\Tucker Lewis\Documents\AFRL\N3+ N4+\testing\N4+ testing.BATCHIN"

BLS = 0

inputs_tuple = get_all_inputs(kinin, batchin, BLS)
param_bounds = inputs_tuple[0]
gofargs = inputs_tuple[1]
nonlincon = inputs_tuple[2]
files_grouped = inputs_tuple[3]
initial_cons = inputs_tuple[4]
species_0 = inputs_tuple[5]
f_jit = inputs_tuple[6]
rxntime = inputs_tuple[7]
k = inputs_tuple[8]
names = inputs_tuple[9]
ydot = inputs_tuple[10]
constraints_new = inputs_tuple[11]
k_h_bounds = inputs_tuple[12]

numpoints = gofargs[0]
numk = gofargs[1]
data = gofargs[2]

neutral_con = gofargs[3]
iso_index = gofargs[4]
k_l_bounds = gofargs[5]

num_analyze = 0

plotting = 'no'

if __name__ == '__main__':
    output_file_path = []
    start = time.time()
    for input_files in files_grouped:
        outputss = []
        num_fits_init = 10
        numcpus = multiprocessing.cpu_count()-1
        if numcpus > 60:
            numcpus = 60
        if num_fits_init < numcpus:
            numcpus = num_fits_init
        p = multiprocessing.Pool(processes = numcpus)
        ares = []
        fit_x = []
        fit_fun = []
        # input('hi')
        test = []
        test_time = time.time()
        for i in range(1):
            test.append(initial_fitting(param_bounds, gofargs, nonlincon, kinin, input_files, ''))
        print(time.time()-test_time)
        input('hi')
        for i in range(num_fits_init):
            ares.append(p.apply_async(initial_fitting,args = (param_bounds,gofargs, nonlincon, kinin, input_files, '')))
        for i in range(num_fits_init):
            res = ares[i].get()
            res_full.append(res)
            outputss.append(res)
            fit_x.append(res.x)
            fit_fun.append(res.fun)
            if i%10 == 0:
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
            in_cons = np.repeat(con0, data[0].shape[0]).reshape(data[0].shape[1],data[0].shape[0])
            in_cons[1] = neutral_con[i]
            fit_initial_cons.append(in_cons)
        
        colors = get_cmap(len(names))
        map_of_zeros = np.copy(data)
        map_of_zeros[map_of_zeros!=0] /= map_of_zeros[map_of_zeros!=0]
        vals = solve(fit_initial_cons[0],res.x[0:numk]*k_l_bounds)*map_of_zeros[0]
        
        plt.figure(figsize = [15, 10])
        for i in range(len(data)):
            sorting_index = np.argsort(neutral_con[i])
            num_analyze = i
            plt.semilogy(neutral_con[i][sorting_index],data[i][sorting_index], "o")
            plt.semilogy(neutral_con[i][sorting_index],vals[sorting_index])
            plt.ylim(np.min(data[data != 0])/10,np.max(data)*10)
        print('Global Fit took: ',round(time.time()-start,2))
        if len(data[0].shape) == 3:
            for j in range(np.delete(data[i],1,axis = 1).shape[1]):
                plt.figure()
                for i in range(len(data)):
                    num_analyze = i
                    plt.semilogy(neutral_con[i],np.delete(data[i],1,axis = 1)[:,j], "o")
                    plt.semilogy(neutral_con[i],np.delete(solve(fit_initial_cons[i],res.x[0:numk]*k_l_bounds),1,axis = 1)[:,j])
        input('hi')
        numsims = 1000
        numcpus = multiprocessing.cpu_count()-1
        if numcpus > 60:
            numcpus = 60
        if numsims < numcpus:
            numcpus = numsims
        param_stdev, fit_low, fit_high, full_sim_data, sim_params, fit_stdev, sim_gofs = error_analysis(best_fit, gofargs, param_bounds, numsims, names, nonlincon, kinin, numcpus, input_files, '')
        print('Error analysis Took: ',round(time.time()-start,2))
        best_fit[0:numk] = best_fit[0:numk]*k_l_bounds
        fit_low[0:numk] = fit_low[0:numk]*k_l_bounds
        fit_high[0:numk] = fit_high[0:numk]*k_l_bounds
        combined_out = np.array([best_fit, fit_low, fit_high]).transpose()
        output_file_path.append(outputfile(input_files,combined_out,k,names))
        print(np.array([fit_low[0:numk],best_fit[0:numk],fit_high[0:numk]]).transpose())
        
        sim_gofs_full = np.copy(sim_gofs)
        sim_gofs_indices = np.argsort(sim_gofs)
        for index in range(len(names)):
            plt.figure(figsize = [15, 10])
            if np.any(full_sim_data[0][0][:,index]):
                for plotting_params in sim_params:
                    rate_constants = plotting_params[0:numk]*k_l_bounds
                    sim_initials = get_fit_initial_cons(plotting_params, (data.shape[2],data.shape[1]))[0]
                    # sim_initials[6] = neutral_con[0]
                    sim_data = solve(sim_initials,rate_constants)*map_of_zeros[0]
                    plt.semilogy(neutral_con[0],sim_data[:,index],'o',color = 'red', alpha = 0.05)
                plt.semilogy(neutral_con[0],data[0][:,index], "o", color = 'black')
                plt.title(names[index])
        
        
# small = []
# for fitnums in range(num_fits_init):
#     small.append(outputss[fitnums].fun)
# res = outputss[np.argmin(small)]
# best_fit = res.x
# outputs = []
# outputs.append(res)