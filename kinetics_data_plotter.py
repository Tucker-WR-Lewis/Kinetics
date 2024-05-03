# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 10:51:11 2024

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
import matplotlib as mpl
import pathlib

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

def get_truncated_params(kvt_temp):
    global sim_gofs, t_fences, indices_95, sim_params, indices, indices_95, gofs_iqr, trunc_index, text_split
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
        # quartiles = np.percentile(gofs, [25,75], axis = 0)
        k_factor = 1.5
        # iqr = (quartiles[1]-quartiles[0])*k_factor
        # t_fences = np.array([quartiles[0]-iqr,quartiles[1]+iqr])
    
        q1 = np.percentile(gofs,25)
        q2 = np.percentile(gofs,75)
        q3 = np.percentile(gofs,50)
        t_fences = [q1 - k_factor*(q3-q1),q2+k_factor*(q2-q3)]
        
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

def get_data(batchin_temp):
    global file_list, BLS
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
        if '.BATCHEXP' or '.batchexp' in file:
            rxntimes, neutral_reactants, datas, neutral_cons, initial_conss = batch_import(names, file, iso_index)
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
    data = np.array(newdata)-BLS
    return data, rxntime, neutral_reactant, neutral_con, initial_cons, names
        
def get_plot_data():
    global num_analyze, initial_vals, rate_constants, plotting_indices, BLS
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
        rate_constants = params[0:numk]*k_l_bounds
        plot_data_temp = []
        for num_analyze in range(int((trunc_params[plotting_indices[0]].shape[1] - numk)/num_cons)):
            in_cons = params[numk+num_analyze*num_cons:numk+num_analyze*num_cons+num_cons]
            initial_vals = np.repeat(in_cons,num_species).reshape(num_cons,num_species)
            initial_vals[1] = neutral_con[plotting_indices[num_analyze]]
            plot_initial_cons.append(initial_vals)
            sorting_index = np.argsort(neutral_con[plotting_indices[num_analyze]])
            plot_data_temp.append(solve(initial_vals,rate_constants)[sorting_index])
        plot_data.append(plot_data_temp)
    for index, params in enumerate(ommited_fits[plotting_indices[0]]):
        plot_initial_cons = []
        rate_constants = params[0:numk]*k_l_bounds
        plot_data_temp = []
        for num_analyze in range(int((trunc_params[plotting_indices[0]].shape[1] - numk)/num_cons)):
            in_cons = params[numk+num_analyze*num_cons:numk+num_analyze*num_cons+num_cons]
            initial_vals = np.repeat(in_cons,num_species).reshape(num_cons,num_species)
            initial_vals[1] = neutral_con[plotting_indices[num_analyze]]
            plot_initial_cons.append(initial_vals)
            sorting_index = np.argsort(neutral_con[plotting_indices[num_analyze]])
            plot_data_temp.append(solve(initial_vals,rate_constants)[sorting_index])
        ommited_data.append(plot_data_temp)
    return plot_data, ommited_data, plotting_indices

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
    error = 100
    while t < t_final:
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
    return np.array(y).transpose()

def get_cmap(n):
    if n < 11:
        return mpl.colormaps['tab10']
    if n < 20 and n > 11:
        return mpl.colormaps['tab20']

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
        initial_con_0_temp = (initial_c[:,0] + initial_c[:,-1])/2
        initial_con_0_temp[1] = 0
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
                con_l_bound.append(con*0.75)
                con_h_bound.append(con*1.25)
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
    
    return [param_bounds, gofargs, nonlincon, files_grouped, initial_cons, species_0, f_jit, rxntime, k, names, ydot, constraints_new, neutral_reactant, reactmap]

def con_fun(params):
    con_vals = []
    for cons in constraints_new:
        con_vals.append(eval(cons))
    return np.array(con_vals)

kinin = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta+ + CH4_34reactions.KININ"
rois = ''

batchin = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta+ and Ta(CH2) simul\Ta+ and Ta(CH2)+ simul.BATCHIN"
kvt = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta+ and Ta(CH2) simul\Ta+ and Ta(CH2)+ simul.KVT"

# batchin = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta(C3H2)+ and Ta(C3H4)+\Ta(C3H2)+ + CH4 300K simul.BATCHIN"
# kvt = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta(C3H2)+ and Ta(C3H4)+\Ta(C3H2)+ + CH4 300K simul.KVT"

# batchin = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta(C3H2)+ and Ta(C3H4)+\Ta(C3H4)+ + CH4 300K simul.BATCHIN"
# kvt = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta(C3H2)+ and Ta(C3H4)+\Ta(C3H4)+ + CH4 300K simul.KVT"

batchin = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta(C2H2)+ + CH4\TaC2H2+ + CH4_34reactions_allT_simul.BATCHIN"
kvt = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta(C2H2)+ + CH4\TaC2H2+ + CH4_34reactions_allT_simul.KVT"

# batchin = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta+ + CH4 data\data\Ta+ + CH4_allT_simul_2.BATCHIN"
# kvt = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta+ + CH4 data\data\Ta+ + CH4_allT_simul_full.KVT"
# kvt = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta+ + CH4 data\data\34 reactions final\Ta+ + CH4 34 reactions final_2.KVT"

# batchin = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta(CH2)+ +CH4\data\Ta(CH2)+ + CH4_allT_simul.BATCHIN"
# kvt = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta(CH2)+ +CH4\data\34 reactions final\Ta(CH2)+ + CH4 34 reactions final.KVT"

batchin = r"C:\Users\Tucker Lewis\Documents\AFRL\N3+ N4+\testing\N3+ testing.BATCHIN"
kvt = r"C:\Users\Tucker Lewis\Documents\AFRL\N3+ N4+\testing\N3+ testing.KVT"
kinin = r"C:\Users\Tucker Lewis\Documents\AFRL\N3+ N4+\testing\N4+ testing_5.KININ"

batchin = r"C:/Users/Tucker Lewis/Documents/AFRL/N3+ N4+/testing/N3+ and N4+ smiul testing.BATCHIN"
kvt = r"C:/Users/Tucker Lewis/Documents/AFRL/N3+ N4+/testing/N3+ and N4+ simul fit_tesdting.KVT"
kinin = r"C:\Users\Tucker Lewis\Documents\AFRL\N3+ N4+\testing\N4+ testing_6.KININ"

BLS = np.array([10,10,0,0,0])[...,np.newaxis,np.newaxis]
# BLS = 0

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
neutral_reactant = inputs_tuple[12]
reactmap = inputs_tuple[13]

numpoints = gofargs[0]
numk = gofargs[1]
data = gofargs[2]
neutral_con = gofargs[3]
iso_index = gofargs[4]
k_l_bounds = gofargs[5]

trunc_params, ommited_fits, temps = get_truncated_params(kvt)

plotting_temp = 300

to_plot, ommited_to_plot, plotting_indices = get_plot_data()

# species = ['Ta+','Ta(CH2)+']
# species = ['Ta+',]
species = 'all'
species_index = []
if species != 'all':
    for spec in species:
        species_index.append(np.where(np.array(names) == spec)[0][0])
else:
    species_index = [item for item in range(len(names))]

num_plots = len(plotting_indices)
cmap = get_cmap(num_plots)
used_names = np.array(names)[plotting_indices]
# used_names = np.array(['N4+ 005','N3+ 009'])
used_names = np.array(['N3+ 009', 'N4+ 005','N3+ 004','N3+ 005','N4+ 002'])

ommited_to_plot = []

ymin = np.min(np.abs(np.array(data)[np.nonzero(data)]))/10
ymax = np.max(np.delete(data,1,axis = 2))*2

plt.rcParams['font.size'] = 15

file_path = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta+ and Ta(CH2) simul\new_figures" + r'\{}'.format(plotting_temp)
# file_path = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta(C3H2)+ and Ta(C3H4)+\new figures" + r'\{}'.format(plotting_temp)
file_path = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta(C2H2)+ + CH4\new figures" + r'\{}'.format(plotting_temp)
# file_path = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta+ + CH4 data\data\new_figures" + r'\{}'.format(plotting_temp)
# file_path = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta+ + CH4 data\data\34 reactions final\new figures" + r'\{}'.format(plotting_temp)
# file_path = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta(CH2)+ +CH4\data\34 reactions final\new figures" + r'\{}'.format(plotting_temp)
file_path = r"C:\Users\Tucker Lewis\Documents\AFRL\N3+ N4+\testing\figures\simul"

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
        sorting_index = np.argsort(neutral_con[plotting_indices[num_plot]])
        leg_handles.append(plt.semilogy(np.sort(neutral_con[plotting_indices[num_plot]]), plots[:,species_plot][sorting_index], 'o', color = cmap(num_plot), label = used_names[num_plot], 
                                        markeredgecolor = 'black', markersize = 10))
    title = names[species_plot] + ' {}K'.format(plotting_temp)
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