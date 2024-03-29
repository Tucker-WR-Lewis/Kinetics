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

def get_data(batchin_temp):
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
        
def get_plot_data():
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
        rate_constants = params[0:numk]
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

def get_cmap(n):
    if n < 11:
        return mpl.colormaps['tab10']
    if n < 20 and n > 11:
        return mpl.colormaps['tab20']

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
        
        weighted = res_per_square*np.sqrt(np.abs(ydata+1)) #changed from 1 to 0.01 to try and reduce impact of 0 or low count data
        final_res.append(np.sum(weighted**2))
    return np.sum(final_res)

kinin = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta+ + CH4_34reactions.KININ"
rois = ''

batchin = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta+ and Ta(CH2) simul\Ta+ and Ta(CH2)+ simul.BATCHIN"
kvt = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta+ and Ta(CH2) simul\Ta+ and Ta(CH2)+ simul.KVT"

batchin = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta(C3H2)+ and Ta(C3H4)+\Ta(C3H2)+ + CH4 300K simul.BATCHIN"
kvt = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta(C3H2)+ and Ta(C3H4)+\Ta(C3H2)+ + CH4 300K simul.KVT"

batchin = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta(C3H2)+ and Ta(C3H4)+\Ta(C3H4)+ + CH4 300K simul.BATCHIN"
kvt = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta(C3H2)+ and Ta(C3H4)+\Ta(C3H4)+ + CH4 300K simul.KVT"

batchin = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta(C2H2)+ + CH4\TaC2H2+ + CH4_34reactions_allT_simul.BATCHIN"
kvt = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta(C2H2)+ + CH4\TaC2H2+ + CH4_34reactions_allT_simul.KVT"

ydot, y, k, k_l_bounds, k_h_bounds, species_0, constraints_new, con_limits_low, con_limits_high, names, reactmap, prodmap, iso_index = getodes(kinin)
t = sym.symbols('t')
f_lamb = sym.lambdify((t, y) + k, ydot, "numpy")
f_jit = nb.njit(f_lamb)
numk = len(k)

trunc_params, ommited_fits, temps = get_truncated_params(kvt)
data, rxntime, neutral_reactant, neutral_con, initial_cons, names = get_data(batchin)

plotting_temp = 600

to_plot, ommited_to_plot, plotting_indices = get_plot_data()

# species = ['Ta+','Ta(CH2)+']
# species = ['Ta+',]
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
file_path = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta(C3H2)+ and Ta(C3H4)+\new figures" + r'\{}'.format(plotting_temp)
file_path = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta(C2H2)+ + CH4\new figures" + r'\{}'.format(plotting_temp)

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