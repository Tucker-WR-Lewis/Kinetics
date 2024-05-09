# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:32:32 2024

@author: Tucker Lewis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sympy as sym
import string

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

def plot_data(data_temp, ks_indices_temp, data_indices):
    ret = []
    for k_index in ks_indices_temp:
        ret_temp = []
        for index, part in enumerate(data_temp):
            ret_temp.append(part[data_indices,k_index].flatten())
        ret.append(ret_temp)
    return ret

def sum_plot_data(data_temp, ks_indices_temp, data_indices):
    ret = []
    for k_index in ks_indices_temp:
        ret_temp = []
        for index, part in enumerate(data_temp):
            ret_temp.append(part[data_indices,k_index].flatten())     
        ret.append(ret_temp)
    outs = []
    for tosum in range(len(ret[0])):
        sums = []
        for index in range(len(ret)):
            sums.append(ret[index][tosum])
        outs.append(np.sum(sums, axis = 0))
    return outs

kvt = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta(CH2)+ +CH4\data\Ta(CH2)+ + CH4_allT_simul.KVT"
kvt = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta+ and Ta(CH2) simul\Ta+ and Ta(CH2)+ simul.KVT"
kvt = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta(C3H2)+ and Ta(C3H4)+\Ta(C3H2)+ + CH4 300K simul.KVT"
kvt = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta(C2H2)+ + CH4\TaC2H2+ + CH4_34reactions_allT_simul_forplotter.KVT"
kvt = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Comparison2.KVT"
kinin = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta+ + CH4_34reactions.KININ"

kvt = r"C:\Users\Tucker Lewis\Documents\AFRL\N3+ N4+\testing\N4+ testing.KVT"
kvt = r"C:\Users\Tucker Lewis\Documents\AFRL\N3+ N4+\testing\N3+ and N4+ simul fit_tesdting.KVT"
kinin = r"C:\Users\Tucker Lewis\Documents\AFRL\N3+ N4+\testing\N4+ testing_6.KININ"
ydot, y, k, k_l_bounds, k_h_bounds, species_0, constraints_new, con_limits_low, con_limits_high, names, reactmap, prodmap, iso_index, mass_descrim = getodes(kinin)

with open(kvt) as f:
    file_list = f.read()
file_list = file_list.split('\n')

ks = []
species = []
temps = []
data = []
gofs = []
blanks = []
for file in file_list:
    ks_temp = []
    species_temp = []
    temps.append(int(file[0:3]))
    file = file[6:-1]
    
    with open(file) as f:
        text = f.read()
    text_split = text.split('\n')    
    
    stop = 0
    for index, strings in enumerate(text_split):
        if strings == 'parameter\tvalue\tlow\thigh':
            ks_start = index+1
        if strings == '':
            blanks.append(index)
        if strings == 'Sim Params':
            sim_start = index
        if len(strings) > 0:
            if strings[0] != 'p' and strings[0] != 'k' and stop == 0:
                stop = 1
                numk = index-1
        if strings == 'Sim Gofs':
            sim_gofs_start = index
    temp_data_loop = np.genfromtxt(text_split[sim_start+1:sim_gofs_start-1])
    temp_data_loop[:,0:numk] = temp_data_loop[:,0:numk]*k_l_bounds
    data.append(temp_data_loop)
    gofs.append(np.genfromtxt(text_split[sim_gofs_start+1:]))

data = np.array(data)
gofs = np.array(gofs)
# quartiles = np.percentile(gofs, [25,75], axis = 1)
k_factor = 1.5
# iqr = (quartiles[1]-quartiles[0])*k_factor
# t_fences = np.array([quartiles[0]-iqr,quartiles[1]+iqr])

q1 = np.percentile(gofs,25,axis = 1)
q2 = np.percentile(gofs,75,axis = 1)
q3 = np.percentile(gofs,50,axis = 1)
t_fences = [q1 - k_factor*(q3-q1),q2+k_factor*(q2-q3)]

# indices = []
# gofs_trunc = []
# indices_95 = []
# for gof_temp_index, gof_temp in enumerate(gofs):
#     indices.append(np.where(gof_temp < t_fences[1][gof_temp_index]))
#     gofs_trunc.append(gof_temp[indices[gof_temp_index]])
# for gof_temp_index, gof_trunc_temp in enumerate(gofs_trunc):
#     gofs_high_95 = np.percentile(gof_trunc_temp,95)
#     indices_95.append(np.where(gof_trunc_temp < gofs_high_95)[-1])
#     gofs_iqr_95 = gof_trunc_temp[np.where(gof_trunc_temp < gofs_high_95)[-1]]
# indices = np.array(indices_95)

a = np.where(gofs[0] < np.max(t_fences))[0]
b = np.where(gofs[0] < np.percentile(gofs[0][a],95))[0]
indices = np.copy(b)

for index, strings in enumerate(text_split[ks_start:blanks[0]]):
    if strings.split()[0][0] == 'k' and strings.split()[0][1].isdigit():
        ks.append(strings.split()[0])
    if strings.split()[0][0] != 'k':
        species.append(strings.split()[0])
species.remove(reactmap[0][1])

temps = np.array(temps)
ks = np.array(ks)

Plots =     (['ks','all'],)

# Plots =    (['ks',['k1','k2']],
#               ['ks',['k3','k4','k5']],
#               ['ks',['k6','k7']],
#               ['ks',['k8','k9']],
#               ['ks',['k10','k11','k12']],
#               ['ks',['k13','k14','k15']],
#               ['ks',['k16','k17']],
#               ['ks',['k18','k19']],
#               ['ks',['k20','k21','k22']],
#               ['ks',['k23','k24']],
#               ['ks',['k25','k26']],
#               ['ks',['k27','k28']],
#               ['ks',['k29']],
#               ['ks',['k30','k31']],
#               ['ks',['k32','k33']])

# Plots = (['ks',['k1','k2']],)

# Plots = (['ks',['k6','k7','k8','k9','k10','k12','k13','k14','k15','k16','k17','k18']],)
# Plots = (['ks',['k17','k23','k24','k25','k27']],)
Plots = (['ks',['k5','k6','k7','k8','k9','k10']],)
# Plots = (['ks',['k11','k12','k13','k14','k15','k16','k17','k26']],)
# Plots = (['ks',['k15']],)

# Plots = (['IC','all'],)
# Plots = (['IC',[i for i in names if '+' in i]],)
# Plots = (['IC',names[0]],)


# Plots =     (['kT','all'],)
# Plots =     (['kT','Ta+'],
#               ['kT','Ta(CH2)+'])

# Plots = (['compare',[]],)

for plot in Plots:
    plt.figure(figsize = (10.5,7.5))
    axes = plt.axes()
    temp_data = []
    legends = []
    if plot[0] == 'ks':
        if plot[1] == 'all':
            for chem_compare in species:
                ks_indices_temp = []
                for chem_index, chem in enumerate(reactmap):
                    if chem_compare in chem:
                        ks_indices_temp.append(chem_index)
                if np.any(np.sum(data[:,:,ks_indices_temp],axis = 2)):
                    legends.append(ks[ks_indices_temp])
                    temp_data.append(plot_data(data,ks_indices_temp,indices))
        else:
            ks_indices_temp = []
            for ks_index, cur_ks in enumerate(ks):
                if cur_ks in plot[1]:
                    ks_indices_temp.append(ks_index)
            temp_data.append(plot_data(data,ks_indices_temp,indices))
            legends.append(ks[ks_indices_temp])
    
    if plot[0] == 'kT':
        titles = []
        if plot[1] == 'all':
            for chem_compare in species:
                kt_indices_temp = []
                for chem_index, chem in enumerate(reactmap):
                    if chem_compare in chem:
                        kt_indices_temp.append(chem_index)
                if np.any(np.sum(data[:,:,kt_indices_temp],axis = 2)):
                    titles.append(chem_compare)
                    temp_data.append([sum_plot_data(data,kt_indices_temp,indices)])
        else:
            kt_indices_temp = []
            for chem_index, chem in enumerate(reactmap):
                if plot[1] in chem:
                    kt_indices_temp.append(chem_index)
            if np.any(np.sum(data[:,:,kt_indices_temp],axis = 2)):
                titles.append(plot[1])
                temp_data.append([sum_plot_data(data,kt_indices_temp,indices)])  
    
    if plot[0] == 'compare':
        pos = ["Ta+", "Ta(CH2)+", "Ta(C2H2)+", "Ta(C3H2)+", "Ta(C3H4)+", "Ta+ and Ta(CH2)+"]
        pos = ["Ta+", "Ta(CH2)+", "Ta(C2H2)+", "Ta(C3H2)+", "Ta(C3H4)+"]
        # pos = pos[0:3]
        legends = ['300K', '400K', '500K', '600K']
        for k_num in range(ks.shape[0]):
            temp_data_k = []
            for T in np.unique(temps):
                temp_data_temp = []
                same_T = np.where(temps == T)
                temp_comp = data[np.where(temps == T)]
                for comp_index, temp_data_part in enumerate(temp_comp):
                    temp_data_temp.append(temp_data_part[indices[same_T[0][comp_index]],k_num][0])
                temp_data_k.append(temp_data_temp)
            temp_data.append(temp_data_k)
    
    if plot[0] == 'IC':
        if plot[1] == 'all':
            ICs = names
        else:
            ICs = plot[1]
        ICs_index = []
        for name in ICs:
            ICs_index.append(numk+names.index(name))
        temp_data.append(plot_data(data,ICs_index,indices))
        legends.append(np.array(names)[[i-numk for i in ICs_index]])
    
    for title_index, big_items in enumerate(temp_data):
        labels = []
        a = 1
        comp_adjust = 0.5
        for offset, items in enumerate(big_items):
            leg_index = offset
            if plot[0] == 'compare':
                temps = [i+0.2*offset for i in range(len(items))]
                offset = []
                comp_adjust = 0.01
            parts = axes.violinplot(items, positions = temps+offset*10, widths = 20*comp_adjust, points = 100, showmedians=True,showextrema=False)
            ax = plt.gca()
            if plot[0] == 'ks' or 'IC':
                color = parts["bodies"][0].get_facecolor().flatten()
                labels.append((mpatches.Patch(color=color), legends[title_index][offset]))
            if plot[0] == 'compare':
                color = parts["bodies"][0].get_facecolor().flatten()
                labels.append((mpatches.Patch(color=color), legends[leg_index]))
        ax.set_xticks(temps, temps)
        ax.set_yscale('log')
        if plot[0] =='kT':
            plt.title(titles[title_index])
        if plot[0] == 'ks' or 'IC':
            plt.legend(*zip(*labels), frameon = False)  
        if plot[0] == 'compare':
            ax.set_xticks([i for i in range(len(pos))], labels = pos) 
            plt.legend(*zip(*labels), frameon = False, loc = 'lower left')
            plt.title(ks[title_index])
        if plot[0] =='kT' or plot[1] == 'all' or plot[0] == 'compare':
            plt.figure(figsize = (10.5,7.5))
            axes = plt.axes()