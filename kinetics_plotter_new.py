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
    for index, con in enumerate(text_out):
        if con == 'Conditions':
            # text_con = text_out[index+1:iso_index]
            text_out = text_out[0:index]
            break    
    for index, i in enumerate(text_out):      
        for j in i:
            if i[0] == 'k' and i[1].isnumeric():
                k_out.append(i)
                r_start.append(index + 3)
                p_end.append(index)
            if j.isalpha():
                break
            if j == '>':
                r_end.append(index)
                p_start.append(index+1)
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
    sym.init_printing()
    return reactants3, products3

def plot_data(data_temp, ks_indices_temp, data_indices):
    ret = []
    for k_index in ks_indices_temp:
        ret_temp = []
        for index, part in enumerate(data_temp):
            ret_temp.append(part[data_indices[index],k_index].flatten())
        ret.append(ret_temp)
    return ret

def sum_plot_data(data_temp, ks_indices_temp, data_indices):
    ret = []
    for k_index in ks_indices_temp:
        ret_temp = []
        for index, part in enumerate(data_temp):
            ret_temp.append(part[data_indices[index],k_index].flatten())     
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

reactmap, prodmap = getodes(kinin)

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
    data.append(np.genfromtxt(text_split[sim_start+1:sim_gofs_start-1])[:,0:numk]*1e-10)
    gofs.append(np.genfromtxt(text_split[sim_gofs_start+1:]))

data = np.array(data)
gofs = np.array(gofs)
quartiles = np.percentile(gofs, [25,75], axis = 1)
k_factor = 1.5
iqr = (quartiles[1]-quartiles[0])*k_factor
t_fences = np.array([quartiles[0]-iqr,quartiles[1]+iqr])
indices = []
for gof_temp_index, gof_temp in enumerate(gofs):
    indices.append(np.where(gof_temp < t_fences[1][gof_temp_index]))

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

# Plots =     (['kT','all'],)

# Plots =     (['kT','Ta+'],
#               ['kT','Ta(CH2)+'])

Plots = (['compare',[]],)

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
    
    for title_index, big_items in enumerate(temp_data):
        labels = []
        a = 1
        comp_adjust = 1
        for offset, items in enumerate(big_items):
            leg_index = offset
            if plot[0] == 'compare':
                temps = [i+0.2*offset for i in range(len(items))]
                offset = []
                comp_adjust = 0.01
            parts = axes.violinplot(items, positions = temps+offset*10, widths = 20*comp_adjust, points = 100, showmedians=True,showextrema=False)
            ax = plt.gca()
            if plot[0] == 'ks':
                color = parts["bodies"][0].get_facecolor().flatten()
                labels.append((mpatches.Patch(color=color), legends[title_index][offset]))
            if plot[0] == 'compare':
                color = parts["bodies"][0].get_facecolor().flatten()
                labels.append((mpatches.Patch(color=color), legends[leg_index]))
        ax.set_xticks(temps, temps)
        ax.set_yscale('log')
        if plot[0] =='kT':
            plt.title(titles[title_index])
        if plot[0] == 'ks':
            plt.legend(*zip(*labels), frameon = False)  
        if plot[0] == 'compare':
            ax.set_xticks([i for i in range(len(pos))], labels = pos) 
            plt.legend(*zip(*labels), frameon = False, loc = 'lower left')
            plt.title(ks[title_index])
        if plot[0] =='kT' or plot[1] == 'all' or plot[0] == 'compare':
            plt.figure(figsize = (10.5,7.5))
            axes = plt.axes()