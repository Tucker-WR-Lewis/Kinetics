# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 13:54:13 2024

@author: Tucker Lewis
"""

import numpy as np
import matplotlib.pyplot as plt

def powspace(start, stop, power, num):
    start = np.power(start, 1/float(power))
    stop = np.power(stop, 1/float(power))
    return np.power( np.linspace(start, stop, num=num), power) 

kvt = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta+ + CH4 data\33_reactions_iqr\33_reactions_1.KVT"
with open(kvt) as f:
    file_list = f.read()
file_list = file_list.split('\n')

labels = []
data = []
data2 = []
for file in file_list:
    labels.append(int(file[0:3]))
    file = file[6:-1]
    
    with open(file) as f:
        text = f.read()
    text_split = text.split('\n')
    
    stop = 0
    for index, strings in enumerate(text_split):
        if strings == 'Sim Params':
            sim_start = index
        if len(strings) > 0:
            if strings[0] != 'p' and strings[0] != 'k' and stop == 0:
                stop = 1
                numk = index-1
            
    data.append(np.genfromtxt(text_split[sim_start+1:])[:,0]*1e-10)
    data2.append(np.genfromtxt(text_split[sim_start+1:])[:,1]*1e-10)
labels = np.array(labels)

number_of_bins = 1000
num_data = 500
# labels = [150,200,300,400,500,600]
k150 = np.random.normal(1e-10,3e-10*0.2,num_data)
k200 = np.random.normal(3e-10,3e-10*0.2,num_data)
k300 = np.abs(np.random.normal(5e-11,1e-11*0.5,num_data))
k400 = np.random.normal(1e-9,1e-9*0.05,num_data)
k500 = np.random.normal(3e-10,3e-10*0.2,num_data)
k600 = np.random.normal(7e-11,7e-11*0.2,num_data)    

# data = np.array([k150, k200, k300, k400, k500, k600])

hist_range = (np.min(data),np.max(data))
binned_datas = [np.histogram(d, range = hist_range, bins = number_of_bins)[0] for d in data]
binned_max = np.max(binned_datas, axis = 1)
x_locations = np.arange(0,sum(binned_max), np.max(binned_max))

# # bin_edges = 10**np.linspace(np.log(hist_range[0]),np.log(hist_range[1]), number_of_bins+1)
# bin_edges = powspace(hist_range[0], hist_range[1], 10, number_of_bins+1)
# heights = np.diff(bin_edges)
# centers = bin_edges[0:-1] + heights / 2

# fig, ax = plt.subplots()
# for x_loc, binned_data in zip(labels,binned_datas):
#     lefts = x_loc - 0.5 * binned_data
#     ax.barh(centers, binned_data, height = heights, left = lefts)

fig, axs = plt.subplots(nrows = 1, ncols = 1)
dic = plt.violinplot(data, positions = labels, widths = 25, points = 100)
dic = plt.violinplot(data2, positions = labels+5, widths = 25, points = 100)
ax = plt.gca()
ax.set_xticks(labels, labels)
ax.set_yscale('log')
# ax.set_ylabel("Rate Constant Values")
# ax.set_xlabel("Temperature (K)")

plt.show()