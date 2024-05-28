# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 10:14:03 2023

@author: Tucker Lewis
"""

import numpy as np
import matplotlib.pyplot as plt

# f = open("C:/Users/Tucker Lewis/Documents/AFRL/Ta+ + CH4 _2.kvt")
# f = open("C:/Users/Tucker Lewis/Documents/AFRL/Ta+ + CH4 _4.KVT")
# f = open(r"C:\Users\Tucker Lewis\Documents\AFRL\Ta(C3)+ + CH4.KVT")
# f = open("C:/Users/Tucker Lewis/Documents/AFRL/Ta(CH2)+ + CH4_2.kvt")
# f = open(r"C:\Users\Tucker Lewis\Documents\AFRL\Ta(CH2)+ + CH4_3.KVT")
f = open(r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta(CH2)+ +CH4\Ta(CH2)+ + CH4_33reactions_corr.KVT")
# f = open(r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4\Ta(CH2)+ + CH4 data analysis\Ta(CH2)+ + CH4_4.KVT")

title = "Ta+ + CH4"
file_list = f.read()
f.close()

# Plots = (['ks',['k1','k4','k6','k8','k10','k12','k14','k16','k18','k20','k22','k24','k27','k29']],)

# Plots =    ( ['ks',['k1','k2']],
#              ['ks',[]],
#              ['ks',['k3','k4','k5']],
#              ['ks',['k6','k7']],
#              ['ks',['k8','k9']],
#              ['ks',['k10','k11']],
#              ['ks',['k12','k13']],
#              ['ks',['k14','k15']],
#              ['ks',['k16','k17']],
#              ['ks',['k18','k19']],
#              ['ks',['k20','k21']],
#              ['ks',['k22','k23']],
#              ['ks',['k24','k25']],
#              ['ks',[]],
#              ['ks',['k26']],
#              ['ks',['k27','k28']],
#              ['ks',['k29','k30']])

Plots =    (['ks',['k1','k2']],
              ['ks',[]],
              ['ks',['k3','k4','k5']],
              ['ks',['k6','k7']],
              ['ks',['k8','k9']],
              ['ks',['k10','k11','k12']],
              ['ks',['k13','k14','k15']],
              ['ks',['k16','k17']],
              ['ks',['k18','k19']],
              ['ks',['k20','k21','k22']],
              ['ks',['k23','k24']],
              ['ks',['k25','k26']],
              ['ks',['k27','k28']],
              ['ks',[]],
              ['ks',['k29']],
              ['ks',['k30','k31']],
              ['ks',['k32','k33']])

# Plots =    (['ks',['k1','k2']],
#              ['ks',[]],
#              ['ks',['k3','k4','k5']],
#              ['ks',['k6','k7']],
#              ['ks',['k8','k9']],
#              ['ks',['k10','k11','k12']],
#              ['ks',['k13','k14']],
#              ['ks',['k15','k16']],
#              ['ks',['k17','k18']],
#              ['ks',['k19','k20','k21']],
#              ['ks',['k22','k23']],
#              ['ks',['k24','k25']],
#              ['ks',['k26','k27']],
#              ['ks',[]],
#              ['ks',['k28']],
#              ['ks',['k29','k30']],
#              ['ks',['k31','k32']])

# Plots = (['ks',['k1','k2']],
#          ['ks',['k3','k4']],
#          ['ks',['k5','k6']],
#          ['ks',['k7','k8']],
#          ['ks',['k9','k10']],
#          ['ks',['k11']],
#          ['ks',['k12','k13']],
#          ['ks',['k14','k15']],
#          ['ks',['k16','k17']],)

# Plots = (['ks',['k1','k2']],
#          ['ks',['k3','k4']],
#          ['ks',['k5']],
#          ['ks',['k6']],
#          ['ks',['k7']],)

# Plots = (['ks',['k1','k2','k3']],)
# Plots = ['ks',['k1']]
# Plots = (['kT', ['all']],)
# Plots = (['kT',['Ta+','Ta(CH2)+','Ta(C2H4)+','Ta(C3H6)+']],)
# Plots = ['io', ['Ta(C5H16)+']]
# title = '$Ta^+$ + CD4'


file_list = file_list.replace(r'"', '')
file_list_split = file_list.split('\n')

Tvals = []
for file in file_list_split:
    Tvals.append(int(file[0:3]))
Tvals = np.array(Tvals)

files = []
for file in file_list_split:
    f = open(file[5:])
    temp_file = f.read()
    temp_file = temp_file.split('\n')
    files.append(temp_file)
    f.close()

blanks = []
for index, line in enumerate(files[0]):
    if line == '':
        blanks.append(index)
blanks = np.array(blanks)


ks = []
ks_name = []
for line in files[0]:
    if line == '':
        break
    if line.split()[0][0] == 'k':
        ks_name.append(line.split()[0])
        ks.append(float(line.split()[1]))
ks_name = np.array(ks_name)

all_ks = []
for file in files:
    for line in file:
        if line == '':
            break
        if line.split()[0][0] == 'k':
            all_ks.append(float(line.split()[1]))
all_ks = np.array(all_ks)

avgs = []
for Plot in Plots:
    if Plot[0] == 'ks':
        plt.figure()
        forplot = []
        for search in Plot[1]:
            temp_list = []
            for file in files:
                for line in file:
                    if line != '':
                        if search == line.split()[0]:
                            floated = []
                            for tofloat in line.split()[1:4]:
                                floated.append(float(tofloat))
                            temp_list.append(floated)
            forplot.append(temp_list)
        forplot = np.array(forplot)
        for offset, plots in enumerate(forplot):
            errors = np.array([plots[:,0]-plots[:,1], plots[:,2]-plots[:,0]])
            # plt.errorbar(Tvals,plots[:,0]/9.75e-10, yerr = (plots[:,0] - plots[:,1])/9.75, marker = 'o', linestyle='None', capsize = 5)
            plt.errorbar(Tvals+offset*5,plots[:,0], yerr = errors, marker = 'o', linestyle='None', capsize = 5)
        # plt.yscale('linear')
        plt.yscale('log')
        plt.legend(Plot[1], loc = 4)
        plt.title(title)
        plt.xlabel('Temperature (K)')
        plt.ylabel('Rate Constant')
        # plt.ylim([0,1])
        plt.xlim([125,625])
        # plt.xlim([275,625])
        # plt.xlim([295,395])
        # plt.ylim([1e-10,1e-9])
        
        for plots in forplot:
            vals = []
            errs = []
            for T in np.unique(Tvals):
                val = []
                err = []
                for i, temp in enumerate(Tvals):
                    if T == temp:
                        val.append(plots[i,0])
                        err.append(plots[i,1])
                val = np.array(val)
                err = np.array(err)
                vals.append(np.average(val,weights = val - err))
                errs.append(np.average(err,weights = val - err))
            vals = np.array(vals)
            errs = np.array(errs)
            avgs.append(np.array([vals,vals - errs]).transpose().flatten())
        
    if Plot[0] == 'kT':
        if Plot[1][0] == 'all':
            new_search = []
            for line in file[blanks[0]+2:blanks[1]]:
                if line.split()[1] != '0':
                    new_search.append(line.split()[0])
            Plot[1] = new_search
        forplot = []
        for search in Plot[1]:
            temp_list = []
            for file in files:
                for line in file[blanks[0]+2:blanks[1]]:
                    if search == line.split()[0]:
                        floated = []
                        for tofloat in line.split()[1:]:
                            floated.append(float(tofloat))
                        temp_list.append(floated)
            forplot.append(temp_list)
        forplot = np.array(forplot)
        for index, plots in enumerate(forplot):
            plt.figure()
            errors = np.array([plots[:,0] - plots[:,1], plots[:,2] - plots[:,0]])
            
            
            plt.errorbar(Tvals,plots[:,0], yerr = errors, marker = 'o', linestyle='None', capsize = 5)
            plt.title(Plot[1][index])
        # plt.yscale('linear')
            plt.yscale('log')
            # plt.legend(Plot[1], loc = 4)
            # plt.title(title)
            plt.xlabel('Temperature (K)')
            plt.ylabel('Rate Constant')
            plt.xlim([125,625])
            # plt.xlim([275,625])
    
    if Plot[0] == 'io':
        loss = []
        formed = []
        for search in Plot[1]:
            temp_list = []
            for file in files:
                for line in file[blanks[-3]+1:blanks[-2]]:
                    if search == line.split()[0]:
                        floated = []
                        for tofloat in line.split()[1:]:
                            floated.append(float(tofloat))
                        temp_list.append(floated)
            loss.append(temp_list)
        loss = np.array(loss)
        
        loss_ks = np.intersect1d(ks,loss[0][0],return_indices=True)
        loss_ks = ks_name[loss_ks[1]]
        
        for plots in loss:
            if np.any(plots):
                    for i in range(int(plots.shape[1]/2)):
                        plt.errorbar(Tvals,plots[:,i], yerr = plots[:,i+1], marker = 'o', linestyle='None', capsize = 5)
        
        for search in Plot[1]:
            temp_list = []
            for file in files:
                for line in file[blanks[-2]+1:blanks[-1]]:
                    if search == line.split()[0]:
                        floated = []
                        for tofloat in line.split()[1:]:
                            floated.append(float(tofloat))
                        temp_list.append(floated)
            formed.append(temp_list)
        formed = np.array(formed)
        
        form_ks = np.intersect1d(ks,formed[0][0],return_indices=True)
        form_ks = ks_name[form_ks[1]]
        
        for plots in formed:
            if np.any(plots):
                    for i in range(int(plots.shape[1]/2)):
                        plt.errorbar(Tvals,plots[:,i], yerr = plots[:,i+1], marker = 's', linestyle='None', capsize = 5)
        
        plt.title(Plot[1])
        plt.yscale('log')
        plt.xlabel('Temperature (K)')
        plt.ylabel('Rate Constant')
        plt.legend(np.append(loss_ks,form_ks))
        
avgs = np.array(avgs)