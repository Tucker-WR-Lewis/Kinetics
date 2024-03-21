# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:11:49 2024

@author: Tucker Lewis
"""

import numpy as np
import matplotlib.pyplot as plt

min_val = 0.01

kvt = r"C:\Users\Tucker Lewis\Documents\AFRL\Ta+ + CH4 new\Ta+ + CH4 data\data\Ta+ + CH4_allT_simul.KVT"
with open(kvt) as f:
    file_list = f.read()
file_list = file_list.split('\n')


for file in file_list:
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
            if strings == 'Sim Gofs':
                sim_gofs_start = index
            
    params = np.genfromtxt(text_split[sim_start+1:sim_gofs_start-1])
    gofs = np.genfromtxt(text_split[sim_gofs_start+1:])
    
    quartiles = np.percentile(gofs, [25,75], axis = 0)
    k_factor = 1.5
    iqr = (quartiles[1]-quartiles[0])*k_factor
    t_fences = np.array([quartiles[0]-iqr,quartiles[1]+iqr])
    
    params_trunc = []
    fit_low = []
    fit_high = []
    indices =[]
    bad = []
    bad2 = []
    
    # indices = np.where((gofs > t_fences[0]) & (gofs < t_fences[1]))
    indices = np.where(gofs < t_fences[1])
    gofs_iqr = gofs[indices]
    gofs_high_95 = np.percentile(gofs_iqr,95)
    indices_95 = np.where(gofs_iqr < gofs_high_95)
    gofs_iqr_95 = gofs_iqr[indices_95]
    ommiteds = []
    for omitted_index, ommited in enumerate(gofs):
        if ommited not in gofs_iqr_95:
            ommiteds.append(omitted_index)
    
    for trunc_index, to_trunc in enumerate(params.transpose()):
        params_trunc.append(to_trunc[indices][indices_95])
        if len(to_trunc[indices]) > 0:
            fit_low.append(np.percentile(to_trunc[indices][indices_95],2.5))
            fit_high.append(np.percentile(to_trunc[indices][indices_95],97.5))
    
    new_params = []
    for hist_index, to_hist in enumerate(params_trunc[0:numk]):
        hist, hist_bins = np.histogram(to_hist,25)
        prob_index = np.argmax(hist)
        new_params.append(np.average([hist_bins[prob_index],hist_bins[prob_index+1]]))
    
    file_out = file[0:-4] + '_corr.txt'
    
    new_text = []
    new_text.append(text_split[0])
    for i in range(numk):
        temp_str = text_split[i+1].split()
        temp_str[1] = str(new_params[i]*1e-10)
        temp_str[2] = str(fit_low[i]*1e-10)
        temp_str[3] = str(fit_high[i]*1e-10)
        temp_str = '\t'.join(temp_str)
        new_text.append(temp_str)
    
    # kT =    (['ks',['k1','k2']],
    #          ['ks',[]],
    #          ['ks',['k3','k4','k5']],
    #          ['ks',['k6','k7']],
    #          ['ks',['k8','k9']],
    #          ['ks',['k10','k11','k12']],
    #          ['ks',['k13','k14']],
    #          ['ks',['k15','k16']],
    #          ['ks',['k17','k18']],
    #          ['ks',['k19','k20','k21']],
    #          ['ks',['k22','k23']],
    #          ['ks',['k24','k25']],
    #          ['ks',['k26','k27']],
    #          ['ks',[]],
    #          ['ks',['k28']],
    #          ['ks',['k29','k30']],
    #          ['ks',['k31','k32']])
    
    kT =    (['ks',['k1','k2']],
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
                  ['ks',['k29','k30']],
                  ['ks',['k31','k32']],
                  ['ks',['k33','k34']])
    
    kT_vals = []
    for rate_i in kT:
        vals = np.zeros(params.shape[0])
        for add_index in [int(i.strip('k'))-1 for i in rate_i[1]]:
            vals = vals + params[:,add_index]
        kT_vals.append(vals)
    kT_vals = np.array(kT_vals)
    quartiles_kT = np.percentile(kT_vals, [25,75], axis = 1)
    k_factor_kT = 1.5
    iqr_kT = (quartiles_kT[1]-quartiles_kT[0])*k_factor_kT
    t_fences_kT = np.array([quartiles_kT[0]-iqr_kT,quartiles_kT[1]+iqr_kT])
    
    params_trunc_kT = []
    fit_low_kT = []
    fit_high_kT = []
    for trunc_index_kT, to_trunc_kT in enumerate(kT_vals):
        indices_kT = np.where((to_trunc_kT > t_fences_kT[:,trunc_index_kT][0]) & (to_trunc_kT < t_fences_kT[:,trunc_index_kT][1]))
        params_trunc_kT.append(to_trunc_kT[indices_kT])
        if len(to_trunc_kT[indices_kT]) > 0:
            fit_low_kT.append(np.percentile(to_trunc_kT[indices_kT],2.5))
            fit_high_kT.append(np.percentile(to_trunc_kT[indices_kT],97.5))
        if len(to_trunc_kT[indices_kT]) == 0:
            fit_low_kT.append(0)
            fit_high_kT.append(0)
            
    new_params_kT = []
    for hist_index_kT, to_hist_kT in enumerate(params_trunc_kT):
        if len(to_hist_kT) == 0:
            new_params_kT.append(0)
        else:
            hist_kT, hist_bins_kT = np.histogram(to_hist_kT,25)
            prob_index_kT = np.argmax(hist_kT)
            new_params_kT.append(np.average([hist_bins_kT[prob_index_kT],hist_bins_kT[prob_index_kT+1]]))
    
    kT_stop_index = []
    for line_index, line in enumerate(text_split):
        if line.find('kT') != -1:
            kT_index_start = line_index
        if line == '':
            kT_stop_index.append(line_index)
    kT_stop_index = kT_stop_index[1]
        
    kT_string = text_split[kT_index_start] + '\n'
    for index_kT, line_kT in enumerate(text_split[kT_index_start+1:kT_stop_index]):
        line_kT = line_kT.split()
        if index_kT < len(new_params_kT):
            line_kT[1] = str(new_params_kT[index_kT]*1e-10)
            line_kT[2] = str(fit_low_kT[index_kT]*1e-10)
            line_kT[3] = str(fit_high_kT[index_kT]*1e-10)
        if index_kT >= len(new_params_kT):
            line_kT[1] = str(0)
            line_kT[2] = str(0)
            line_kT[3] = str(0)
        kT_string = kT_string + '\t'.join(line_kT) + '\n'
    
    new_text = '\n'.join(new_text)
    new_text = new_text + '\n\n'  + kT_string
    
    for strings in text_split[kT_stop_index:]:
        new_text = new_text + strings + '\n'
    # new_text = '\n'.join(new_text)    
    
    f = open(file_out, 'w')
    f.write(new_text)
    f.close()
    
kvt_out_name = kvt[0:-4] + '_corr.KVT'
kvt_out = []
for string in file_list:
    kvt_out.append(string[0:-5]+'_corr.txt')
kvt_out = '\n'.join(kvt_out)

f = open(kvt_out_name, 'w')
f.write(kvt_out)
f.close()