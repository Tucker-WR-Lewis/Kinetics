# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:53:16 2023

@author: Tucker Lewis
"""
import numpy as np
import tkinter as tk
from tkinter import filedialog

def batch_import():
    global rxntime_temp, neutral_reactant_temp, data_temp, neutral_con_temp, initial_cons_temp, initial_cons, data, species_temp, names
    filename = filedialog.askopenfilename()
    f = open(filename)
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
    
    species_temp = []
    for lists in string_list[start_index+1:]:
        for spec in lists[1::2]:
            if spec != '':
                species_temp.append(spec)
                
    species_temp = np.array(species_temp)
    trash, idx = np.unique(species_temp, return_index = True)
    species_temp = species_temp[np.sort(idx)]    
    
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
    data_temp = data_temp.transpose()
    names = np.insert(species_temp, 0, neutral_reactant_temp)
    data_temp = np.concatenate((data_temp[:,[1,0]], data_temp[:,2:]), 1)
    
    
    tocopy = names[0]
    for i in names[1:]:
        tocopy = tocopy + '\t' + i
    tocopy = tocopy + '\n'
    for i in data_temp:
        for index, j in enumerate(i):
            if index == 0:
                tocopy = tocopy + str(j)
            else:
                tocopy = tocopy + '\t' + str(j)
        tocopy = tocopy + '\n'
    text_gui.delete("1.0", tk.END)
    text_gui.insert(tk.END,tocopy)
    
window = tk.Tk()
window.title("BatchExp to Table")
window.geometry('1000x600')
frm_toprow = tk.Frame(master = window)
button_importbatch = tk.Button(master = window, 
                          text ='Import BatchExp', 
                          command=batch_import, height = 5, width = 15)
text_gui = tk.Text(master = window, height = 30, width = 120)

button_importbatch.pack(side = 'top')
text_gui.pack(side = 'bottom')
window.mainloop()