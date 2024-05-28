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

def mk_exprs_symbs(rxns, temp_temp):
    global kD_fun
    global r, k,c,rk,t,p,r_stoich,f, coeff, anions_grouped
    # create symbols for reactants
    symbs = sym.symbols(" ".join(temp_temp))
    t = sym.symbols('t')
    # map between reactant symbols and keys in r_stoich, net_stoich
    c = dict(zip(temp_temp, symbs))
    f = {n: 0 for n in temp_temp}
    k = []    
    for coeff, r_stoich, net_stoich in rxns:
        if type(coeff) == str:
            k.append(sym.S(coeff))
            r = k[-1]*prod([sym.Function(str(c[rk]))(t)**p for rk, p in r_stoich.items()])  # EXERCISE: c[rk]**p
        else:
            # r = kD_fun*prod([sym.Function(str(c[rk]))(t)**p for rk, p in r_stoich.items()])  # EXERCISE: c[rk]**p
            r = kD_fun*[rk for rk, p in r_stoich.items()][0]  # EXERCISE: c[rk]**p

        for net_key, net_mult in net_stoich.items():
            f[net_key] += net_mult*r  # EXERCISE: net_mult*r
    return [f[n] for n in temp_temp], [sym.Function(str(i))(t) for i in symbs], tuple(k)

kinin = r"C:\Users\Tucker Lewis\Documents\AFRL\Triflic Acid\TrifAnh_1_old.KININ"
a = 13.757
b = 0.133 
global reactions
f = open(kinin)
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

anions = ' + '.join(['{}'.format(j) for j in [i for i in temp if '-' in i and i != 'e-']]) #removed (t)
# kD = '(1-({}*e-**{})*(0.003*( {} )/(0.003*( {} )+e-)))/2+0.5'.format(a,b,anions,anions)
t = sym.symbols('t')
p = 1
anions_grouped = 0
for j in [sym.Function(i)(t)**p for i in anions.split(' + ')]:
    anions_grouped += j
anions_grouped = anions_grouped
kD = (a*sym.Function('e-')(t)**b)
switch_kD = ((1-(0.003*( anions_grouped )/(0.003*( anions_grouped )+sym.Function('e-')(t))))/2)+0.5
kD_fun = (a*sym.Function('e-')(t)**b)*((1-(0.003*( anions_grouped )/(0.003*( anions_grouped )+sym.Function('e-')(t))))/2+0.5)
# kD_fun = kD*switch_kD
switch = (0.003*( anions_grouped )/(0.003*( anions_grouped )+sym.Function('e-')(t)))

reactions.append([kD_fun,{switch*sym.Function('Ar+')(t):1},{'Ar+':-1}])
reactions.append([kD_fun,{switch*sym.Function('e-')(t):1},{'e-':-1}])
for to_diffuse in [i for i in temp if '-' in i and i != 'e-']:
    # reactions.append([kD,{'(0.003*({})/(0.003*({})+e-))*{}'.format(anions,anions,to_diffuse):1},{to_diffuse:-1}])
    reactions.append([kD_fun,{switch*sym.Function(to_diffuse)(t):1},{to_diffuse:-1}])
    # reactions.append([kD_fun,{to_diffuse:1},{to_diffuse:-1}])

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
    
t = sym.symbols('t')
f_lamb = sym.lambdify((t, y) + k, ydot, "numpy")
f_jit = nb.njit(f_lamb)