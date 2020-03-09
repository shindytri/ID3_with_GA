#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
import random as r
import pandas as pd
import numpy as np
import csv


# In[2]:


ukPop  = 100                # ukuran populasi
ukKrom = 15                # ukuran minimal kromosom
pc     = 0.7               # probabilitas crossover
pm     = 0.43              # probabilitas mutasi
maxGen = 550               # maximal generasi

# klasifikasi
# suhu           : rendah = 0, normal = 1, tinggi = 2
# waktu          : pagi = 0, siang = 1, sore = 2, malam = 3
# kondisi langit : cerah = 0, berawan = 1, rintik = 2, hujan = 3
# kelembapan     : rendah = 0, normal = 1, tinggi = 2
# terbang/tidak  : ya = 1, tidak = 0


# In[3]:


data_latih = pd.read_csv("data_latih.csv", names=['suhu','waktu','kondisi langit','kelembapan','terbang/tidak'])
data_uji = pd.read_csv("data_uji.csv", names=['suhu','waktu','kondisi langit','kelembapan'])

# ENCODING DATA LATIH DAN DATA UJI DENGAN BINARY ENCODING
arr_train=[[0 for x in range(ukKrom)] for x in range(80)]
for i in range(80):
    if (data_latih['suhu'][i] == 'rendah'):
        arr_train[i][0] = 1
        arr_train[i][1] = 0
        arr_train[i][2] = 0
    elif (data_latih['suhu'][i] == 'normal'):
        arr_train[i][0] = 0       
        arr_train[i][1] = 1
        arr_train[i][2] = 0
    elif (data_latih['suhu'][i] == 'tinggi'):
        arr_train[i][0] = 0
        arr_train[i][1] = 0
        arr_train[i][2] = 1
    
    if (data_latih['waktu'][i] == 'pagi'):
        arr_train[i][3] = 1
        arr_train[i][4] = 0
        arr_train[i][5] = 0
        arr_train[i][6] = 0
    elif (data_latih['waktu'][i] == 'siang'):
        arr_train[i][3] = 0
        arr_train[i][4] = 1
        arr_train[i][5] = 0
        arr_train[i][6] = 0
    elif (data_latih['waktu'][i] == 'sore'):
        arr_train[i][3] = 0
        arr_train[i][4] = 0
        arr_train[i][5] = 1
        arr_train[i][6] = 0
    elif (data_latih['waktu'][i] == 'malam'):
        arr_train[i][3] = 0
        arr_train[i][4] = 0
        arr_train[i][5] = 0
        arr_train[i][6] = 1

    if (data_latih['kondisi langit'][i] == 'cerah'):
        arr_train[i][7] = 1
        arr_train[i][8] = 0
        arr_train[i][9] = 0
        arr_train[i][10] = 0
    elif (data_latih['kondisi langit'][i] == 'berawan'):
        arr_train[i][7] = 0
        arr_train[i][8] = 1
        arr_train[i][9] = 0
        arr_train[i][10] = 0
    elif (data_latih['kondisi langit'][i] == 'rintik'):
        arr_train[i][7] = 0
        arr_train[i][8] = 0
        arr_train[i][9] = 1
        arr_train[i][10] = 0
    elif (data_latih['kondisi langit'][i] == 'hujan'):
        arr_train[i][7] = 0
        arr_train[i][8] = 0
        arr_train[i][9] = 0
        arr_train[i][10] = 1
    
    if (data_latih['kelembapan'][i] == 'rendah'):
        arr_train[i][11] = 1
        arr_train[i][12] = 0
        arr_train[i][13] = 0
    elif (data_latih['kelembapan'][i] == 'normal'):
        arr_train[i][11] = 0
        arr_train[i][12] = 1
        arr_train[i][13] = 0
    elif (data_latih['kelembapan'][i] == 'tinggi'):
        arr_train[i][11] = 0
        arr_train[i][12] = 0
        arr_train[i][13] = 1
    
    if (data_latih['terbang/tidak'][i] == 'ya'):
        arr_train[i][14] = 1
    elif (data_latih['terbang/tidak'][i] == 'tidak'):
        arr_train[i][14] = 0

arr_test = [[0 for x in range(14)] for x in range(20)]
for i in range(20):
    if (data_uji['suhu'][i] == 'Rendah'):
        arr_test[i][0] = 1
        arr_test[i][1] = 0
        arr_test[i][2] = 0
    elif (data_uji['suhu'][i] == 'Normal'):
        arr_test[i][0] = 0
        arr_test[i][1] = 1
        arr_test[i][2] = 0
    elif (data_uji['suhu'][i] == 'Tinggi'):
        arr_test[i][0] = 0
        arr_test[i][1] = 0
        arr_test[i][2] = 1
    
    if(data_uji['waktu'][i] == 'Pagi'):
        arr_test[i][3] = 1
        arr_test[i][4] = 0
        arr_test[i][5] = 0
        arr_test[i][6] = 0
    elif(data_uji['waktu'][i] == 'Siang'):
        arr_test[i][3] = 0
        arr_test[i][4] = 1
        arr_test[i][5] = 0
        arr_test[i][6] = 0
    elif(data_uji['waktu'][i] == 'Sore'):
        arr_test[i][3] = 0
        arr_test[i][4] = 0
        arr_test[i][5] = 1
        arr_test[i][6] = 0
    elif(data_uji['waktu'][i] == 'Malam'):
        arr_test[i][3] = 0
        arr_test[i][4] = 0
        arr_test[i][5] = 0
        arr_test[i][6] = 1
        
    if(data_uji['kondisi langit'][i] == 'Cerah'):
        arr_test[i][7] = 1
        arr_test[i][8] = 0
        arr_test[i][9] = 0
        arr_test[i][10] = 0
    elif(data_uji['kondisi langit'][i] == 'Berawan'):
        arr_test[i][7] = 0
        arr_test[i][8] = 1
        arr_test[i][9] = 0
        arr_test[i][10] = 0
    elif(data_uji['kondisi langit'][i] == 'Rintik'):
        arr_test[i][7] = 0
        arr_test[i][8] = 0
        arr_test[i][9] = 1
        arr_test[i][10] = 0
    elif(data_uji['kondisi langit'][i] == 'Hujan'):
        arr_test[i][7] = 0
        arr_test[i][8] = 0
        arr_test[i][9] = 0
        arr_test[i][10] = 1
    
    if(data_uji['kelembapan'][i] == 'Rendah'):
        arr_test[i][11] = 1
        arr_test[i][12] = 0
        arr_test[i][13] = 0
    elif(data_uji['kelembapan'][i] == 'Normal'):
        arr_test[i][11] = 0
        arr_test[i][12] = 1
        arr_test[i][13] = 0
    elif(data_uji['kelembapan'][i] == 'Tinggi'):
        arr_test[i][11] = 0
        arr_test[i][12] = 0
        arr_test[i][13] = 1


# In[4]:


def check(k,data):
    suhu       = (k[0] == 1 and k[0] == data[0]) or (k[1] == 1 and k[1] == data[1]) or (k[2] == 1 and k[2] == data[2])
    waktu      = (k[3] == 1 and k[3] == data[3]) or (k[4] == 1 and k[4] == data[4]) or (k[5] == 1 and k[5] == data[5]) or (k[6] == 1 and k[6] == data[6])
    cuaca      = (k[7] == 1 and k[7] == data[7]) or (k[8] == 1 and k[8] == data[8]) or (k[9] == 1 and k[9] == data[9]) or (k[10] == 1 and k[10] == data[10])
    kelembapan = (k[11] == 1 and k[11] == data[11]) or (k[12] == 1 and k[12] == data[12]) or (k[13] == 1 and k[13] == data[13]) 
    if (suhu and waktu and cuaca and kelembapan):
        return k[14] == data[14]
    else:
        return k[14] != data[14]

def divList(pop, n):
    for i in range(0, len(pop), n):
        yield pop[i:i+n]

def fitness(ind):
    num = int(len(ind)/ukKrom)
    rules = []
    count = 0

    if (num > 1) :
        for i in range(num):
            rules.append(list(divList(ind,ukKrom))[i])
    else : 
        rules.append(ind)
    
    for y in range(80):
        x = 0
        while(x < num-1 and check(rules[x],arr_train[y])!= 0):
            x+=1
        if(check(rules[x],arr_train[y])):
            count+=1

    return count/80

def fitnessAll(pop,fit):
    for x in range(ukPop):
        fit[x] = fitness(pop[x])

def rouletteWheel(fit):
    sum = 0
    for x in range(len(fit)):
        sum += fit[x]
    idx = 0
    prob = r.uniform(0,1)
    while (prob > 0):
        prob -= fit[idx]/sum
        idx += 1
    return idx-1

def crossover(par1,par2):
    x = r.uniform(0,1)
    if (x > pc):
        cek = len(par1) - len(par2)
        if(cek > 0):
            temp_par = par1
            par1 = par2
            par2 = temp_par

        x = r.randint(1,len(par1)) # titik potong 1 parent 1
        y = r.randint(1,len(par1)) # titik potong 2 parent 1

        while x == y:
            y = r.randint(1,len(par1))

        if (x > y):
            temp = x
            x = y
            y = temp

        p1 = y - x
        gap = p1 % ukKrom

        p2 = []
        p2.append([x,x+p1])
        p2.append([x,x+gap])
        p2.append([y-p1,y])
        p2.append([y-gap,y])
        
        dot = r.randint(0,3)
        x2 = p2[dot][0] # titik potong 1 parent 2
        y2 = p2[dot][1] # titik potong 2 parent 2

        temp1 = []
        temp2 = []

        temp1.extend(par1[:x])
        temp1.extend(par2[x2:y2])
        temp1.extend(par1[y:])

        temp2.extend(par2[:x2])
        temp2.extend(par1[x:y])
        temp2.extend(par2[y2:])
        
        if(cek>0):
            return temp2,temp1
        else:
            return temp1,temp2
    else:
        return par1, par2
    
def mutasi(ind):
    x = r.uniform(0,1)
    if(x > pm):
        swap = [r.randint(0,1) for x in range(len(ind))]
        for i in range(len(swap)):
            if swap[i] == 1:
                if ind[i] == 0:
                    ind[i] = 1
                else:
                    ind[i] = 0

def elitisme(fit):
    for i in range(ukPop):
        if (fit[i] == max(fit)):
            return i      

def generationalReplacement(par, elit):
    par[0] = elit
    par[1] = elit
    return par

def cekDataTest(k,data):
    suhu       = (k[0] == 1 and k[0] == data[0]) or (k[1] == 1 and k[1] == data[1]) or (k[2] == 1 and k[2] == data[2])
    waktu      = (k[3] == 1 and k[3] == data[3]) or (k[4] == 1 and k[4] == data[4]) or (k[5] == 1 and k[5] == data[5]) or (k[6] == 1 and k[6] == data[6])
    cuaca      = (k[7] == 1 and k[7] == data[7]) or (k[8] == 1 and k[8] == data[8]) or (k[9] == 1 and k[9] == data[9]) or (k[10] == 1 and k[10] == data[10])
    kelembapan = (k[11] == 1 and k[11] == data[11]) or (k[12] == 1 and k[12] == data[12]) or (k[13] == 1 and k[13] == data[13]) 
    if (suhu and waktu and cuaca and kelembapan):
        return data[14]
    else:
        if (data[14] == 1):
            return 0
        else:
            return 1

def getResult(rules,ind):
    t = 0
    f = 0
    for i in range(len(rules)):
        if cekDataTest(ind,rules[i]):
            t+=1
        else:
            f+=1 
    return t > f

def dekodeKromosom(ind,dec):
    if(ind[0] != 1 or ind[1] != 1 or ind[2] != 1):
        if (ind[0] == 1 and ind[1] == 1 and ind[2] == 0):
            dec.append("(Suhu : Rendah or Suhu : Normal)")
        elif (ind[0] == 1 and ind[1] == 0 and ind[2] == 1):
            dec.append("(Suhu : Rendah or Suhu : Tinggi)")
        elif (ind[1] == 1 and ind[2] == 1):
            dec.append("(Suhu : Normal or Suhu : Tinggi)")
        elif (ind[0] == 1):
            dec.append("(Suhu : Rendah)")
        elif (ind[1] == 1):
            dec.append("(Suhu : Normal)")
        elif (ind[2] == 1):
            dec.append("(Suhu : Tinggi)")
    
    if(ind[3] != 1 or ind[4] != 1 or ind[5] != 1 or ind[6] != 1):
        if (ind[3] == 1 and ind[4] == 0 and ind[5] == 1 and ind[6] == 1):
            dec.append("(Waktu : Pagi or Waktu : Sore or Waktu : Malam)")
        elif (ind[3] == 1 and ind[4] == 1 and ind[5] == 0 and ind[6] == 1):
            dec.append("(Waktu : Pagi or Waktu : Siang or Waktu : Malam)")
        elif (ind[3] == 1 and ind[4] == 1 and ind[5] == 1 and ind[6] == 0):
            dec.append("(Waktu : Pagi or Waktu : Siang or Waktu : Sore)")
        elif (ind[3] == 0 and ind[4] == 1 and ind[5] == 1 and ind[6] == 1):
            dec.append("(Waktu : Siang or Waktu : Sore or Waktu : Malam)")
        elif(ind[4] == 1 and ind[6] == 1):
            dec.append("(Waktu : Siang or Waktu : Malam)")
        elif(ind[4] == 1 and ind[5] == 1):
            dec.append("(Waktu : Siang or Waktu : Sore)")
        elif(ind[3] == 1 and ind[4] == 1):
            dec.append("(Waktu : Pagi or Waktu : Siang)")
        elif(ind[3] == 1 and ind[6] == 1):
            dec.append("(Waktu : Pagi or Waktu : Malam)")
        elif(ind[3] == 1 and ind[5] == 1):
            dec.append("(Waktu : Pagi or Waktu : Sore)")
        elif(ind[5] == 1 and ind[6] == 1):
            dec.append("(Waktu : Sore or Waktu : Malam)")
        elif(ind[3] == 1):
            dec.append("(Waktu : Pagi)")
        elif(ind[4] == 1):
            dec.append("(Waktu : Siang)")
        elif(ind[5] == 1):
            dec.append("(Waktu : Sore)")
        elif(ind[6] == 1):
            dec.append("(Waktu : Malam)")
    
    if(ind[7] != 1 or ind[8] != 1 or ind[9] != 1 or ind[10] != 1):
        if (ind[7] == 1 and ind[8] == 0 and ind[9] == 1 and ind[10] == 1):
            dec.append("(Langit : Cerah or Langit : Rintik or Langit : Hujan)")
        elif (ind[7] == 1 and ind[8] == 1 and ind[9] == 0 and ind[10] == 1):
            dec.append("(Langit : Cerah or Langit : Berawan or Langit : Hujan)")
        elif (ind[7] == 1 and ind[8] == 1 and ind[9] == 1 and ind[10] == 0):
            dec.append("(Langit : Cerah or Langit : Berawan or Langit : Rintik)")
        elif (ind[7] == 0 and ind[8] == 1 and ind[9] == 1 and ind[10] == 1):
            dec.append("(Langit : Berawan or Langit : Rintik or Langit : Hujan)")
        elif(ind[8] == 1 and ind[10] == 1):
            dec.append("(Langit : Berawan or Langit : Hujan)")
        elif(ind[8] == 1 and ind[9] == 1):
            dec.append("(Langit : Berawan or Langit : Rintik)")
        elif(ind[7] == 1 and ind[8] == 1):
            dec.append("(Langit : Cerah or Langit : Berawan)")
        elif(ind[7] == 1 and ind[10] == 1):
            dec.append("(Langit : Cerah or Langit : Hujan)")
        elif(ind[7] == 1 and ind[9] == 1):
            dec.append("(Langit : Cerah or Langit : Rintik)")
        elif(ind[9] == 1 and ind[10] == 1):
            dec.append("(Langit : Rintik or Langit : Hujan)")
        elif(ind[7] == 1):
            dec.append("(Langit : Cerah)")
        elif(ind[8] == 1):
            dec.append("(Langit : Berawan)")
        elif(ind[9] == 1):
            dec.append("(Langit : Rintik)")
        elif(ind[10] == 1):
            dec.append("(Langit : Hujan)")

    if(ind[11] != 1 or ind[12] != 1 or ind[13] != 1):
        if (ind[11] == 1 and ind[12] == 1 and ind[13] == 0):
            dec.append("(Kelembapan : Rendah or Kelembapan : Normal)")
        elif (ind[11] == 1 and ind[12] == 0 and ind[13] == 1):
            dec.append("(Kelembapan : Rendah or Kelembapan : Tinggi)")
        elif (ind[12] == 1 and ind[13] == 1):
            dec.append("(Kelembapan : Normal or kelembapan : Tinggi)")
        elif (ind[11] == 1):
            dec.append("(Kelembapan : Rendah")
        elif (ind[12] == 1):
            dec.append("(Kelembapan : Normal)")
        elif (ind[13] == 1):
            dec.append("(Kelembapan : Tinggi)")

    if (ind[14] == 1):
        dec.append("Terbang = Ya")
    elif (ind[14] == 0):
        dec.append("Terbang = Tidak")


# In[5]:


# Inisiasi Populasi   
par_arr = [[] for x in range(ukPop)]
populasi = [[] for x in range(ukPop)]
fit_arr = [0 for x in range(ukPop)]
for x in range(ukPop):
    for y in range(r.randint(1,9)*ukKrom):
        populasi[x].append(r.randint(0,1))
    
#Pencarian Fitness Terbaik
for x in range(maxGen):
    temp = copy.deepcopy(populasi)
    
    # Hitung Fitness
    fitnessAll(populasi,fit_arr)
    
    # Parent Selection
    i = 0
    while i < ukPop:
        par1 = populasi[rouletteWheel(fit_arr)]
        par2 = populasi[rouletteWheel(fit_arr)]
        while par1 == par2:
            par2 = populasi[rouletteWheel(fit_arr)]
        # Crossover
        par1, par2 = crossover(par1,par2)
        # Mutation
        mutasi(par1)
        mutasi(par2)
        par_arr[i] = par1
        if(i != ukPop-1):
            par_arr[i+1] = par2
        i+=2
    
    # General Replacement
    indElit = temp[elitisme(fit_arr)]
    populasi = generationalReplacement(par_arr,indElit)
    
# Hasil Akhir
fitnessAll(populasi,fit_arr)
idx = elitisme(fit_arr)
indElit = populasi[idx]
print("Kromosom Terbaik : ",indElit)
print("Banyak Gen       : ",len(indElit)," gen")
print("Banyak Rule      : ",int(len(indElit)/ukKrom), " rule")
print("Fitness          : ",fit_arr[idx])
print()

# Decoding Best Chromosom and Saving to CSV File
num = int(len(indElit)/ukKrom)
rules = []
if (num > 1):
    for i in range(num):
        rules.append(list(divList(indElit,ukKrom))[i])
else:
    rules.append(indElit)
    
decode = [[] for x in range(num)]
for i in range(num):
    dekodeKromosom(rules[i],decode[i])
    print('RULE ',i+1)
    for x in range(len(decode[i])):
        if (x == len(decode[i])-1):
            print('then ',decode[i][x])
        elif (x == 0):
            print('if ',decode[i][x])
        else :
            print('and ',decode[i][x])
    

# DATA TESTING
cek = [[] for x in range(len(arr_test))]
for i in range(len(arr_test)):
    if getResult(rules, arr_test[i]):
        cek[i] = "Ya"
    else :
        cek[i] = "Tidak"
    
with open("target_latih.csv","w") as f:
    wr = csv.writer(f,delimiter="\n")
    wr.writerow(cek)

