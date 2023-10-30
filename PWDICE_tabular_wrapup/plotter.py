import matplotlib
# import matplotlib.pyplot as plt
import numpy as np
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_bar(title, names, means, std, pth=""):
    plt.cla()
    fig, ax = plt.subplots(figsize=(9, 6))
    lst = sum([[i - 0.125, i + 0.125] for i in range(len(names))], [])
    color_lst = sum([["C0", "C1"] for i in range(len(names))], [])
    label = sum([["determinstic", "stochastic"] for i in range(len(names))], [])
    for i in range(len(names)):
        print(i)
        if i == 0:
            ax.bar(lst[i * 2], means[i * 2], yerr=std[i * 2], width=0.25, align="center", alpha=0.5, color=color_lst[i * 2], ecolor='black', capsize=10, label=label[i * 2])
            ax.bar(lst[i * 2 + 1], means[i * 2 + 1], yerr=std[i * 2 + 1], width=0.25, align="center", alpha=0.5, color=color_lst[i * 2 + 1], ecolor='black', capsize=10, label=label[i * 2 + 1])
        else: # no label
            ax.bar(lst[i * 2], means[i * 2], yerr=std[i * 2], width=0.25, align="center", alpha=0.5, color=color_lst[i * 2], ecolor='black', capsize=10)
            ax.bar(lst[i * 2 + 1], means[i * 2 + 1], yerr=std[i * 2 + 1], width=0.25, align="center", alpha=0.5, color=color_lst[i * 2 + 1], ecolor='black', capsize=10)
            
    ax.set_ylabel(title)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=90)
    ax.yaxis.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(pth+title+".pdf")
    plt.savefig(pth+title+".jpg")

def make_bar_time(title, names, means, std, pth=""):
    plt.cla()
    fig, ax = plt.subplots()
    lst = [i for i in range(len(names))]
    color_lst = ["C0" for i in range(len(names))]
    ax.bar(lst, means, yerr=std, width=0.25, align="center", alpha=0.5, color=color_lst, ecolor='black', capsize=10)
    ax.set_ylabel(title)
    ax.set_yscale('log')
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=90)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(pth+title+".pdf")
    plt.savefig(pth+title+".jpg")
    

# 1st term is "argmax=yes", 2nd term is "argmax=no"

f = open("res/restotal.txt", "r")
g = open("res/resparam.txt", "r")
lines = f.readlines()
names, rew_means, rew_stds, suc_means, suc_stds, len_means, len_stds, tm_means, tm_stds = [], [], [], [], [], [], [], [], []
for line in lines:
    contents = line.split()
    names.append(contents[0])
    #print(contents[1:])
    rew0_mean, rew0_std, suc0_mean, suc0_std, le0_mean, le0_std, rew_mean, rew_std, suc_mean, suc_std, le_mean, le_std, tm_mean, tm_std = list(map(float, contents[1:]))
    rew_means.extend([rew0_mean, rew_mean])
    rew_stds.extend([rew0_std, rew_std])
    suc_means.extend([suc0_mean, suc_mean])
    suc_stds.extend([suc0_std, suc_std])
    len_means.extend([le0_mean, le_mean])
    len_stds.extend([le0_std, le_std])
    tm_means.append(tm_mean)
    tm_stds.append(tm_std)
print("names:", names)
pth_name = ""

lines2 = g.readlines()
for line2 in lines2:
    contents2 = line2.split()
    pth_name += "_"+contents2[1]

pth_name = pth_name[1:]
print("pth_name:", pth_name)
pth = "summary/"+pth_name+"/"
if not os.path.exists(pth):
    os.mkdir(pth)
title = "success rate"
make_bar(title, names, suc_means, suc_stds, pth)
title = "reward"
make_bar(title, names, rew_means, rew_stds, pth)
title = "average_length"
make_bar(title, names, len_means, len_stds, pth)
title = "running_time"
make_bar_time(title, names, tm_means, tm_stds, pth)
print(tm_means)
print(tm_stds)

"""
pth = "summary/5_1000_9_0.01_100_0/full/"

title = "success rate"
names = ["LP_drc", "LP_mht", "SMODICE_chi", 'SMODICE_KL', 'LObsDICE', "Lag_drc", "Lag_mht"]
means = [1, 1, 1, 1, 0.01, 0.17, 1, 1, 1, 0.99, 1, 1, 1, 1]
std = [0, 0, 0, 0, 0.3, 0.11, 0, 0, 0, 0.03, 0, 0, 0, 0]
    
make_bar(title, names, means, std, pth)

title = "reward"
names = ["LP_drc", "LP_mht","SMODICE_chi", 'SMODICE_KL', 'LObsDICE', "Lag_drc", "Lag_mht"]
means = [0.84, 0.84, 0.84, 0.84, -0.98, -0.82, 0.84, 0.77, 0.84, 0.78, 0.84, 0.84, 0.84, 0.84]
std = [0, 0.01, 0, 0, 0.05, 0.14, 0, 0.01, 0, 0.06, 0, 0, 0, 0.01]

make_bar(title, names, means, std, pth)

title = "average_length"
names = ["LP_drc", "LP_mht", "SMODICE_chi",'SMODICE_KL', 'LObsDICE', "Lag_drc", "Lag_mht"]
means = [16.60, 17.22, 16.56, 16.96, 49.00, 75.46, 16.66, 24.45, 16.69, 21.10, 16.71, 17.37, 16.65, 17.19]
std = [0.47, 0.60, 0.43, 0.40, 0.00, 14.19, 0.24, 0.94, 0.27, 3.16, 0.26, 0.41, 0.40, 0.53]

make_bar(title, names, means, std, pth)

title = "running_time"
names = ["LP_drc", "LP_mht","SMODICE_chi", 'SMODICE_KL', 'LObsDICE', "Lag_drc", "Lag_mht"]
means = [1.18, 1.19, 0.51, 0.77, 336.42, 245.17, 631.51]
std = [0.04, 0.07, 0.11, 0.01, 1.93, 6.16, 170.38]

make_bar_time(title, names, means, std, pth)

pth = "summary/5_1000_9_0.01_100_0/goal/"

title = "success rate"
names = ["LP_drc", "LP_mht", "SMODICE_chi", 'SMODICE_KL', "Lag_drc", "Lag_mht"]
means = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
std = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
make_bar(title, names, means, std, pth)

title = "reward"
names = ["LP_drc", "LP_mht","SMODICE_chi", 'SMODICE_KL', "Lag_drc", "Lag_mht"]
means = [0.85, 0.84, 0.85, 0.84, 0.85, 0.84, 0.84, 0.84, 0.85, 0.84, 0.84, 0.84]
std = [0, 0, 0, 0, 0, 0, 0, 0.01, 0, 0, 0, 0, 0, 0]

make_bar(title, names, means, std, pth)

title = "average_length"
names = ["LP_drc", "LP_mht", "SMODICE_chi",'SMODICE_KL', "Lag_drc", "Lag_mht"]
means = [16.49, 16.89, 16.49, 16.89, 16.44, 16.84, 16.66, 16.58, 16.55, 16.42, 16.76, 16.68, 16.59]
std = [0.31, 0.39, 0.31, 0.38, 0.31, 0.44, 0.20, 0.46, 0.24, 0.33, 0.46, 0.34]

make_bar(title, names, means, std, pth)

title = "running_time"
names = ["LP_drc", "LP_mht","SMODICE_chi", 'SMODICE_KL', "Lag_drc", "Lag_mht"]
means = [1.18, 1.15, 0.43, 194.91, 386.8, 912.17]
std = [0.08, 0.03, 0.13, 4.17, 165.3, 195.30]

make_bar_time(title, names, means, std, pth)


pth = "summary/5_1000_9_0.05_100_0/full/"


title = "reward"
names = ["LP_drc", "LP_mht","SMODICE_chi", 'SMODICE_KL', 'LObsDICE', "Lag_drc", "Lag_mht"]
means = [0.77, 0.79, 0.8, 0.8, -0.47, -0.67, 0.81, 0.65, 0.7, 0.48, 0.8, 0.79, 0.81, 0.79]
std = [0.09, 0.01, 0.05, 0.02, 0.49, 0.09, 0.01, 0.02, 0.16, 0.14, 0.01, 0.02, 1, 0]

make_bar(title, names, means, std, pth)

title = "success rate"
names = ["LP_drc", "LP_mht", "SMODICE_chi", 'SMODICE_KL', 'LObsDICE', "Lag_drc", "Lag_mht"]
means = [1, 1, 1, 1, 0.31, 0.13, 1, 1, 0.95, 0.92, 1, 1, 1, 1]
std = [0, 0, 0, 0, 0.28, 0.10, 0, 0, 0.09, 0.07, 0, 0, 0, 0]
    
make_bar(title, names, means, std, pth)

title = "average_length"
names = ["LP_drc", "LP_mht", "SMODICE_chi",'SMODICE_KL', 'LObsDICE', "Lag_drc", "Lag_mht"]
means = [24.09, 21.78, 21.32, 21.07, 54.74, 71.07, 19.98, 35.87, 22.26, 40.59, 21, 21.9, 20.32, 21.6]
std = [9.26, 0.11, 5.19, 1.56, 19.69, 12.68, 0.61, 1.97, 3.85, 3.43, 1, 1.51, 0.84, 2.02]

make_bar(title, names, means, std, pth)

title = "running_time"
names = ["LP_drc", "LP_mht","SMODICE_chi", 'SMODICE_KL', 'LObsDICE', "Lag_drc", "Lag_mht"]
means = [2.15, 2.18, 0.44, 110.26, 222.86, 240.85, 665.46]
std = [0.11, 0.48, 0.12, 18.28, 14.54, 26.55, 42.82]

pth = "summary/5_1000_9_0.05_100_0/goal/"


title = "reward"
names = ["LP_drc", "LP_mht","SMODICE_chi", 'SMODICE_KL', "Lag_drc", "Lag_mht"]
means = [0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.82, 0.81, 0.81, 0.81, 0.81, 0.82]
std = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0]

make_bar(title, names, means, std, pth)

title = "success rate"
names = ["LP_drc", "LP_mht", "SMODICE_chi", 'SMODICE_KL', "Lag_drc", "Lag_mht"]
means = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
std = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
make_bar(title, names, means, std, pth)

title = "average_length"
names = ["LP_drc", "LP_mht", "SMODICE_chi",'SMODICE_KL', "Lag_drc", "Lag_mht"]
means = [19.51, 19.7, 19.52, 19.71, 19.5, 19.5, 19.38, 19.53, 19.92, 20.25, 20.33, 18.53]
std = [0.92, 1.05, 0.82, 1.04, 0.78, 1.13, 0.58, 0.78, 0.95, 0.91, 1.20, 0.25]

make_bar(title, names, means, std, pth)

title = "running_time"
names = ["LP_drc", "LP_mht","SMODICE_chi", 'SMODICE_KL', 'LObsDICE', "Lag_drc", "Lag_mht"]
means = []
std = []

pth = "summary/5_500_9_0.05_100_0/full/"


title = "reward"
names = ["LP_drc", "LP_mht","SMODICE_chi", 'SMODICE_KL', 'LObsDICE', "Lag_drc", "Lag_mht"]
means = [0.8, 0.79, 0.8, 0.8, -0.51, -0.76, 0.81, 0.65, 0.60, 0.42, 0.81, 0.77, 0.81, 0.80]
std = [0.04, 0.02, 0.04, 0.02, 0.42, 0.15, 0.01, 0.02, 0.28, 0.18, 0.01, 0.02, 0.01, 0.02]

make_bar(title, names, means, std, pth)

title = "success rate"
names = ["LP_drc", "LP_mht", "SMODICE_chi", 'SMODICE_KL', 'LObsDICE', "Lag_drc", "Lag_mht"]
means = [1, 1, 1, 1, 0.35, 0.19, 1, 1, 0.90, 0.94, 1, 1, 1, 1]
std = [0, 0, 0, 0, 0.3, 0.12, 0, 0, 0.15, 0.08, 0, 0, 0, 0]
    
make_bar(title, names, means, std, pth)

title = "average_length"
names = ["LP_drc", "LP_mht", "SMODICE_chi",'SMODICE_KL', 'LObsDICE', "Lag_drc", "Lag_mht"]
means = [21.22, 21.92, 21.01, 21.41, 58.05, 71.86, 20.68, 41.14, 23.56, 50.21, 19.94, 24.32, 19.76, 21.4]
std = [4.23, 1.92, 4.25, 2.13, 13.48, 18.31, 0.44, 5.24, 2.29, 7.75, 0.75, 2.29, 1.17, 1.87]

make_bar(title, names, means, std, pth)

title = "running_time"
names = ["LP_drc", "LP_mht","SMODICE_chi", 'SMODICE_KL', 'LObsDICE', "Lag_drc", "Lag_mht"]
means = [2.14, 2.30, 0.5, 46.92, 106.42, 597.08, 458.31]
std = [0.29, 0.39, 0.11, 12.34, 11.2, 5.43, 244.67]

make_bar_time(title, names, means, std, pth)

pth = "summary/5_500_9_0.05_100_0/goal/"


title = "reward"
names = ["LP_drc", "LP_mht","SMODICE_chi", 'SMODICE_KL', "Lag_drc", "Lag_mht"]
means = [0.81, 0.81, 0.82, 0.81, 0.81, 0.82, 0.81, 0.81, 0.81, 0.81, 0.82, 0.81]
std = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0, 0.01, 0.01, 0.01, 0.01, 0]

make_bar(title, names, means, std, pth)

title = "success rate"
names = ["LP_drc", "LP_mht", "SMODICE_chi", 'SMODICE_KL', "Lag_drc", "Lag_mht"]
means = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
std = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
make_bar(title, names, means, std, pth)

title = "average_length"
names = ["LP_drc", "LP_mht", "SMODICE_chi",'SMODICE_KL', "Lag_drc", "Lag_mht"]
means = [19.55, 19.60, 19.48, 19.62, 19.5, 19.46, 20.1, 20.2, 20, 19.5, 19.23, 19.77]
std = [0.92, 1.07, 0.79, 1.08, 0.78, 1.12, 0.34, 1.26, 0.55, 0.77, 0.63, 0.33]

make_bar(title, names, means, std, pth)

title = "running_time"
names = ["LP_drc", "LP_mht","SMODICE_chi", 'SMODICE_KL', "Lag_drc", "Lag_mht"]
means = [2.15, 2.01, 0.43, 46.55, 703.14, 254.02]
std = [0.21, 0.19, 0.08, 12, 17.26, 4.94]

make_bar_time(title, names, means, std, pth)

pth = "summary/5_250_9_0.05_100_0/full/"


title = "reward"
names = ["LP_drc", "LP_mht","SMODICE_chi", 'SMODICE_KL', 'LObsDICE', "Lag_drc", "Lag_mht"]
means = [0.81, 0.79, 0.8, 0.79, -0.27, -0.64, 0.81, 0.5, 0.38, 0.39, 0.81, 0.77, 0.81, 0.78]
std = [0.01, 0.01, 0.05, 0.02, 0.6, 0.18, 0.02, 0.06, 0.28, 0.2, 0, 0, 0.01, 0.03]

make_bar(title, names, means, std, pth)

title = "success rate"
names = ["LP_drc", "LP_mht", "SMODICE_chi", 'SMODICE_KL', 'LObsDICE', "Lag_drc", "Lag_mht"]
means = [1, 1, 1, 1, 0.5, 0.28, 1, 1, 0.78, 0.9, 1, 1, 1, 1]
std = [0, 0, 0, 0, 0.39, 0.13, 0, 0, 0.17, 0.11, 0, 0, 0, 0]
    
make_bar(title, names, means, std, pth)

title = "average_length"
names = ["LP_drc", "LP_mht", "SMODICE_chi",'SMODICE_KL', 'LObsDICE', "Lag_drc", "Lag_mht"]
means = [20.13, 22.03, 21.32, 21.54, 55.21, 75.3, 20.3, 50.78, 23.43, 46.38, 20.07, 24.2, 20.1, 22.8]
std = [1.06, 1.36, 5.17, 2.02, 17.42, 7.98, 1.7, 5.63, 4.46, 8.39, 1.31, 0.94, 0.82, 2.55]

make_bar(title, names, means, std, pth)

title = "running_time"
names = ["LP_drc", "LP_mht","SMODICE_chi", 'SMODICE_KL', 'LObsDICE', "Lag_drc", "Lag_mht"]
means = [2.48, 2.32, 0.43, 15.09, 51.74, 287.45, 284.7]
std = [0.60, 0.20, 0.08, 1.66, 9.29, 6.01, 6.49]

make_bar_time(title, names, means, std, pth)

pth = "summary/5_250_9_0.05_100_0/goal/"


title = "reward"
names = ["LP_drc", "LP_mht","SMODICE_chi", 'SMODICE_KL', "Lag_drc", "Lag_mht"]
means = [0.81, 0.81, 0.81, 0.81, 0.81, 0.82, 0.81, 0.81, 0.81, 0.82, 0.81, 0.81]
std = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0, 0]

make_bar(title, names, means, std, pth)

title = "success rate"
names = ["LP_drc", "LP_mht", "SMODICE_chi", 'SMODICE_KL', "Lag_drc", "Lag_mht"]
means = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
std = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
make_bar(title, names, means, std, pth)

title = "average_length"
names = ["LP_drc", "LP_mht", "SMODICE_chi",'SMODICE_KL', "Lag_drc", "Lag_mht"]
means = [19.61, 19.73, 19.54, 19.91, 19.5, 19.47, 19.68, 19.66, 19.7, 19.23, 19.97, 20.37]
std = [0.87, 1.12, 0.8, 0.85, 0.78, 1.12, 1.38, 0.65, 0.65, 0.62, 0.05, 0.21]

make_bar(title, names, means, std, pth)

title = "running_time"
names = ["LP_drc", "LP_mht","SMODICE_chi", 'SMODICE_KL', "Lag_drc", "Lag_mht"]
means = [1.96, 2.09, 19.47, 15.56, 283.96, 621.31]
std = [0.23, 0.2, 1.12, 1.65, 5.8, 40.3]

make_bar_time(title, names, means, std, pth)


pth = "summary/5_250_9_0.05_40_0/full/"


title = "reward"
names = ["LP_drc", "LP_mht","SMODICE_chi", 'SMODICE_KL', 'LObsDICE', "Lag_drc", "Lag_mht"]
means = [0.76, 0.76, 0.74, 0.78, 0.28, -0.22, 0.8, 0.5, 0.57, 0.21, 0.53, 0.76, 0.63, 0.77]
std = [0.16, 0.05, 0.12, 0.03, 0.46, 0.21, 0.01, 0.1, 0.33, 0.1, 0.4, 0.02, 0.25, 0.01]

make_bar(title, names, means, std, pth)

title = "success rate"
names = ["LP_drc", "LP_mht", "SMODICE_chi", 'SMODICE_KL', 'LObsDICE', "Lag_drc", "Lag_mht"]
means = [0.96, 0.98, 0.95, 0.99, 0.58, 0.16, 1, 0.8, 0.82, 0.54, 0.77, 1, 0.87, 1]
std = [0.12, 0.04, 0.12, 0.03, 0.21, 0.19, 0, 0.09, 0.26, 0.08, 0.33, 0, 0.19, 0]
    
make_bar(title, names, means, std, pth)

title = "average_length"
names = ["LP_drc", "LP_mht", "SMODICE_chi",'SMODICE_KL', 'LObsDICE', "Lag_drc", "Lag_mht"]
means = [20.64, 22.86, 21.72, 22.00, 27.36, 30.79, 20.78, 29.12, 24.24, 28.62, 20.86, 24.67, 22.57, 23.63]
std = [2.00, 2.61, 3.62, 2.46, 6.75, 4.19, 1.12, 1.43, 4.86, 3.43, 1.28, 2.32, 3.88, 1.44]

make_bar(title, names, means, std, pth)

title = "running_time"
names = ["LP_drc", "LP_mht","SMODICE_chi", 'SMODICE_KL', 'LObsDICE', "Lag_drc", "Lag_mht"]
means = [2.61, 2.27, 7.65, 107.54, 221.9, 312.89, 306.38]
std = [0.7, 0.3, 2.37, 24.57, 29.13, 16.34, 17.18]

make_bar_time(title, names, means, std, pth)


pth = "summary/5_250_9_0.05_40_0/goal/"


title = "reward"
names = ["LP_drc", "LP_mht","SMODICE_chi", 'SMODICE_KL', "Lag_drc", "Lag_mht"]
means = [0.64, 0.69, 0.62, 0.69, 0.74, 0.77, 0.8, 0.5, 0.81, 0.75, 0.77, 0.8]
std = [0.34, 0.21, 0.32, 0.2, 0.14, 0.13, 0.01, 0.1, 0.01, 0.05, 0.05, 0.01]

make_bar(title, names, means, std, pth)

title = "success rate"
names = ["LP_drc", "LP_mht", "SMODICE_chi", 'SMODICE_KL', "Lag_drc", "Lag_mht"]
means = [0.86, 0.92, 0.85, 0.92, 0.95, 0.97, 1, 0.8, 1, 0.97, 0.97, 1]
std = [0.27, 0.15, 0.26, 0.15, 0.1, 0.09, 0, 0.09, 0, 0.05, 0.05, 1]
    
make_bar(title, names, means, std, pth)

title = "average_length"
names = ["LP_drc", "LP_mht", "SMODICE_chi",'SMODICE_KL', "Lag_drc", "Lag_mht"]
means = [21.19, 22.91, 21.66, 23.24, 21.46, 21.04, 20.78, 29.12, 20.2, 21.86, 20.23, 20.57]
std = [2.9, 4.45, 3.1, 4.13, 3.21, 2.84, 1.12, 1.43, 0.82, 0.85, 0.24, 1.03]

make_bar(title, names, means, std, pth)

title = "running_time"
names = ["LP_drc", "LP_mht","SMODICE_chi", 'SMODICE_KL', "Lag_drc", "Lag_mht"]
means = [2.13, 2.14, 0.29, 107.54, 301.46, 626.05]
std = [0.23, 0.09, 0.13, 24.57, 16.89, 20.73]

make_bar_time(title, names, means, std, pth)

pth = "summary/5_250_9_0.05_30_0/full/"


title = "reward"
names = ["LP_drc", "LP_mht","SMODICE_chi", 'SMODICE_KL', 'LObsDICE', "Lag_drc", "Lag_mht"]
means = [0.53, 0.5, 0.56, 0.6, 0.67, -0.15, 0.8, 0.09, 0.33, -0.05, 0.5, 0.3, 0.51, 0.39]
std = [0.39, 0.33, 0.33, 0.26, 0.25, 0.2, 0.44, 0.01, 0.04, 0.11, 0.28, 0.22, 0.35, 0.21]

make_bar(title, names, means, std, pth)

title = "success rate"
names = ["LP_drc", "LP_mht", "SMODICE_chi", 'SMODICE_KL', 'LObsDICE', "Lag_drc", "Lag_mht"]
means = [0.75, 0.74, 0.78, 0.82, 0.88, 0.14, 1, 0.38, 0.58, 0.24, 0.73, 0.57, 0.73, 0.63]
std = [0.35, 0.3, 0.29, 0.23, 0.22, 0.19, 0, 0.04, 0.4, 0.1, 0.25, 0.21, 0.31, 0.19]
    
make_bar(title, names, means, std, pth)

title = "average_length"
names = ["LP_drc", "LP_mht", "SMODICE_chi",'SMODICE_KL', 'LObsDICE', "Lag_drc", "Lag_mht"]
means = [21.47, 22.47, 21.64, 21.27, 21.57, 24.77, 21.05, 27.12, 23.61, 25.22, 22.39, 24.72, 21.48, 22.24]
std = [2.53, 0.47, 2.99, 1.78, 2.82, 0.31, 27.12, 1.01, 3.11, 1.67, 2.56, 6.01, 2.51, 3.02]

make_bar(title, names, means, std, pth)

title = "running_time"
names = ["LP_drc", "LP_mht","SMODICE_chi", 'SMODICE_KL', 'LObsDICE', "Lag_drc", "Lag_mht"]
means = [2.52, 2.47, 7.6, 417.72, 344.62, 180.14, 166.65]
std = [0.47, 0.53, 0.31, 3.92, 12.22, 6.01, 4.23]

make_bar_time(title, names, means, std, pth)


pth = "summary/5_250_9_0.05_30_0/goal/"


title = "reward"
names = ["LP_drc", "LP_mht","SMODICE_chi", 'SMODICE_KL', "Lag_drc", "Lag_mht"]
means = [0.6, 0.53, 0.61, 0.55, 0.79, 0.79, 0.78, 0.73, 0.78, 0.73, 0.77, 0.70]
std = [0.3, 0.32, 0.24, 0.28, 0.07, 0.04, 0.95, 0.05, 0.05, 0.05, 0.05, 0.05]

make_bar(title, names, means, std, pth)

title = "success rate"
names = ["LP_drc", "LP_mht", "SMODICE_chi", 'SMODICE_KL', "Lag_drc", "Lag_mht"]
means = [0.81, 0.76, 0.82, 0.77, 0.98, 0.06, 0.97, 0.93, 0.97, 0.93, 0.97, 0.9]
std = [0.27, 0.29, 0.2, 0.25, 0.98, 0.04, 0.05, 0.05, 0.05, 0.05, 0.05, 0.08]
    
make_bar(title, names, means, std, pth)

title = "average_length"
names = ["LP_drc", "LP_mht", "SMODICE_chi",'SMODICE_KL', "Lag_drc", "Lag_mht"]
means = [20.83, 21.98, 20.4, 21.48, 19.71, 19.78, 19.63, 20.63, 19.74, 20.10, 20.23, 20.24]
std = [2.51, 2.06, 1.83, 1.34, 1.01, 1.72, 0.68, 0.19, 1.04, 0.41, 0.4, 0.54]

make_bar(title, names, means, std, pth)

title = "running_time"
names = ["LP_drc", "LP_mht","SMODICE_chi", 'SMODICE_KL', "Lag_drc", "Lag_mht"]
means = [2.27, 2.29, 4.85, 413.05, 163.96, 165.31]
std = [0.37, 0.29, 0.54, 3.71, 1.53, 2.35]

make_bar_time(title, names, means, std, pth)
"""
