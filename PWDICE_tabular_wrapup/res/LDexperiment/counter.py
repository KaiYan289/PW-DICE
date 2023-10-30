import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
TS_dataset = 10000
TA_dataset = 10000
noise = 1
optimality = 0


def make_bar(datas, name, output_flag=False):
    plt.cla()
    fig, ax = plt.subplots(figsize=(15,5))
    """
    mean1, std1 = data1.mean(), data1.std()
    mean2, std2 = data2.mean(), data2.std()
    mean3, std3 = data3.mean(), data3.std()
    mean4, std4 = data4.mean(), data4.std()
    mean5, std5 = data5.mean(), data5.std()
    """
    name_lst = ["ours-Entreg", "ours-LP", "ours-KLreg", "ours_newKL", "SMODICE-CHI", "SMODICE-KL", "LOBSDICE"]
    means, stds, mx, agmx = [], [], [], []
    for data in datas:
        means.append(data.mean())
        stds.append(data.std())
        mx.append(data.max())
        agmx.append(data.argmax())
    if output_flag:
        
        print("name:", name)
        for i in range(len(name_lst)):
            print(name_lst[i], means[i], stds[i], mx[i], agmx[i])
        """
        print("ours-Entreg:", mean1, data1.min(), data1.argmin(), data1.max(), data1.argmax())
        print("ours-LP:", mean2, data2.min(), data2.argmin(), data2.max(), data2.argmax())
        print("SMODICE-CHI:", mean3, data3.min(), data3.argmin(), data3.max(), data3.argmax())
        print("SMODICE-KL:", mean4, data4.min(), data4.argmin(), data4.max(), data4.argmax())
        print("LOBSDICE:", mean5, data5.min(), data5.argmin(), data5.max(), data5.argmax())
        """
    lst = np.arange(len(name_lst))
    for i in range(len(name_lst)):
        ax.bar(lst[i], means[i], yerr=stds[i], width=0.25, ecolor='black', capsize=10)
    
    ax.set_xticks(lst)
    ax.set_xticklabels(name_lst, rotation=90)
    if name.find("regret") == -1 or name.find("hard") != -1: plt.yscale('log')
    plt.title(name)
    plt.tight_layout()
    plt.savefig("figs_new/"+str(TS_dataset)+"_"+str(TA_dataset)+"_"+str(noise)+"_"+str(optimality)+"_"+name+".jpg")

def process(name):
    arr = []
    f = open(name+".txt", "r")
    lines = f.readlines()
    for line in lines:
        contents = line.split()
        if float(contents[2]) == TS_dataset and float(contents[3]) == TA_dataset and float(contents[4]) == noise and float(contents[5]) == optimality:
            arr.append(np.array(list(map(float, contents[-48:]))))
        # print(np.array(list(map(float, contents[-16:]))))
    f.close()
    return np.array(arr)

#arr1 = process("ours_Entreg")
arr2 = process("ours_LP")
# arr3 = process("ours_KLreg")
arr4 = process("ours_newKL")
arr5 = process("SMODICE_CHI")
arr6 = process("SMODICE_KL")
arr7 = process("LobsDICE")
print(arr1.shape, arr2.shape, arr3.shape, arr4.shape, arr5.shape, arr6.shape, arr7.shape)
datas = [arr1, arr2, arr3, arr4, arr5, arr6, arr7]
for o, Z in enumerate(["TVss", "TVs", "regret"]):
    for i, solver in enumerate(["approx", "exact"]):
            for j, TS in enumerate(["soft", "hard"]):
                for k, GT_rho_E in enumerate([False, True]):
                    for l, GT_rho_I in enumerate([False, True]):
                        x = i * 8 + j * 4 + k * 2 + l
                        s = Z + " "
                        s += solver + " MDP + "
                        s += TS + " expert + "
                        s += ("estimated" if k == 0 else "GT") + " d^E +"
                        s += ("estimated" if l == 0 else "GT") + " d^I"
                        output_flag = (k == 0) and (l == 0) and (o == 2) and (i == 0)
                        make_bar([arr[:, x * 3 + o] for arr in datas], s, output_flag)
"""
name = ["all_estimate", "rho_E GT", "rho_I_GT",  "rho_E rho_I GT", "dynamic GT", "dynamic rho_E GT", "dynamic rho_I GT", "all GT"]
for i in range(8):
    make_bar(arr1[:, i], arr2[:, i], arr3[:, i], arr4[:, i], name[i])
"""
