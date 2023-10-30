import numpy as np
import matplotlib
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
TS_dataset = 10000
TA_dataset = 10000
noise = 1
optimality = 0


colors = ['red', 'orange', 'purple', 'blue', 'green']
name_lst = ["Ours (LP)", "Ours (Reg)", "SMODICE-CHI", "SMODICE-KL", "LobsDICE"]

mean_buff, std_buff = {}, {}

LEN_NOISE_LEVEL = 3
LEN_EXPERT_TRAJ = 4

def clear(name):
    global mean_buff, std_buff
    
    mean_buff[name] = [[[np.array([]) for _ in range(len(name_lst))] for __ in range(LEN_EXPERT_TRAJ)] for ___ in range(LEN_NOISE_LEVEL)]
    std_buff[name]  = [[[np.array([]) for _ in range(len(name_lst))] for __ in range(LEN_EXPERT_TRAJ)] for ___ in range(LEN_NOISE_LEVEL)]

def add_point(datas, name, noise_level_id, expert_traj_id, output_flag=False):
    global mean_buff, std_buff
    
    for i, data in enumerate(datas):
        # print(mean_buff[name][i], np.array([data.mean()]))
        mean_buff[name][noise_level_id][expert_traj_id][i] = np.concatenate([mean_buff[name][noise_level_id][expert_traj_id][i], np.array([data.mean()])])
        std_buff[name][noise_level_id][expert_traj_id][i] = np.concatenate([std_buff[name][noise_level_id][expert_traj_id][i], np.array([data.std()])])

    
    """
    means, stds, mx, agmx = [], [], [], []
    for data in datas:
        means.append(data.mean())
        stds.append(data.std())
        mx.append(data.max())
        agmx.append(data.argmax())
    """ 
    if output_flag:
        pass
        """
        for i in range(len(name_lst)):
            print(name_lst[i], means[i], stds[i], mx[i], agmx[i])
        
        print("ours-Entreg:", mean1, data1.min(), data1.argmin(), data1.max(), data1.argmax())
        print("ours-LP:", mean2, data2.min(), data2.argmin(), data2.max(), data2.argmax())
        print("SMODICE-CHI:", mean3, data3.min(), data3.argmin(), data3.max(), data3.argmax())
        print("SMODICE-KL:", mean4, data4.min(), data4.argmin(), data4.max(), data4.argmax())
        print("LOBSDICE:", mean5, data5.min(), data5.argmin(), data5.max(), data5.argmax())
        """

def process(TA_traj, expert_traj, noise_level, name):
    arr = []
    f = open(name+"/"+name+"-"+str(TA_traj)+"-"+str(expert_traj)+"-"+("1.0" if noise_level == 1 else str(noise_level))+".txt", "r")
    lines = f.readlines()
    for line in lines:
        contents = line.split()
        arr.append(np.array(list(map(float, contents))))
        assert len(contents) == 48, "Error!"
    f.close()
    return np.array(arr)

formal_name = ["State Pair Occupancy Divergence", "State Occupancy Divergence", "Regret"]

for o, Z in enumerate(["TVss", "TVs", "regret"]):
    for oo, noise_level in enumerate([0.01, 0.1, 1]): 
        for ooo, expert_traj in enumerate([10, 100, 1000, 10000]):
            if oo == 0 and ooo == 0:  
                for i, solver in enumerate(["approx", "exact"]):
                    for j, TS in enumerate(["soft", "hard"]):
                        for k, GT_rho_E in enumerate([False, True]):
                            for l, GT_rho_I in enumerate([False, True]):
                                s = Z + " "
                                s += solver + " MDP + "
                                s += TS + " expert + "
                                s += ("estimated" if k == 0 else "GT") + " d^E +"
                                s += ("estimated" if l == 0 else "GT") + " d^I"
                                clear(s)
            
            try:
                os.mkdir("figs_new/"+str(expert_traj)+"-"+str(noise_level))
            except:
                pass 
            for TA_traj in [10, 100, 1000, 10000]:
            
                #arr1 = process("ours_Entreg")
                arr2 = process(TA_traj, expert_traj, noise_level, "LP")
                # arr3 = process("ours_KLreg")
                arr4 = process(TA_traj, expert_traj, noise_level, "ours-newKL")
                arr5 = process(TA_traj, expert_traj, noise_level, "SMODICE-CHI")
                arr6 = process(TA_traj, expert_traj, noise_level, "SMODICE-KL")
                arr7 = process(TA_traj, expert_traj, noise_level, "LobsDICE")
                
                # print(arr2.shape, arr4.shape, arr5.shape, arr6.shape, arr7.shape)
                datas = [arr2, arr4, arr5, arr6, arr7]
            
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
                                    add_point([arr[:, x * 3 + o] for arr in datas], s, oo, ooo, output_flag)
                                
            """                    
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
                            plt.cla()
                            for _ in range(len(colors)):
                                plt.plot([10, 100, 1000, 10000], mean_buff[s][_], alpha=1, color=colors[_], label=name_lst[_])
                                plt.fill_between([10, 100, 1000, 10000], mean_buff[s][_] - std_buff[s][_], mean_buff[s][_] + std_buff[s][_], alpha=0.5, color=colors[_])
                            plt.xlabel("# Non-Expert Steps")
                            plt.ylabel(formal_name[o])
                            plt.legend()
                            plt.xscale('log')
                            if Z != "regret" or TS == "hard": plt.yscale('log')
                            plt.tight_layout()
                            plt.savefig("figs_new/"+str(expert_traj)+"-"+str(noise_level)+"/arbitrary-"+s+".png")
            """
for i, solver in enumerate(["approx", "exact"]):
    for j, TS in enumerate(["soft", "hard"]):
        for k, GT_rho_E in enumerate([False, True]):
                for l, GT_rho_I in enumerate([False, True]):            
                    for o, Z in enumerate(["TVss", "TVs", "regret"]):
                        
                        fig, axes = plt.subplots(nrows=3, ncols=4, sharex=True)
                        
                        fig.set_figheight(12)
                        fig.set_figwidth(30)
                        plt.xlabel('# Non-Expert Steps', fontsize=15)
                        plt.ylabel(formal_name[o], fontsize=15)
                        
                        plt.xscale('log')
                        
                        for eks, noise_level in enumerate([0.01, 0.1, 1]):
                             for wai, expert_traj in enumerate([10, 100, 1000, 10000]):
                                 ax = axes[eks, wai]
                                 ax.set_title("#Expert Step "+str(expert_traj)+", noise "+str(noise_level))
                                     
                                 x = i * 8 + j * 4 + k * 2 + l
                                 s = Z + " "
                                 s += solver + " MDP + "
                                 s += TS + " expert + "
                                 s += ("estimated" if k == 0 else "GT") + " d^E +"
                                 s += ("estimated" if l == 0 else "GT") + " d^I"
                                 
                                 if Z != "regret" or TS == "hard": 
                                     ax.set_yscale('log')
                                     mean_buff[s][eks][wai] = np.maximum(mean_buff[s][eks][wai], 1e-30)
                                 
                                 print(mean_buff[s])
                                 
                                 for _ in range(len(colors)):
                                     if eks == 0 and wai == 0: 
                                         print("label triggered!", name_lst[_])
                                         ax.plot([10, 100, 1000, 10000], mean_buff[s][eks][wai][_], alpha=1, color=colors[_], label=name_lst[_])
                                     else: ax.plot([10, 100, 1000, 10000], mean_buff[s][eks][wai][_], alpha=1, color=colors[_])
                        fig.legend(loc='center', bbox_to_anchor=(0.5, 0.05),  fancybox=True, shadow=True, ncol=5)     
                        fig_name = Z+"-"+solver + " MDP + " + TS + " expert + " + ("estimated" if k == 0 else "GT") + " d^E +" + ("estimated" if l == 0 else "GT") + " d^I"
                        plt.savefig("figs_new/"+str(expert_traj)+"-"+str(noise_level)+"-arbitrary-"+fig_name+".png", bbox_inches='tight', pad_inches=0.05)         
                        plt.cla()
                        # exit(0)
                    
         
