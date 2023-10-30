import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cur = "goal"
f = open("data_"+cur+".txt", "r")
lines = f.readlines()
flag = 0
eks = [0, 10, 30, 50, 70, 90]

plt.title('deterministic_'+cur)
plt.xlabel("percent of poisonous data")
plt.ylabel("success rate")
for line in lines:
    content = line.split()
    if content[0] == "IDENTIFIER": 
        plt.savefig("determinstic_"+cur)
        plt.cla()
        plt.title('stochastic_'+cur)
        plt.xlabel("percent of poisonous data")
        plt.ylabel("success rate")
        continue
    wai = list(map(float, content[1:]))
    plt.plot(eks, wai, label=content[0])
    plt.legend()
plt.savefig("stochastic_"+cur)