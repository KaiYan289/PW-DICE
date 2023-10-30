# import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections as mc
import matplotlib
import time

matplotlib.use('Agg')
import matplotlib.pyplot as plt
class Plotter:
    def __init__(self, S, st, ed, time, directory="fig"):
        self.S = S
        self.st, self.ed = st, ed
        print(S, st, ed)
        self.time = time.replace("/", "-").replace(" ", "_")
        self.stx, self.sty = self.get_pos(self.st)
        self.edx, self.edy = self.get_pos(self.ed)
        self.directory = directory
        self.fig = plt.figure()
        self.ax = plt.axes()
        
    def draw_circle(self, x, y, color):
        circle = plt.Circle((x, y), 0.4, color=color, fill=False)
        self.ax.add_patch(circle)

    def draw_arrow(self, x1, y1, x2, y2, color, scale=1):
        # self.ax.arrow(x1, y1, x2 - x1, y2 - y1, head_width=0.05 * scale, head_length=0.1 * scale, color=color)
        self.ax.arrow(x1, y1, scale * (x2 - x1), scale * (y2 - y1), head_width=0.15 * (0.25 + 0.75 * scale), head_length=0.3 * (0.25 + 0.75 * scale), color=color)
    def get_pos(self, x):
        return x // self.S, x % self.S
    
    def draw_traj(self, dataset, color):
        print("trajlen:", len(dataset))
        for i in range(len(dataset)):
            x1, y1 = self.get_pos(dataset[i]["state"])
            x2, y2 = self.get_pos(dataset[i]["next_state"])
            self.draw_arrow(x1 + 0.5, y1 + 0.5, x2 + 0.5, y2 + 0.5, color)

    def draw_grid(self):
        
        lines = [[(i, 0), (i, self.S)] for i in range(self.S + 1)] + [[(0, i), (self.S, i)] for i in range(self.S + 1)]
        #lc = mc.LineCollection(lines)
        #self.ax.add_collection(lc)
        for line in lines: 
            print(line)
            plt.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], color='black')
        #print(plt.xlim(), plt.ylim())
        #plt.xlim(-10, self.S + 10)
        #plt.ylim(-10, self.S + 10)
        # print(plt.xlim(), plt.ylim())
        # print("lines:", lines)
        # self.ax.autoscale()
        # self.ax.margins(0.1)
        # print(self.S, self.stx, self.sty)
        self.draw_circle(0.5 + self.stx, 0.5 + self.sty, 'green')
        self.draw_circle(0.5 + self.edx, 0.5 + self.edy, 'red')
        
    def draw_policy(self, pi):
      
        for i in range(self.S):
            for j in range(self.S):
                dx, dy = [0, 0, 0.5, -0.5], [0.5, -0.5, 0, 0] # 0 = right, 1 = left, 2 = down, 3 = up
                state = i * self.S + j
                for action in range(len(dx)):
                    arrow_goal = (i + 0.5 + dx[action], j + 0.5 + dy[action])
                    p = pi[state, action]
                    # print("p:", p)
                    if p > 0.01: 
                        # print("drawing:", state, action)
                        self.draw_arrow(i + 0.5, j + 0.5, arrow_goal[0], arrow_goal[1], 'blue', scale = p * 0.75)
                    
    def clear(self):
        plt.cla()
    def save(self, prefix=""):
        plt.savefig(self.directory+"/"+prefix+"-fig-"+str(self.time)+".png")
        
if __name__ == "__main__":
    a = Plotter(9, 20, 70)
    a.clear()
    a.draw_grid()
    a.save()
