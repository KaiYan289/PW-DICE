import numpy as np
import math
from visualizer import Plotter
from tqdm import tqdm
class TabularMDP:
    def __init__(self, n, m, max_step):
        self.n, self.m = n, m
        self.p0 = np.zeros(n)
        self.T, self.r = None, None
        self.gamma = 0.99
        self.max_step = max_step
        self.generate_dynamics() 
        
    def generate_dynamics(self):
        pass
    
    def check_p0(self):
        return abs(self.p0.sum() - 1) < 1e-10
    
    def check_transition(self):
        for i in range(self.n):
            for j in range(self.m):
                if abs(self.T[:, i, j].sum() - 1) >= 1e-10: return False
        return True
    
    def evaluation(self, pi, collect=False, log=False, deterministic=False, return_reward=False):
        buf = []
        self.EP_LEN = self.max_step
        T = self.EP_LEN
        state = np.random.choice(self.n, 1, p=self.p0)[0]
        r, l = 0, 0
        # print("EP_len:", self.EP_LEN, "collect:", collect, "T:", T, "state:", state, "self.ed:", self.ed)
        while state != self.ed and T > 0:
            old_state = state
            if not deterministic: action = np.random.choice(self.m, 1, p=pi[state])[0]
            else: action = np.argmax(pi[state])
            # print("action:", action)
            # print(self.T[:, state, action].reshape(-1))
            state = np.random.choice(self.n, 1, p=self.T[:, state, action].reshape(-1))[0]
            # state = np.argmax(self.T[:, state, action].reshape(-1))
            if collect: 
                buf.append({"state": old_state, "action": action, "next_state": state, "step": self.EP_LEN - T}) 
            # if log: print("state:", state)
            l += 1
            r += self.r[state]
            T -= 1
        if log: print("reward:", r)
        # print("buf:", buf)
        if not return_reward: return buf
        else: return buf, r, l
        
def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x - np.max(x)))
    return f_x

class Arbitrary(TabularMDP):
    def __init__(self, n, m, noise=0.01):
        self.noise = noise
        self.connectivity = 4
        self.edges = np.zeros((n, n))
        super().__init__(n, m, 999999) # infinite horizon MDP
        self.p0 = np.array([1] + [0 for _ in range(self.n - 1)])
        self.ed = -1 # unending
    
    def check_connectivity(self):
        vis = np.zeros(self.n)
        vis[0] = 1
        q = [0]
        cnt = 1
        while len(q) > 0:
            x = q.pop(0)
            for y in range(self.n):
                if self.edges[x, y] == 1 and vis[y] == 0:
                    q.append(y)
                    vis[y], cnt = 1, cnt + 1 
        return cnt == self.n 
    
    def generate_dynamics(self):
        # tot_connect = 0
        # for i in tqdm(range(100000)):
        self.gamma = 0.95
        self.T = np.zeros((self.n, self.n, self.m))
        for i in range(self.n):
            for j in range(self.m):
                target = np.random.choice(self.n, self.connectivity, replace=False)
                X = np.random.multinomial(1, [1 / self.connectivity] * self.connectivity)
                Y = np.random.dirichlet(tuple([1 for _ in range(self.connectivity)]))
                transition = (1 - self.noise) * X + self.noise * Y
                for k in range(target.shape[0]):
                    self.T[target[k], i, j] = transition[k]
                    self.edges[i, target[k]] = 1
        #    tot_connect += self.check_connectivity()
        # print("check connectivity:", tot_connect / 100000)
        
        V_rec, Q_rec, r_rec = 10 * np.ones(self.n), None, None
        
        for i in range(1, self.n):
            self.r = np.zeros((self.n, 1)) # insignificant in lobsdice experiment
            self.r[i] = 1 
            T = self.T.transpose(1, 2, 0)
            # trying to solve the MDP
            V, Q = np.zeros(self.n), np.zeros((self.n, self.m))
            for _ in range(100000):
                Q_new = self.r + self.gamma * T.dot(V)
                V_new = np.max(Q_new, axis=1)
    
                if np.max(np.abs(V - V_new)) < 1e-8:
                    break
    
                V, Q = V_new, Q_new
            # print(i, "V:", V)
            if V[0] < V_rec[0]: V_rec, Q_rec, r_rec = V.copy(), Q.copy(), i 

        pi, pi2 = np.zeros((self.n, self.m)), np.zeros((self.n, self.m))
        # pi[np.arange(self.n), np.argmax(Q, axis=1)] = 1.
        for i in range(self.n):
            pi[i] = softmax(Q_rec[i])
            pi2[i, Q_rec[i].argmax()] = 1
            # print("Q[", i, "] =", Q_rec[i], "pi[", i, "] =", pi[i])
        # print("pi:", pi)
        self.optimal_Q = Q_rec
        self.expert_pi_argmax = pi2
        self.expert_pi = pi
        for i in range(self.n):
            for j in range(self.m):
                print(self.expert_pi[i, j], end=" ")
            print()
                    
        self.target = r_rec
        
    def evaluation(self, pi, collect=False, log=False, deterministic=False, return_reward=False, max_step=1000):
        self.max_step = max_step
        return super().evaluation(pi, log=False, deterministic=deterministic, return_reward=return_reward, collect=collect)
        
    def generate_argmax_expert_traj(self, N):
        return self.evaluation(self.expert_pi_argmax, collect=True, max_step=N)
    
    def generate_expert_traj(self, N):   
        return self.evaluation(self.expert_pi, collect=True, max_step=N)
    
    def generate_random_traj(self, N):
        return self.evaluation(np.ones((self.n, self.m)) / self.m, collect=True, max_step=N)
                
class GridWorld(TabularMDP):
     
    def get_idx(self, x, y):
        return x * self.S + y
    def get_pos(self, s):
        return s // self.S, s % self.S
    
    def __init__(self, n, stx, sty, edx, edy, noise=0.05, max_step=25): # n * n grid, (stx, sty) as start, (edx, edy) as end
        self.S = n
        self.n = n ** 2
        self.st, self.ed = self.get_idx(stx, sty), self.get_idx(edx, edy)
        self.stx, self.sty, self.edx, self.edy = stx, sty, edx, edy
        self.noise = noise
        super().__init__(self.n, 4, max_step)
        
    
    def generate_expert_traj(self, N, balance=False):
        pi = np.zeros((self.n, self.m))
        if not balance:
            for i in range(self.n):
                x, y = self.get_pos(i)
                if self.edy < y: pi[i, 1] = 1 # left
                elif self.edy > y: pi[i, 0] = 1 # right
                elif self.edx < x: pi[i, 3] = 1 # up
                elif self.edx > x: pi[i, 2] = 1 # down  
                else: pi[i] = np.ones(self.m) / self.m # at optimal point
            buf = []
            for i in range(N):
                buf += self.evaluation(pi, collect=True)
        else:
            print("balance!")
            for i in range(self.n):
                x, y = self.get_pos(i)
                if x == self.edx and y == self.edy: 
                    pi[i] = np.ones(self.m) / self.m
                    continue
                if self.edy < y: pi[i, 1] = 1 # left
                if self.edy > y: pi[i, 0] = 1 # right
                if self.edx < x: pi[i, 3] = 1 # up
                if self.edx > x: pi[i, 2] = 1 # down
                pi[i] /= pi[i].sum()  
                # print(i, pi[i])
                # else: pi[i] = np.ones(self.m) / self.m # at optimal point
            buf = []
            for i in range(N):
                traj = self.evaluation(pi, collect=True)
                buf += traj
                print("buf:", traj[0]["state"], traj[0]["action"])
                # print("buf:", [(x["state"] // self.S, x["state"] % self.S) for x in traj])
        return buf
        
    def generate_random_traj(self, N, optimality=0):
        pi_random = np.ones((self.n, self.m)) / self.m
        pi_expert = np.zeros((self.n, self.m))
        for i in range(self.n):
            x, y = self.get_pos(i)
            if self.edy < y: pi_expert[i, 1] = 1 # left
            elif self.edy > y: pi_expert[i, 0] = 1 # right
            elif self.edx < x: pi_expert[i, 3] = 1 # up
            elif self.edx > x: pi_expert[i, 2] = 1 # down  
            else: pi_expert[i] = np.ones(self.m) / self.m # at optimal point
        pi = optimality * pi_expert + (1 - optimality) * pi_random
        """
        if self.S < 11: pi = np.ones((self.n, self.m)) / self.m
        else: 
            # pi = np.array([[0.25, 0.25, 0.25, 0.25] for i in range(self.n)])
            # print("pi!")
            
            exp_pi = np.zeros((self.n, self.m))
            for i in range(self.n):
                x, y = self.get_pos(i)
                if self.edy < y: exp_pi[i, 1] = 1 # left
                elif self.edy > y: exp_pi[i, 0] = 1 # right
                elif self.edx < x: exp_pi[i, 3] = 1 # up
                elif self.edx > x: exp_pi[i, 2] = 1 # down  
                else: exp_pi[i] = np.ones(self.m) / self.m # at optimal point
            pi = 0.3 * np.ones((self.n, self.m)) / self.m + 0.7 * exp_pi
        """
        buf = []
        for i in range(N):
            # old_len = len(buf)
            buf += self.evaluation(pi, collect=True)
            # if len(buf) - old_len < 30: print("success:", i, len(buf) - old_len)
        return buf
    
    def generate_dynamics(self):
        m = self.m
        self.T = np.zeros((self.n, self.n, 4))
        assert (self.S ** 2) == self.n, "Not Equal!"

        
        def not_feasible(tar_i, tar_j):
            return tar_i >= self.S or tar_i < 0 or tar_j >= self.S or tar_j < 0
        dx, dy = [0, 0, 1, -1], [1, -1, 0, 0] # 0 = right, 1 = left, 2 = down, 3 = up
        
        # set initial probability
        """
        tot_prob = 1
        for k in range(len(dx)):
            tar_i, tar_j = self.stx + dx[k], self.sty + dy[k]
            if not_feasible(tar_i, tar_j): continue
            tar = self.get_idx(tar_i, tar_j)
            tot_prob -= self.noise
            self.p0[tar] += self.noise
        self.p0[self.st] += tot_prob
        """
        self.p0[self.st] = 1
        
        # transition
        S = self.S
        for i in range(S):
           for j in range(S):
               cur = self.get_idx(i, j)
               for k in range(len(dx)):
                   tar_i, tar_j = i + dx[k], j + dy[k]
                   if not_feasible(tar_i, tar_j): tar = cur # stay
                   else: tar = self.get_idx(tar_i, tar_j)
                   tot_prob = 1
                   for l in range(len(dx)):
                       if l == k: continue
                       t_i, t_j = i + dx[l], j + dy[l]
                       if not_feasible(t_i, t_j): t = cur
                       else: t = self.get_idx(t_i, t_j)
                       tot_prob -= self.noise
                       self.T[t, cur, k] += self.noise  # p(s' | s, a)
                       # print("t:", t, "cur:", cur, "k:", k, "val=", self.T[t, cur, k], "l:", l)
                   self.T[tar, cur, k] += tot_prob
                   # print("next state:", tar, "current state:", cur, "action:", k, "val=", self.T[tar, cur, k])
                   # print("state =", cur, "action =", k, self.T[:, cur, k])
        self.r = -np.ones(self.n) * 0.01
        self.r[self.ed] = 1
        
if __name__ == "__main__":
    arbit = Arbitrary(20, 4)
    