import gym
import random
from math import *

INF=float("inf")

n_rewards=0.0
sum_rewards=0.0
epoch=0

env = gym.make("LunarLander-v2")
env.reset()

def avg(a):
    return sum(a)/len(a)

def list_mul(a,b):
    assert type(a)==list and type(b)==list
    assert len(a)==len(b)
    c=[]
    for i in range(len(a)):
        c.append(a[i]*b[i])
    return c

def shrink(x):
    if x>1.0:
        return 1.0
    elif x<-1.0:
        return -1.0
    else:
        return x
def shrinka(a):
    a_new=[]
    for x in a:
        a_new.append(shrink(x))
    return a_new

def select_action(a):
    action=0
    m=-INF
    n=0
    for i in range(len(a)):
        if a[i]>THRESHOLD and a[i]>m:
            n=i+1
            m=a[i]
    return n

def aabs(a):
    for i in range(len(a)):
        a[i]=abs(a[i])
    return a
def sign(x):
    if x<0:
        return -1
    elif x==0:
        return 0
    else:
        return 1

class Agent:
    def __init__(self,n_input,n_layers,n_hidden,n_out,k_positive=2.0,inhibit=0.4 ,lr=0.2):
        self.n_input=n_input
        self.n_layers=n_layers
        self.n_hidden=n_hidden
        self.n_out=n_out
        self.sz=n_input+n_hidden
        self.network=[]
        self._activity=[]
        self.lr=lr
        self._k_positive=k_positive
        for _ in range(n_layers):
            l=[]
            for _ in range(self.sz**2+1):
                l.append(random.uniform(-1.0,1.0))
            self.network.append(l)
        self._output=[random.uniform(-1.0,1.0) for _ in range(self.sz)]
        self.inhibit=inhibit

    def forward_layer(self,inp,layer):
        #print(inp)
        #print(layer)
        _out=[]
        for i in range(self.sz):
            _out.append(self.inhibit*sum(list_mul(inp,layer[self.sz*i:self.sz*(i+1)])))
        return _out

    def forward(self,inp):
        self._activity=[]
        assert len(inp)==self.n_input, "Expeted length %d, but got length %d"%(self.n_input,len(inp))
        inp=inp+self._output[self.n_input:]
        for layer in self.network:
            self._activity.append(inp)
            inp=self.forward_layer(inp, layer)
        self._output=inp
        return inp[:self.n_out]

    def reward_layer(self, layer, activity, reward):
        assert self.sz==len(activity)
        if reward>0:
            reward*=self._k_positive
        for i in range(self.sz):
            for j in range(i,len(layer),self.sz):
                layer[j]=layer[j]+activity[i]*sign(layer[j])*reward*self.lr
        return layer

    def reward(self,_reward):
        assert len(self._activity)==len(self.network)
        for i in range(len(self.network)):
            self.network[i] = self.reward_layer(self.network[i],self._activity[i],_reward)
    def reset(self):
        for i in range(len(self._output)):
            self._output[i]=0.0


def loop(observation):
    global n_rewards
    global sum_rewards
    global epoch
    observation=observation.tolist()
    agent=Agent(8,17,12,3)
    env.render()
    decision=agent.forward(observation)
    action=select_action(decision)
    observation, reward, done, info = env.step(action)
    n_rewards=n_rewards+1
    sum_rewards=sum_rewards+reward
    avg_reward=sum_rewards/n_rewards
    agent.reward(reward)
    if done:
        observation=env.reset()
        
        print("[DONE #%d]"%epoch,"Reward:%.3f"%reward,";Avg reward:%.3f"%avg_reward,"Decision:%.3f,%.3f"%(decision[0],decision[1]), "Reward sum:%.3f"%sum_rewards,"N:",n_rewards," "*8)
        agent.reset()
        n_rewards=0.0
        sum_rewards=0.0
        epoch+=1
    else:
        print("Reward:%.3f"%reward,";Avg reward:%.3f"%avg_reward,"Decision:%.3f,%.3f"%(decision[0],decision[1]), "Reward sum:%.3f"%sum_rewards,"N:",n_rewards,end="   \r",flush=True)
    return reward,observation

if __name__=='__main__':
    observation=env.reset()
    try:
        while True:
            reward,observation = loop(observation)
    except KeyboardInterrupt:
        print("Interrupted.")
        env.close()



