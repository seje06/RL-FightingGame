import numpy as np
import random
import collections
import copy
import datetime
import platform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
#from torch.utils.tensorboard import SummaryWriter
from collections import deque
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import mlagents_envs
import mlagents
import os

learning_rate=0.002
gamma=0.98
action_size=1
n_rollout= 50
print_interval=10
save_interval=20
buffer_limit  = 50000
batch_size    = 200

load_model=True
train_mode=False
run_step= 10000 if train_mode else 0
test_step=100

game="RL_Fighting"
os_name=platform.system()
env_name=f"C:/Unity_Program/MovingFighting/{game}.exe"
if os_name=='Windows':
    env_name=f"C:/Unity_Program/MovingFighting/{game}.exe"
elif os_name=='Darwin':
    env_name=f"../envs/{game}_{os_name}"

date_time=datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path=f"C:/ML/SavedModels/MovingFighting_DQN/{date_time}"
load_path=f"C:/ML/SavedModels/MovingFighting_DQN/20240204201636"
if not load_model:
    os.makedirs(save_path, exist_ok=True)
else:
    save_path=load_path

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float).to(device), torch.tensor(a_lst).to(device), \
               torch.tensor(r_lst).to(device), torch.tensor(s_prime_lst, dtype=torch.float).to(device), \
               torch.tensor(done_mask_lst).to(device)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 258)
        self.fc2 = nn.Linear(258, 512)
        self.fc3 = nn.Linear(512, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Qnet_agent:
    def __init__(self,q,optimizer):
        #self.writer=SummaryWriter(save_path)
        if load_model==True:
            print(f"...Load Model from {load_path}/ckpt ...")
            checkpoint = torch.load(load_path+'/ckpt.pth', map_location=device)
            q.load_state_dict(checkpoint["network"])
            optimizer.load_state_dict(checkpoint["optimizer"])

            #입실론 부분, 네트워크모드 생략
    
    def sample_action(self,q, obs, epsilon, memory_size):
        out = q.forward(obs)
        coin = random.random()
        if coin < epsilon and train_mode:
            return random.randint(0, 3)
        else : 
            if memory_size>20000:
                return out.argmax().item()
            else:
                return random.randint(0, 3) # 0,1,2,3중 랜덤으로 할당.

    def save_model(self,network, optimizer):
        print("save")
        torch.save({
            "network" : network.state_dict(),
            "optimizer" : optimizer.state_dict()
        }, save_path+"/ckpt.pth")

    def write_summary(self, score, loss, step):
        #self.writer.add_scalar("run/score",score,step)
        #self.writer.add_scalar("model/loss",loss, step)
        notting=1

def train(q, q_target, memory, optimizer,step_size):
    for i in range(2):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        #only_r=0 if step_size<3000 else 1
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad() # 그라디언트 0으로 초기화
        loss.backward() #평균을 내지 않았기에 32개의 오류요소가 있음으로 각 파라미터
                        #에 계산될 각 그라디언트는 32번 누적된다
        optimizer.step() #파라미터 갱신

def main():
    engine_configuration_channel=EngineConfigurationChannel()
    python_version = mlagents_envs.__version__
    print(python_version)
    env=UnityEnvironment(file_name=env_name,timeout_wait=10,
                         side_channels=[engine_configuration_channel])
    print("Success Communication")
    env.reset()
    behavior_name=list(env.behavior_specs.keys())[0]
    spec=env.behavior_specs[behavior_name]
    env.reset()
    if train_mode:
        engine_configuration_channel.set_configuration_parameters(time_scale=50)
    else :
        engine_configuration_channel.set_configuration_parameters(time_scale=2)
    dec, term=env.get_steps(behavior_name)
    actor_losses, critic_losses, scores, episode, score=[],[],[],0,0
    q = Qnet()

    memory = ReplayBuffer()
    score = 0.0  

    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    agent=Qnet_agent(q,optimizer)
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    
    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        #s, _ = env.reset()
        done = False

        while not done:
            s=dec.obs[0]
            s=torch.tensor(s, dtype=torch.float).to(device)
            s=s.squeeze()
            s=np.array(s)
            a = agent.sample_action(q,torch.from_numpy(s).float().to(device), epsilon, memory.size())   
            #print(a)
            a_tuple=ActionTuple()
            a_tuple.add_discrete(np.array([[a]]))
            env.set_actions(behavior_name,a_tuple)
            env.step()

            dec, term= env.get_steps(behavior_name)
            done=len(term.agent_id)>0
            r=term.reward if done else dec.reward 
            #print(r)
            s_prime=term.obs[0] if done else dec.obs[0]
            s_prime=torch.tensor(s_prime, dtype=torch.float).to(device)
            s_prime=s_prime.squeeze()
            s_prime=np.array(s_prime)

            done_mask = 0.0 if done else 1.0
            if train_mode:
                memory.put((s,a,r,s_prime, done_mask))
            s = s_prime
            #print(done)
            score += r
            if memory.size()%100==0:
                print(memory.size())
            if done:
                break

        print(score)

        if memory.size()>20000:
            if train_mode:
                print("train")
                train(q, q_target, memory, optimizer, memory.size())
                agent.save_model(q,optimizer)

        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())
            print(n_epi, score, memory.size(), epsilon*100)
            score = 0.0
    env.close()

if __name__ == '__main__':
    main()