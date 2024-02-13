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
import json

learning_rate=0.002
gamma=0.98
action_size=1
n_rollout= 50
print_interval=10
save_interval=20
buffer_limit  = 50000
batch_size    = 200

getAction_StartPoint=15000
train_StartPoint=10000

load_model=False
train_mode=True
load_Buffer=False
save_Buffer=False

game="RL_Fighting"
os_name=platform.system()
env_name=f"C:/Unity_Program/ActionFighting/{game}.exe"
if os_name=='Windows':
    env_name=f"C:/Unity_Program/ActionFighting/{game}.exe"
elif os_name=='Darwin':
    env_name=f"../envs/{game}_{os_name}"

date_time=datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path=f"C:/ML/SavedModels/ActionFighting_DQN/{date_time}"
load_path=f"C:/ML/SavedModels/ActionFighting_DQN/20240213164534"
if not load_model:
    os.makedirs(save_path, exist_ok=True)
else:
    save_path=load_path

BufferFile_Name='ActionBuffer.json'

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.__version__)

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
        # 리스트를 텐서로 변환시 속도가 좋지않아 넘파이 배열로 변환
        s_arr = np.array(s_lst, dtype=np.float32)
        a_arr = np.array(a_lst, dtype=np.int64)
        r_arr = np.array(r_lst)
        s_prime_arr = np.array(s_prime_lst, dtype=np.float32)
        done_mask_arr = np.array(done_mask_lst)

        return torch.tensor(s_arr, dtype=torch.float).to(device), torch.tensor(a_arr, dtype=torch.int64).to(device), \
               torch.tensor(r_arr).to(device), torch.tensor(s_prime_arr, dtype=torch.float).to(device), \
               torch.tensor(done_mask_arr).to(device)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(8, 258)
        self.fc2 = nn.Linear(258, 512)
        self.fc3 = nn.Linear(512, 9)

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
            return random.randint(0, 8)
        else : 
            if memory_size>getAction_StartPoint:
                return out.argmax().item()
            else:
                return random.randint(0, 8) # 0~8중 랜덤으로 할당.

    def save_model(self,network, optimizer):
        print("save model")
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
        q_a = q_out.gather(1,a) #q_out은 batch_size만큼, 각 상태s에 대해서 행동들에대한 가치를 전부
                                #가지고 있다 ex) [s1[가치1,가치2...],s2[],s3[]....] 
                                #여기서 gather()의 인자 a는 [[a1],[a2],[a3]...] 똑같이 2차원이고 batchSize만큼있다
                                #샘플에 넣을때[a]해준 이유다. 인자1은 a의 행,열중에 열로 내려가서 
                                #해당값을 q_out의 인덱스로 쓰겠다는뜻이다. 가로로 적었지만 저걸 세로로 보면된다.
                                #즉 a1값이 1이라고하면 s1[가치2]를 뽑고, a2값이 3이면 s2[가치4]를 뽑아서 q_a에 저장하는것이다.
                                #비로소 q_a는 batchSize만큼 있는 각행동의 가치 집합이다.
        #print(q_a) q_a를 print해보면 똑같이 2차원인걸 알수있고, 차원의 통일은 중요한부분이다.
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
        engine_configuration_channel.set_configuration_parameters(time_scale=10)
    else :
        engine_configuration_channel.set_configuration_parameters(time_scale=2)
    dec, term=env.get_steps(behavior_name)
    actor_losses, critic_losses, scores, episode, score=[],[],[],0,0
    q = Qnet().to(device)

    memory = ReplayBuffer()
    
    if load_Buffer and train_mode:
        with open(BufferFile_Name,'r') as f:
            memory.buffer=collections.deque(json.load(f))
            print("load buffer")

    score = 0.0  

    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    agent=Qnet_agent(q,optimizer)
    q_target = Qnet().to(device)
    q_target.load_state_dict(q.state_dict())
    
    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
       
        done = False

        while not done:
            s=dec.obs[0]
            s=torch.tensor(s, dtype=torch.float).to(device)
            s=s.squeeze()
            s=s.cpu().detach().numpy() # CUDA 장치에서 호스트 메모리로 텐서를 복사 후 NumPy 배열로 변환
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
            s_prime=s_prime.cpu().detach().numpy()

            done_mask = 0.0 if done else 1.0

            if train_mode:
                memory.put((s.tolist(),a,r.tolist(),s_prime.tolist(), done_mask))

            s = s_prime
            #print(done)
            score += r
            if memory.size()%100==0:
                print(memory.size())
            if done:
                break

        print(score)

        if save_Buffer and train_mode:
            with open(BufferFile_Name,'w') as f:
                json.dump(list(memory.buffer),f)
                print("save buffer")

        if memory.size()>train_StartPoint:
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