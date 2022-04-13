import time
import cv2.cv2 as cv2
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import GameCapture as cap
import GameData as data
import GameInput as input

trans = T.ToTensor()
# 建立 matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# 如果使用gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """保存变换"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, h, w):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # 线性输入连接的数量取决于conv2d层的输出，因此需要计算输入图像的大小。
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, 6) # 448 或者 512

    # 使用一个元素调用以确定下一个操作，或在优化期间调用批处理。返回张量
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))






BATCH_SIZE = 32
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


init_screen = cv2.resize(cap.grab_screen(region=cap.Full_Screen),(224,224))

print(init_screen.shape)
screen_height, screen_width,_ = init_screen.shape

policy_net = DQN(screen_height, screen_width).to(device)
target_net = DQN(screen_height, screen_width).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0



def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1）将为每行的列返回最大值。max result的第二列是找到max元素的索引，因此我们选择预期回报较大的操作。
            t1=trans(cap.ROI(state,*cap.Full_Screen)).unsqueeze(0).to(device)
            return policy_net(t1).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(6)]], device=device, dtype=torch.int32)
def take_action(action):
    if action == 0:     # n_choose
        pass
    elif action == 1:   # w
        input.up()
    elif action == 2:   # s
        input.down()
    elif action == 3:   # d
        input.right()
    elif action == 4:   # a
        input.left()
    elif action == 5:  # space
        input.confirm()
    elif action ==6:
        input.choose(1)
    elif action ==7:
        input.choose(2)
    elif action ==8:
        input.choose(3)
    elif action ==9:
        input.choose(4)
def get_reward(state,next_state):
    old=data.get_state(state)
    new=data.get_state(next_state)

    hp_var=new[0]-old[0]
    exp_var=new[1]-old[1]
    if hp_var==0:
        if exp_var>0:
            return exp_var*2,False
        elif exp_var==0:
            if new[1]!=98:
                take_action(5)
                if data.get_self_HP()==46:
                    return -200,True
                else: return 1,False
            return -1,False
    return 5*hp_var,False


episode_durations = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float32)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # 平均 100 次迭代画一次
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # 暂定一会等待屏幕更新
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # 转置批样本(有关详细说明，请参阅https://stackoverflow.com/a/19343/3343043）。这会将转换的批处理数组转换为批处理数组的转换。
    batch = Transition(*zip(*transitions))

    # 计算非最终状态的掩码并连接批处理元素(最终状态将是模拟结束后的状态）
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None]).to(device)

    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    # 计算Q(s_t, a)-模型计算 Q(s_t)，然后选择所采取行动的列。这些是根据策略网络对每个批处理状态所采取的操作。
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 计算下一个状态的V(s_{t+1})。非最终状态下一个状态的预期操作值是基于“旧”目标网络计算的；选择max(1)[0]的最佳奖励。这是基于掩码合并的，这样当状态为最终状态时，我们将获得预期状态值或0。
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # 计算期望 Q 值
    expected_state_action_values = ((next_state_values * GAMMA) + reward_batch).float()


    # 计算 Huber 损失
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # 优化模型
    optimizer.zero_grad()
    loss.backward()
#    for param in policy_net.parameters():
#        param.grad.data.clamp_(-1, 1)
    optimizer.step()

num_episodes = 50
for i_episode in range(num_episodes):
    # 初始化环境和状态
    state=cap.grab_screen(region=(0,0,960,1080))
    print("start%d" % i_episode  )
    for t in count():
        # 选择并执行动作
        action=select_action(state)
        take_action(action.item())
        next_state=cap.grab_screen(region=(0,0,960,1080))
        reward,done=get_reward(state,next_state)
        reward = torch.tensor([reward], device=device)

        state_tensor=trans(cv2.resize(cap.ROI(state,*cap.Full_Screen),(224,224))).unsqueeze(0)
        next_state_tensor=trans(cv2.resize(cap.ROI(next_state,*cap.Full_Screen),(224,224))).unsqueeze(0)
        # 在内存中储存当前参数
        memory.push(state_tensor, action,next_state_tensor, reward)

        # 进入下一状态
        state = next_state

        # 记性一步优化 (在目标网络)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            input.restart()
            time.sleep(0.5)
            break
    #更新目标网络, 复制在 DQN 中的所有权重偏差
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

torch.save(policy_net.state_dict(),"params.pth")
print('Complete')
plt.ioff()
plt.show()
