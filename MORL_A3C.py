# This is based on A3C.py
# import agent form MORL_agent
#

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from New_Network import (ActorNetwork, CriticNetwork)
from datetime import datetime

PATH = './results/'

RAND_RANGE = 1000


class A3C(object):
    def __init__(self, is_central, model_type, s_dim, action_dim, actor_lr=1e-4, critic_lr=1e-3):
        self.s_dim = s_dim
        self.a_dim = action_dim
        self.discount = 0.99
        self.entropy_weight = 0.5
        self.entropy_eps = 1e-6
        self.model_type = model_type

        self.is_central = is_central
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actorNetwork = ActorNetwork(self.s_dim, self.a_dim).double().to(self.device)
        '''
        Original ActorNetwork
        input: state_dim, action_dim
        a_net = ActorNetwork()
        action=a_net.forward(npState)
        
        Now:
        input: input_size, output_size, reward_size
        output(policy)
        forward(state, preference)
        '''

        if self.is_central:
            # unify default parameters for tensorflow and pytorch
            self.actorOptim = torch.optim.RMSprop(self.actorNetwork.parameters(), lr=actor_lr, alpha=0.9, eps=1e-10)
            self.actorOptim.zero_grad()
            if model_type < 2:
                '''
                model==0 mean original
                model==1 mean critic_td
                model==2 mean only actor
                '''
                self.criticNetwork = CriticNetwork(self.s_dim, self.a_dim).double().to(self.device)
                self.criticOptim = torch.optim.RMSprop(self.criticNetwork.parameters(), lr=critic_lr, alpha=0.9,
                                                       eps=1e-10)
                self.criticOptim.zero_grad()
        else:
            self.actorNetwork.eval()

        self.loss_function = nn.MSELoss()

# Do I need to put find_preference here?
    def getNetworkGradient(self, s_batch, a_batch, r_batch, w_batch, terminal):
        s_batch = torch.from_numpy(s_batch).to(self.device)
        a_batch = torch.from_numpy(a_batch).to(self.device)
        R_batch = torch.zeros(r_batch.shape, dtype=torch.double).to(self.device)
        r_batch = torch.from_numpy(r_batch).to(self.device)
        w_batch = torch.FloatTensor(w_batch).to(self.device)

        if terminal:
            pass
        else:
            R_batch[-1, 0] = v_batch[-1, 0]
        for t in reversed(range(r_batch.shape[0] - 1)):
            R_batch[t, -1] = r_batch[t, -1] + self.discount * R_batch[t + 1, -1]

        if self.model_type < 2:
            with torch.no_grad():
                v_batch = self.criticNetwork.forward(s_batch, w_batch).to(self.device)
            td_batch = R_batch - v_batch
        else:
            td_batch = R_batch

        probability = self.actorNetwork.forward(s_batch, w_batch)
        actor_loss = torch.sum(torch.log(torch.sum(probability * a_batch, 1, keepdim=True)) * (
            -td_batch)) + self.entropy_weight * torch.sum(probability * torch.log(probability + self.entropy_eps))
        actor_loss.backward()

        if self.model_type < 2:
            if self.model_type == 0:
                # original
                critic_loss = self.loss_function(R_batch, self.criticNetwork.forward(s_batch))
            else:
                # cricit_td
                # Do I need to slice w_batch here???
                v_batch = self.criticNetwork.forward(s_batch[:-1], w_batch)
                next_v_batch = self.criticNetwork.forward(s_batch[1:], w_batch).detach()
                critic_loss = self.loss_function(r_batch[:-1] + self.discount * next_v_batch, v_batch)

            critic_loss.backward()

        # use the feature of accumulating gradient in pytorch

    # add preference to the action select
    def actionSelect(self, stateInputs, preference):
        stateInputs = torch.from_numpy(stateInputs).to(self.device)
        weights = torch.from_numpy(preference).to(self.device)
        if not self.is_central:
            with torch.no_grad():
                probability = self.actorNetwork.forward(stateInputs, weights)
                return probability.cpu().numpy()

    def hardUpdateActorNetwork(self, actor_net_params):
        for target_param, source_param in zip(self.actorNetwork.parameters(), actor_net_params):
            target_param.data.copy_(source_param.data)

    def updateNetwork(self):
        # use the feature of accumulating gradient in pytorch
        if self.is_central:
            self.actorOptim.step()
            self.actorOptim.zero_grad()
            if self.model_type < 2:
                self.criticOptim.step()
                self.criticOptim.zero_grad()

    def getActorParam(self):
        return list(self.actorNetwork.parameters())

    def getCriticParam(self):
        return list(self.criticNetwork.parameters())


if __name__ == '__main__':
    # test maddpg in convid,ok
    SINGLE_S_LEN = 19

    AGENT_NUM = 5
    BATCH_SIZE = 200

    S_INFO = 6
    S_LEN = 8
    ACTION_DIM = 6

    discount = 0.9

    IS_CENTRAL = True

    obj = A3C(IS_CENTRAL, 2, [S_INFO, S_LEN], ACTION_DIM, actor_lr=1e-4, critic_lr=1e-3)
    timenow = datetime.now()

    episode = 3000
    for i in range(episode):
        state2Select = np.random.randn(1, S_INFO, S_LEN)
        state = np.random.randn(AGENT_NUM, S_INFO, S_LEN)
        action = np.random.randn(AGENT_NUM, ACTION_DIM)
        reward = np.random.randn(AGENT_NUM, 1)
        weight = np.random.randn(AGENT_NUM, 1)
        # reward=0.47583
        # print('action: '+str(out))
        probability = obj.actionSelect(state2Select, weight)
        # return [[1,2,3,4,5,6]]
        # updateNetwork ok
        obj.updateNetwork()

    print('train:' + str(episode) + ' times use:' + str(datetime.now() - timenow))



