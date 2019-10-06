import os
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from six.moves import xrange
import scipy.io
import collections
from sklearn.metrics import roc_curve, auc
import random
from torch.autograd import Variable
from torch.distributions.normal import Normal





class Atention_Op(nn.Module):
    def __init__(self,hidden_units,embed_size,steps,strd_id):
        super(Atention_Op,self).__init__()
        self.hidden_units = hidden_units
        self.embed_size = embed_size
        self.steps = steps
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.strd_id = strd_id
        if strd_id == 'alpha':
            self.p_att_shape = [self.hidden_units,1]
        elif strd_id == 'beta':
            self.p_att_shape = [self.hidden_units,self.embed_size]
        else:
            raise Value("You must re-check the attention id \n Required to \'alpha\' or \'beta\' ")

        self.mu = nn.Linear(self.p_att_shape[0],self.p_att_shape[1])
        self.sigma = nn.Linear(self.p_att_shape[0],self.p_att_shape[1])
        # print("Attention OP")
        # print(self.named_parameters(),'gaga')

        # for name,param in self.named_parameters():
        #     print('gaga')
        #     print(name,param.data)

    def forward(self,rnn_outputs):
        #Create Mu
        mu = []
        for _i in range(self.steps):
            #mu_tmp = torch.matmul(rnn_outputs[:,_i,:],mu_w)+mu_b
            mu_tmp = self.mu(rnn_outputs[:,_i,:])
            mu.append(mu_tmp)
        mu = torch.cat(mu,dim=1).view(-1,self.steps,self.p_att_shape[1])

        #Create Sigma
        sigma = []
        for _k in range(self.steps):
            sigma_tmp = self.sigma(rnn_outputs[:,_k,:])
            sigma.append(sigma_tmp)
        sigma = torch.cat(sigma,1).view(-1,self.steps,self.p_att_shape[1])
        sigma = self.softplus(sigma)

        distribution = Normal(loc=mu,scale=sigma)

        att = distribution.rsample([1])
        att = torch.squeeze(att,0)


        if self.strd_id == 'alpha':
            squashed_att = self.softmax(att)
            #print('Done with generating alpha attention.')
        elif self.strd_id == 'beta':
            squashed_att = self.tanh(att)
            #print('Done with generating beta attention.')
        else:
            raise ValueError('You must re-check the attention id. required to \'alpha\' or \'beta\'')

        return squashed_att



class UA_Model(nn.Module):
    def __init__(self,args):
        super(UA_Model,self).__init__()
        self.num_features = args.num_features_UA
        self.steps = args.steps_UA
        self.hidden_units = args.hidden_units_UA
        self.embed_size = args.embed_size_UA
        self.num_layers = args.num_layers_UA
        print('Making Model...')
        self.lstm_net = nn.Sequential()
        self.dropout = 0.25
        #self.model.add_module(lstm_block_cell(args.num_features,self.hidden_units,args.num_layers))
        self.lstm_net.add_module('lstm-net-1',nn.LSTM(input_size=self.embed_size,
            hidden_size=self.hidden_units,num_layers=self.num_layers,dropout=self.dropout))
        self.attention_op_alpha = Atention_Op(self.hidden_units,self.embed_size,
            self.steps,'alpha')
        self.attention_op_beta = Atention_Op(self.hidden_units,self.embed_size,
            self.steps,'beta')

        #self.embedding = nn.Embedding(self.num_features,self.embed_size)
        self.embedding = nn.Linear(self.num_features,self.embed_size,bias=False)
        self.prediction = nn.Sequential()
        self.prediction.add_module('FC1',nn.Linear(self.embed_size,1))
        self.prediction.add_module('softmax,',nn.Sigmoid())

        if args.cuda:
            self.lstm_net.cuda()
            self.embedding.cuda()
            self.attention_op_alpha.cuda()
            self.attention_op_beta.cuda()
            self.prediction.cuda()
        print('Done with building the model')
        for name,param in self.named_parameters():
            print(name,param.data.shape,True==param.requires_grad)



    def forward(self,x):
        if not x.is_cuda:
            x = x.cuda()
        embedding_v = self.embedding(x)
        #print(embedding_v.shape,'Embedding v')
        reverse_embed = torch.flip(embedding_v,[1])
        #print(reverse_embed.shape,'reverse embedding v and shape ', reverse_embed )
        alpha_rnn_outputs, _ = self.lstm_net(reverse_embed)
        beta_rnn_outputs, _ = self.lstm_net(reverse_embed)
        #print(alpha_rnn_outputs.shape,'alpha rnn outputs')
        #print(beta_rnn_outputs.shape,'beta rnn outputs')
        #alpha
        alpha_embed_output = self.attention_op_alpha (alpha_rnn_outputs)
        self.rev_alpha_embed_output = torch.flip(alpha_embed_output,[1])
        #print(alpha_embed_output.shape,self.rev_alpha_embed_output.shape,'alpha embed output and reverse embed output')
        #beta
        beta_embed_output = self.attention_op_beta(beta_rnn_outputs)
        self.rev_beta_embed_output = torch.flip(beta_embed_output,[1])
        #print(beta_embed_output.shape,self.rev_alpha_embed_output.shape,'beta embed output')
        #attention sum
        c_i = torch.sum(self.rev_alpha_embed_output*(self.rev_beta_embed_output*embedding_v),1)
        #print(c_i.shape,'shape of attention sum')
        return self.prediction(c_i)












