"""
Pytorch Implementation of  "Uncetainty-Aware Attention for Reliable Interpretation and Prediction"
by Hae et al., 2017 NIPS
Author Jansen Domoguen
"""


import torch
import os
import argparse
import numpy as np
from skimage.transform import rotate
from torch.utils import data
from torch.autograd import Variable
import time
from model import *
from distutils.util import strtobool
from torch.utils.tensorboard import SummaryWriter
from model import UA_Model
from data_proc import *
from utils import *
import torch
import copy

parser = argparse.ArgumentParser()

dataset_dir = "physionet_dataset/physionet_dataset/1_"
#Data Info
parser.add_argument('--data-dir', type=str, default=dataset_dir,
                    help='dataset directory')
parser.add_argument('--data-name',type=str,default='omniglot',
                    help='Dataset tag:omniglot, miniImagenet, etc')
parser.add_argument('--batch-size', required=False, type=int, default=16,
                    help='Batch size for each task and inner gradient updates')
parser.add_argument('--output-dir',required=False,type=str,default=os.getcwd(),
                    help='Output directory of either saved model and/or output text')



#Training info
parser.add_argument('--mode', type=str, required=False, default='train',
                    help='Train or test mode. Train will consist of train and validation')
parser.add_argument('--gpu-num', type=int, required=False, default=0,
                    help='GPU device tag number')
parser.add_argument('--model', type=str,required=False, default='softmax',
            help='?')
parser.add_argument('--n-epochs', type=int, default=200,
            help='Total Epoch of Few-shot?')

parser.add_argument('--cuda', type=bool, default=True,
            help='if cuda or not')


#Training info relevant to Few-shot
parser.add_argument('--num-inn-updt', type=int, required=False, default=1,
                help='number of inner gradient updates during training.')
parser.add_argument('--K-shot', type=int, default=5,
                    help='K_shot size (default: 5)')
parser.add_argument('--num-classes', type=int, default=5,
                    help='N-Way classes (default: 5-way classes)')
parser.add_argument('--meta-train-iter', type=int, default=15000,
                    help='Number of Iterations for different Tasks')

parser.add_argument('--learning-rate-meta', type=float, default=0.001,
                    help='The base learning rate of the generator')
parser.add_argument('--learning-rate-inner', type=float, default=1e-3,
                    help='step size alpha for inner gradient update.')


#**********
#Sanity Check
#**********

parser.add_argument('--print-vars', type=bool, default=False,
                    help='Whether to print all trainable parameters for sanity check '
                         '(default: False)')


#Directory and Saving
parser.add_argument('--save-interval',required=False, type=int, default=100,
                    help="Number of epochs between saving model")
parser.add_argument('--savedir',required=False, type=str, default=os.getcwd(),
                help='Directory to save the models')



#*********
#Argument for UA
#*********


#Model Info
#Training Info
parser.add_argument('--task-UA',required=False,type=str,default='UA+_source_code',
                    help='?')
parser.add_argument('--max-epoch-UA',required=False,type=int,default=100,
                    help='Max Epoch for number of epochs')

parser.add_argument('--num-layers-UA',required=False,type=int,default=1,
                help ='Single Layer of RNN')

parser.add_argument('--num-features-UA',required=False,type=int,default=35,
                help ='Number of Features generated per input;related to RETAIN RNN')

parser.add_argument('--steps-UA',required=False,type=int,default=155,
                help ='Since dataset is a sequence, number of steps per input; RETAIN RNN')

parser.add_argument('--hidden-units-UA',required=False,type=int,default=33,
                help ='Number of units per layer in the retain RNN')

parser.add_argument('--embed-size-UA',required=False,type=int,default=33,
                help ='Number of units in embedding layer of RETAIN RNN')


parser.add_argument('--lr-UA',required=False,type=float,default=1e-4,
                help ='Learning Rate of RETAIN RNN')

parser.add_argument('--batch-size-UA',required=False,type=int,default=100,
                help ='Mini Batch Size of RETAIN')

parser.add_argument('--num-sampling-UA',required=False,type=int,default=30,
                help ='Number of MC dropout samples')

parser.add_argument('--lamb-UA',required=False,type=float,default=0.002,
                help ='?')

#Saving Info for UA
parser.add_argument('--save-iter-UA',required=False,type=int,default=20,
                help ='Save Interval')





omni_dir = 'C:\\Users\\Jansen Domoguen\\Desktop\\Proposal-Paper-Code\\Code\\MetaGAN-Propos\\omniglot-dataset'


class UA_Classifier():
    def __init__(self,args):
        self.batch_size =  args.batch_size_UA
        self.lamb = args.lamb_UA
        self.lr = args.lr_UA
        self.num_sampling = args.num_sampling_UA
        self.max_epoch = args.max_epoch_UA
        self.save_iter = args.save_iter_UA
        #self.loss = nn.BCEWithLogitsLoss()
        self.loss = nn.BCELoss()
        self.UA_model = UA_Model(args)
        if args.cuda:
            self.loss.cuda()
            self.UA_model.cuda()

        self.opt_UA = torch.optim.Adam(self.UA_model.parameters(),lr=self.lr,
            weight_decay=args.lamb_UA)

        # for (name,param) in self.UA_model.named_parameters():
        #     print(name,param.requires_grad)



    def run_epoch(self,data_loader,if_test=False):
        total_preds = []
        total_labels = []
        total_loss = []
        if if_test:
            #model = self.UA_model.eval()
            None
        else:
            #model = self.UA_model.train()
            None

        for (inputs,labels) in data_loader:
            inputs,labels = Variable(inputs).cuda(), Variable(labels).cuda()
            #outputs = model(inputs)
            outputs = self.UA_model(inputs)
            loss = self.loss(outputs,labels)
            if not if_test:
                self.opt_UA.zero_grad()
                loss.backward()
                self.opt_UA.step()

            total_preds.append(outputs.detach().cpu().numpy())
            total_labels.append(labels.detach().cpu().numpy())
            total_loss.append(loss.detach().cpu().numpy())
        # if not if_test:
        #     self.opt_UA.zero_grad()
        #     self.opt_UA.step()

        total_loss = np.mean(total_loss,axis=0)
        total_preds = np.concatenate(total_preds,axis=0)
        total_labels = np.concatenate(total_labels,axis=0)

        eval_preds = copy.deepcopy(total_preds)

        roc,auc = ROC_AUC(total_preds,total_labels)
        total_preds = total_preds >= 0.5
        acc = accuracy(total_preds,total_labels)
        if if_test:
            preds = eval_preds
        else:
            preds = total_preds
        return total_loss,auc,acc,preds



    def run(self,dataset):
        text = "test_output_text_NoDrpTest_0.25"
        f = open('output//'+text+'.txt','w')
        f = open('output//'+text+'.txt','a')
        train_loader = Tensor_Dataset_loader(dataset['train_x'],dataset['train_y'],
            self.batch_size)
        # valid_loader = Tensor_Dataset_loader(dataset["val_x"],dataset["val_y"],
        # batch_size=self.batch_size,shuffle=False)
        # test_loader = Tensor_Dataset_loader(dataset["eval_x"],dataset["eval_y"],
        # batch_size=self.batch_size,shuffle=False)
        valid_loader = [(dataset['val_x'],dataset['val_y'])]
        test_loader = [(dataset['eval_x'],dataset['eval_y'])]
        print('Run Model')
        for i_ in range(self.max_epoch):
            train_loss, train_auc, train_acc, _ = self.run_epoch(train_loader)
            print(" [*] Epoch: %d,      Train loss: %.4f,      Train AUC: %.4f,      Train ACC: %.4f"
            % (i_+1, train_loss, train_auc, train_acc))

            print('\n','\n','\n',file=f)
            print(" [*] Epoch: %d,      Train loss: %.4f,      Train AUC: %.4f,      Train ACC: %.4f"
            % (i_+1, train_loss, train_auc, train_acc),file=f)

            total_val_preds = []
            total_val_loss = []

            for sample in range(self.num_sampling):
                valid_loss, _, _, valid_preds = self.run_epoch(valid_loader,if_test=True)
                total_val_preds.append(valid_preds)
                total_val_loss.append(valid_loss)

            val_labels = dataset['val_y'].numpy()
            #val_labels = np.load("Physionet_dataset//physionet_dataset//1_val_y.npy")

            val_stacked_preds = np.reshape(np.concatenate(total_val_preds, 0), [self.num_sampling, dataset['val_x'].shape[0], dataset['val_y'].shape[1]])
            val_preds = np.mean(val_stacked_preds, axis=0)
            val_loss = np.mean(total_val_loss, axis=0)
            val_preds = val_preds >= 0.5
            roc, valid_auc = ROC_AUC(val_preds, val_labels)
            val_acc = accuracy(val_preds, val_labels)
            print(" [*] Epoch: %d, Validation loss: %.4f, Validation AUC: %.4f, Validation ACC: %.4f" % (i_+1, valid_loss, valid_auc, val_acc))
            print('\n','\n','\n',file=f)
            print(" [*] Epoch: %d, Validation loss: %.4f, Validation AUC: %.4f, Validation ACC: %.4f" % (i_+1, valid_loss, valid_auc, val_acc),file=f)


            total_eval_preds=[]
            total_eval_loss=[]
            for sample in range(self.num_sampling):
                eval_loss, eval_auc, eval_acc, eval_preds = self.run_epoch(test_loader,if_test=True)
                total_eval_preds.append(eval_preds)
                total_eval_loss.append(eval_loss)

            eval_labels = dataset['eval_y'].numpy()

            #eval_labels = np.load("Physionet_dataset//physionet_dataset//1_eval_y.npy")
            eval_stacked_preds = np.reshape(np.concatenate(total_eval_preds, 0), [self.num_sampling, dataset['eval_x'].shape[0], dataset['eval_y'].shape[1]])
            eval_preds = np.mean(eval_stacked_preds, axis=0)
            eval_loss = np.mean(total_eval_loss, axis=0)
            roc, eval_auc = ROC_AUC(eval_preds, eval_labels)
            eval_preds = eval_preds >=0.5
            eval_acc = accuracy(eval_preds, eval_labels)

            print(" [*] Epoch: %d, Evaluation loss: %.4f, Evaluation AUC: %.4f, Evaluation ACC: %.4f" % (i_+1, eval_loss, eval_auc, eval_acc))
            print("=======================================================================================")

            print('\n','\n','\n',file=f)
            print(" [*] Epoch: %d, Evaluation loss: %.4f, Evaluation AUC: %.4f, Evaluation ACC: %.4f" % (i_+1, eval_loss, eval_auc, eval_acc),file=f)
            print("=======================================================================================",file=f)
            if (i_+1)%self.save_iter == 0:
                print(10*"**"+"Saved"+10*"**")
                print('\n',file=f)
                print(10*"**"+"Saved"+10*"**",file=f)
                torch.save(self.UA_model.state_dict(),text+'_'+str(i_+1)+'_%.4f'%(eval_loss)+'_%.4f'%(eval_auc)+'.pt')
        f.close()



    # def fit(self,dataset,test='valid'):
    #     train_loader = Tensor_Dataset_loader(dataset['train_x'],dataset['train_y'],
    #         self.batch_size)
    #     if test=='valid':
    #         test_loader = Tensor_Dataset_loader(dataset["val_x"],dataset["val_y"],
    #         self.batch_size,shuffle=False)
    #     elif test == 'eval':
    #         test_loader = Tensor_Dataset_loader(dataset["eval_x"],dataset["eval_y"],
    #         self.batch_size,shuffle=False)
    #     for i_ in range(self.max_epoch):
    #         total_preds = []
    #         total_labels = []
    #         total_loss =  []
    #         for (inputs,labels) in train_loader:
    #             inputs,labels = Variable(inputs).cuda(), Variable(labels).cuda()
    #             self.opt_UA.zero_grad()
    #             outputs = self.UA_model(inputs)
    #             loss = self.loss(outputs,labels)
    #             loss.backward()
    #             self.optimizer.step()

    #             total_preds.append(outputs.numpy())
    #             total_labels.append(labels.numpy())
    #             total_loss.append(loss.numpy())

    #         train_loss = np.mean(total_loss,axis=0)
    #         total_preds = np.concatenate(total_preds,axis=0)
    #         total_labels = np.concatenate(total_labels,axis=0)

    #         train_roc,train_auc = ROC_AUC(total_preds,total_labels)
    #         total_preds = total_preds >= 0.5
    #         train_acc = accuracy(total_preds,total_labels)
    #         print(" [*] Epoch: %d,      Train loss: %.4f,      Train AUC: %.4f,      Train ACC: %.4f"
    #             % (i_+1, train_loss, train_auc, train_acc))

    #         self.predict(test_loader)

    # def predict(self,dataset):
    #     test_loader = Tensor_Dataset_loader(dataset["val_x"],dataset["val_y"],
    #         self.batch_size,shuffle=False)

    #     model = self.UA_model.eval()
    #     total_val_preds = []
    #     total_val_loss = []
    #     for sample in range(self.num_sampling):
    #         total_preds = []
    #         total_labels = []
    #         total_loss = []
    #         for (inputs,labels) in test_loader:
    #             inputs,labels = Variable(inputs).cuda(), Variable(labels).cuda()
    #             outputs = model(inputs)
    #             loss = self.loss(outputs,labels)

    #             total_preds.append(outputs.numpy())
    #             total_labels.append(labels.numpy())
    #             total_loss.append(loss.numpy())

    #         test_loss = np.mean(total_loss,axis=0)
    #         test_preds = np.concatenate(total_preds,axis=0)
    #         #test_labels = np.concatenate(total_labels,axis=0)
    #         #test_roc,test_auc = ROC_AUC(test_preds,test_labels)
    #         test_preds = test_preds >= 0.5
    #         #test_acc = accuracy(test_preds,test_labels)
    #         total_val_preds.append(test_preds)
    #         total_val_loss.append(total_loss)







def run(args,dataset):
    UA_model = UA_Classifier(args)
    UA_model.run(dataset)


def main(args):
    path = args.data_dir

    dataset = {}
    dataset["train_x"] = torch.from_numpy(np.load(path + 'train_x.npy')).float()
    dataset["train_y"] = torch.from_numpy(np.load(path + 'train_y.npy')).float()
    dataset["val_x"] = torch.from_numpy(np.load(path + 'val_x.npy')).float()
    dataset["val_y"] = torch.from_numpy(np.load(path + 'val_y.npy')).float()
    dataset["eval_x"] = torch.from_numpy(np.load(path + 'eval_x.npy')).float()
    dataset["eval_y"] = torch.from_numpy(np.load(path + 'eval_y.npy')).float()
    # if args.cuda:
    #     dataset["train_x"] = torch.from_numpy(np.load(path + 'train_x.npy')).float()
    #     dataset["train_y"] = torch.from_numpy(np.load(path + 'train_y.npy')).float()
    #     dataset["val_x"] = torch.from_numpy(np.load(path + 'val_x.npy')).float()
    #     dataset["val_y"] = torch.from_numpy(np.load(path + 'val_y.npy')).float()
    #     dataset["eval_x"] = torch.from_numpy(np.load(path + 'eval_x.npy')).float()
    #     dataset["eval_y"] = torch.from_numpy(np.load(path + 'eval_y.npy')).float()
    # else:
    #     dataset["train_x"] = torch.from_numpy(np.load(path + 'train_x.npy'))
    #     dataset["train_y"] = torch.from_numpy(np.load(path + 'train_y.npy'))
    #     dataset["val_x"] = torch.from_numpy(np.load(path + 'val_x.npy'))
    #     dataset["val_y"] = torch.from_numpy(np.load(path + 'val_y.npy'))
    #     dataset["eval_x"] = torch.from_numpy(np.load(path + 'eval_x.npy'))
    #     dataset["eval_y"] = torch.from_numpy(np.load(path + 'eval_y.npy'))

    run(args,dataset)














if __name__=='__main__':
    args = parser.parse_args()
    main(args)
