import os
import random
import numpy as np
import torch
from torch.utils import data
from copy import deepcopy
from PIL import Image
import sys
from torch.utils.data import DataLoader,TensorDataset
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms
import random


class OmniglotTask(object):
    '''
    Sample a few-shot learning task from the Omniglot dataset
    Sample N-way k-shot train and val sets according to
     - split (dataset/meta level train or test)
     - N-way classification (sample this many chars)
     - k-shot (sample this many examples from each char class)
    Assuming that the validation set is the same size as the train set!
    '''

    def __init__(self, root, num_cls, num_inst, split='train'):
        self.dataset = 'omniglot'
        self.root = '{}/images_background'.format(root) if split == 'train' else '{}/images_evaluation'.format(root)
        #print(self.root,'root')
        self.num_cl = num_cls
        self.num_inst = num_inst
        # Sample num_cls characters and num_inst instances of each
        languages = os.listdir(self.root)
        chars = []
        for l in languages:
            chars += [os.path.join(l, x) for x in os.listdir(os.path.join(self.root, l))]
        random.shuffle(chars)
        classes = chars[:num_cls]
        labels = np.array(range(len(classes)))
        labels = dict(zip(classes, labels))
        #print(labels.keys().split('/')[:-1])
        instances = dict()
        # Now sample from the chosen classes to create class-balanced train and val sets
        self.train_ids = []
        self.val_ids = []
        for c in classes:
            # First get all isntances of that class
            temp = [os.path.join(c, x) for x in os.listdir(os.path.join(self.root, c))]
            instances[c] = random.sample(temp, len(temp))

            # Sample num_inst instances randomly each for train and val
            self.train_ids += instances[c][:num_inst]
            self.val_ids += instances[c][num_inst:num_inst*2]
            #print(self.train_ids[0].split("\\")[:-1],os.path.join(*self.train_ids[0].split("\\")[:-1]),"****"*3,labels[os.path.join(*self.train_ids[0].split("\\")[:-1])])
        # Keep instances separated by class for class-balanced mini-batches
        self.train_labels = [labels[self.get_class(x)] for x in self.train_ids]
        self.val_labels = [labels[self.get_class(x)] for x in self.val_ids]


    def get_class(self, instance):
        return os.path.join(*instance.split('/')[:-1])


class FewShotDataset(data.Dataset):
    """
    Load image-label pairs from a task to pass to Torch DataLoader
    Tasks consist of data and labels split into train / val splits
    """

    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.transform = transform # Torch operations on the input image
        self.target_transform = target_transform
        self.task = task
        self.root = self.task.root
        self.split = split
        self.img_ids = self.task.train_ids if self.split == 'train' else self.task.val_ids
        self.labels = self.task.train_labels if self.split == 'train' else self.task.val_labels

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")

class Omniglot(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(Omniglot, self).__init__(*args, **kwargs)

    def load_image(self, idx):
        ''' Load image '''
        im = Image.open('{}/{}'.format(self.root, idx)).convert('RGB')
        im = im.resize((28,28), resample=Image.LANCZOS) # per Chelsea's implementation
        im = np.array(im, dtype=np.float32)
        return im

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        im = self.load_image(img_id)
        if self.transform is not None:
            im = self.transform(im)
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return im, label


class ClassBalancedSampler(Sampler):
    '''
    Samples class-balanced batches from 'num_cl' pools each
    of size 'num_inst'
    If 'batch_cutoff' is None, indices for iterating over batches
    of the entire dataset will be returned
    Otherwise, indices for the number of batches up to the batch_cutoff
    will be returned
    (This is to allow sampling with replacement across training iterations)
    '''

    def __init__(self, num_cl, num_inst, batch_cutoff=None):
       self.num_cl = num_cl
       self.num_inst = num_inst
       self.batch_cutoff = batch_cutoff

    def __iter__(self):
       '''return a single list of indices, assuming that items will be grouped by class '''
       # First construct batches of 1 instance per class
       batches = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)] for j in range(self.num_cl)]
       batches = [[batches[j][i] for j in range(self.num_cl)] for i in range(self.num_inst)]
       # Shuffle within each batch so that classes don't always appear in same order
       for sublist in batches:
           random.shuffle(sublist)

       if self.batch_cutoff is not None:
           random.shuffle(batches)
           batches = batches[:self.batch_cutoff]

       batches = [item for sublist in batches for item in sublist]

       return iter(batches)

    def __len__(self):
       return 1



def get_data_loader(task, batch_size=1, split='train'):
    # NOTE: batch size here is # instances PER CLASS
    if task.dataset == 'mnist':
        normalize = transforms.Normalize(mean=[0.13066, 0.13066, 0.13066], std=[0.30131, 0.30131, 0.30131])
        dset = MNIST(task, transform=transforms.Compose([transforms.ToTensor(), normalize]), split=split)
    else:
        normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
        dset = Omniglot(task, transform=transforms.Compose([transforms.ToTensor(), normalize]), split=split)
    sampler = ClassBalancedSampler(task.num_cl, task.num_inst, batch_cutoff = (None if split != 'train' else batch_size))
    loader = DataLoader(dset, batch_size=batch_size*task.num_cl, sampler=sampler, num_workers=0, pin_memory=False)
    return loader


def Tensor_Dataset_loader(X_tensor,Y_tensor,batch_size,shuffle=True):
    my_dataset = TensorDataset(X_tensor,Y_tensor)
    return DataLoader(my_dataset,batch_size=batch_size,num_workers=0,shuffle=shuffle)



