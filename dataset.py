import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms as T
import nibabel as nib
import h5py
import random
import torch.nn.functional as F

class create(data.Dataset):
    def __init__(self, data_root1, data_root2,transforms=None, train=True, test=False, synmodel=0):
        self.test = test
        # data_root = os.path.join(data_root, 'TrainingImg')
        # target_root = os.path.join(data_root, 'TrainingMask')

        self.synmodel = int(synmodel)
    
        datas1 = [os.path.join(data_root1, png) for png in os.listdir(data_root1)]
        datas2 = [os.path.join(data_root2, png) for png in os.listdir(data_root2)]

        datas1 = sorted(datas1, key=lambda x: (int(x.split('_')[-3].split('/')[-1]),
                                     int(x.split('_')[-1].split('.')[-2])) )
        datas2 = sorted(datas2, key=lambda x: (int(x.split('_')[-3].split('/')[-1]),
                                     int(x.split('_')[-1].split('.')[-2])) )

        
        self.train =train
        if self.train:
            self.datas1 = datas1[int(0.1* len(datas1)):int(0.8* len(datas1)):3]+datas1[int(1* len(datas1)):int(1* len(datas1)):3]
            self.datas2 = datas2[int(0.1* len(datas2)):int(0.8* len(datas2)):3]+datas2[int(1* len(datas2)):int(1* len(datas2)):3]


        else:
            self.datas1 = datas1[int(0.8* len(datas1)):int(1 * len(datas1)):3]+datas1[int(0* len(datas1)):int(0.1* len(datas1)):3]
            self.datas2 = datas2[int(0.8* len(datas2)):int(1* len(datas2)):3]+datas2[int(0* len(datas2)):int(0.1* len(datas2)):3]

        
        self.datas = self.datas1+self.datas2
        # self.datas = self.datas_.copy()

        self.maskgenerator = RandomMaskingGenerator(60,0.99)

        self.transforms1 = T.RandomRotation(90)

        # self.transforms2 = T.CenterCrop(128)

    def __getitem__(self, index):

        
        # data = self.normalize(np.expand_dims(self.data[:, :, index],0))
        # gt = np.expand_dims(self.target[:, :, index],0)

        f = h5py.File(self.datas[index],'r')
        # for key in f.keys():
        #     print(key)
        t1 =  np.expand_dims(f['t1'][:],0)
        t1ce =  np.expand_dims(f['t1ce'][:],0)
        t2 =  np.expand_dims(f['t2'][:],0)
        flair =  np.expand_dims(f['flair'][:],0)
        gt =  np.expand_dims(f['label'][:],0)

        t1_i = self.normalize(t1)
        t1ce_i = self.normalize(t1ce)
        t2_i = self.normalize(t2)
        flair_i = self.normalize(flair)

        # t1_o = self.normalize_(t1)
        # t1ce_o = self.normalize_(t1ce)
        # t2_o = self.normalize_(t2)
        # flair_o = self.normalize_(flair)

        input = np.concatenate((t1_i,t1ce_i,t2_i,flair_i),axis = 0)
        if self.synmodel == 0:
            output = self.normalize_(t1)
        elif self.synmodel == 1:
            output = self.normalize_(t1ce)
        elif self.synmodel == 2:
            output = self.normalize_(t2)   
        elif self.synmodel == 3:
            output = self.normalize_(flair)   


        if self.train:
            data = np.concatenate((input,output,gt),axis = 0)
            data = torch.from_numpy(data).type(torch.FloatTensor)
            data = self.transforms1(data)

            mask = self.maskgenerator()
            mask = torch.from_numpy(np.expand_dims(mask,0))
            mask = F.interpolate(mask.unsqueeze(0),scale_factor=4,mode='nearest').squeeze(0)
            data[self.synmodel,:] = data[self.synmodel,:]*mask

            input = data[:5,:,:]
            gt = data[5,:,:].unsqueeze(0) 
        else:
            input = torch.from_numpy(input).type(torch.FloatTensor)
            gt = torch.from_numpy(gt*1.0).type(torch.FloatTensor)  
        
        Mask1 = (gt==1).int()
        Mask2 = (gt==2).int()
        # Mask3 = (gt==3).int()
        Mask4 = (gt==4).int()

        label1 = Mask1.sum()>0 
        label2 = Mask2.sum()>0 
        # label3 = Mask3.sum()>0 
        label4 = Mask4.sum()>0 



        return input,label1,label2,label4,self.datas[index],Mask1+Mask4,Mask2+Mask1+Mask4

    def normalize(self, data, smooth=1e-9):
        mean = data.mean()

        std = data.std()
        if (mean == 0) or (std == 0):
            return data
        # mean = data[data!=0].mean()
        # std = data[data!=0].std()
        data = (data - mean + smooth) / (std + smooth)
        # data[data==data.min()] = -10  # data.min() are 0 before normalization
        return data

    def normalize_(self, data, smooth=1e-9):
        data = np.clip(data,0,np.percentile(data, 90))
        data = (data-np.min(data))/(np.max(data)-np.min(data)+smooth)
        data = 2*data-1
        return data


    def __len__(self):

        return len(self.datas)




class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)


    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        mask =mask.reshape((self.height, self.width))
        return mask 