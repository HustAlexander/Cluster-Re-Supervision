from config import opt
import torch
import dataset
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import weight_init,F_Net,C_Net_s
import numpy as np
from torch.utils.data import DataLoader
import os
from loss import CAMLoss,exclusLoss,CAMLoss_km
from torchvision import transforms as T
import random
from evaluation import *
from fast_pytorch_kmeans import KMeans
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

random.seed(309)
np.random.seed(309)
torch.manual_seed(309)
torch.cuda.manual_seed_all(309)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument("modal")
args = parser.parse_args()


model=F_Net(img_ch=4,feature_ch=64)
# state_dict = torch.load('./models/model_cl.ckpt')
# model.load_state_dict(state_dict)
#model = DeepLabV3()
#vgg_model = VGGNet(requires_grad=True)
#model=FCNs(pretrained_net=vgg_model, n_class=1)
# model.apply(weight_init)
# model.load_state_dict(torch.load('./models/AttU_Net.ckpt'))
model = model.cuda()

w = np.loadtxt('Label_cross.txt')
# w1 = (w[0,:]+w[3,:]>0).astype(int)
w1 = w[0,:].sum()/w.shape[1]
w2 = w[1,:].sum()/w.shape[1]
# w3 = w[2,:].sum()/w.shape[1]
# w4 = w[3,:].sum()/w.shape[1]

w1=torch.FloatTensor([w1,1-w1]).cuda()
w2=torch.FloatTensor([w2,1-w2]).cuda()
# w3=torch.FloatTensor([w3,1-w3]).cuda()
# w4=torch.FloatTensor([w4,1-w4]).cuda()

train_data = dataset.create(opt.data_root1, opt.data_root2, train=True,synmodel = args.modal)
val_data = dataset.create(opt.data_root1, opt.data_root2, train=False,synmodel = args.modal)
train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, 1)

lr = opt.lr
criterion1 = CAMLoss()
criterion3 = nn.L1Loss()
criterion4 = exclusLoss()
criterion5 = nn.CrossEntropyLoss(weight=w1)
criterion6 = nn.CrossEntropyLoss(weight=w2)
# criterion7 = nn.CrossEntropyLoss(weight=w3)
# criterion8 = nn.CrossEntropyLoss(weight=w4)
criterion9 = CAMLoss_km()
# optimizer = optim.RMSprop(model.parameters(), lr = opt.lr,alpha=0.9,weight_decay=opt.weight_decay)
optimizer=optim.Adam(model.parameters(),lr=opt.lr)

best_score = 0.3

flip = T.RandomVerticalFlip(p=1)



for epoch in range(opt.max_epoch):
    model.train()
    l=0
    l1=0
    l2=0
    l3=0
    total=0
    correct=0

    acc = 0.  # Accuracy
    TP0 = 0.  
    FP0 = 0.  
    FN0 = 0.  
    TN0 = 0. 
    length = 0
    cl_correct1=0
    cl_correct2=0
    cl_correct3=0
    cl_correct4=0

    for i, (data, label1, label2,label4,data_name,Mask1,Mask2) in enumerate(train_dataloader):
        input = Variable(data[:,[0,1,2,3],:].cuda())
        label1 = Variable(label1.cuda()).long()
        label2 = Variable(label2.cuda()).long()
        label4 = Variable(label4.cuda()).long()

        label = ((label1+label2+label4)>0).long()

        output = Variable(data[:,4,:].unsqueeze(1) ).cuda()

        # input_ = flip(input)
        # input = torch.cat((input,input_),0)

        syn,cl0,cl1,cam,feature = model(input)
        # bh = int(cl0.size(0)/2)
        syn = torch.tanh(syn)


        ###### feature clustering ########
        out_mask = output>-1
        out_mask = out_mask.float()
        out_mask = F.interpolate(out_mask,size=(feature.size(2),feature.size(3)))
        feature = out_mask*feature
        feature = feature.transpose(1,3).transpose(1,2)
        feature_ = feature.reshape(feature.shape[0]*feature.shape[1]*feature.shape[2],feature.shape[3])
        kmeans = KMeans(n_clusters=10, mode='euclidean')
        km = kmeans.fit_predict(feature_)

        km = km.reshape(feature.shape[0],feature.shape[1],feature.shape[2]).unsqueeze(1).float()
        km = F.interpolate(km,size=(cam.size(2),cam.size(3))).detach()

        loss4 = 0
        iter = km.max().int()
        for t in range(iter+1):
            feature_mask = km == t
            feature_mask = feature_mask.float()
            cam_ = cam*feature_mask
            cl_ = nn.functional.adaptive_avg_pool2d(cam_,(1,1))
            cl_ = cl_.view(-1, 2)
            _, predicted_ = torch.max(cl_, 1)
            predicted_n = predicted_*(1-label)
            if predicted_n.sum()>0:
                label_ = 0*label
                loss4 += criterion9(cam_,label_)+criterion6(cl_,label_)
            elif predicted_.sum()>0 and predicted_n.sum()==0:

                label_ = torch.ones_like(label)
                loss4 += criterion9(cam_,label_)+criterion6(cl_,label_)
        
        loss4 = loss4/iter


        loss1 = criterion1(cam,label)
        loss2 = criterion6(cl0,label)+criterion6(cl1,label)
        # loss3 = criterion3(cam[:bh,:],flip(cam[bh:,:]))
        loss5 = criterion3(syn,output)
        l1 +=loss1
        l2 +=loss2


        optimizer.zero_grad()
        (loss1+loss2+0.1*loss4+0.1*loss5).backward()
        optimizer.step()

        total+=label.size(0)

        _, predicted = torch.max(0.5*(cl0+cl1), 1)
        label = label>0
        cl_correct2+=(predicted==label).sum().item()


    Cl_Acc2=cl_correct2/total



    if (epoch + 1) % 20==0 and epoch + 1>=20:
        lr = lr*0.9
        print('reset learning rate to:', lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    print(
        'Epoch [%d/%d], Loss: %.4f, %.4f, %.4f, \n[Training]Cl_Acc2: %.4f, \n' % (
            epoch + 1, opt.max_epoch,l1,l2,l3,
            Cl_Acc2))

            


    model.eval()
    l = 0
    l1 =0
    l2 =0
    total=0
    correct=0
    cl_correct1=0
    cl_correct2=0
    cl_correct3=0
    cl_correct4=0
    
    acc_c = 0.  # Accuracy
    SE_c = 0.  # Sensitivity (Recall)
    SP_c = 0.  # Specificity
    PC_c = 0.  # Precision
    DC_c = 0.  # Dice Coefficient
    acc = 0.  # Accuracy
    SE = 0.  # Sensitivity (Recall)
    SP = 0.  # Specificity
    PC = 0.  # Precision
    DC = 0.  # Dice Coefficient

    id = 0
    s = 1
    p = 0

    with torch.no_grad():
        for i, (data, label1, label2,label4,data_name,Mask1,Mask2) in enumerate(val_dataloader):
            input = Variable(data[:,[0,1,2,3],:].cuda())
            GT_1 = Variable(Mask1.cuda())
            GT_2 = Variable(Mask2.cuda())
            label1 = Variable(label1.cuda()).long()
            label2 = Variable(label2.cuda()).long()
            label4 = Variable(label4.cuda()).long()


            label = ((label1+label2+label4)>0).long()

            
            syn,cl0,cl1,cam,feature = model(input)

            
            loss1 = criterion1(cam,label)
            loss2 = criterion6(cl0,label)+criterion6(cl1,label)
            # loss1 = 0
              

            l1 +=loss1
            l2 +=loss2
            
            total+=label.size(0)

            #if only one label of a picture in a batch is 1,what will happen

            _, predicted = torch.max(0.5*(cl0+cl1), 1)
            label = label>0
            cl_correct2+=(predicted==label).sum().item()

            SR2_c = cam[:,1,:,:].unsqueeze(1)
            SR2_c = SR2_c>0.5



            if data_name[0].split('_')[-3].split('/')[-1] != id:

                if p > 0:
                    if torch.sum(GT2) != 0:

                        acc += get_accuracy(SR2, GT2)
                        SE += get_sensitivity(SR2, GT2)
                        SP += get_specificity(SR2, GT2)
                        DC += get_DC(SR2, GT2)


                id = data_name[0].split('_')[-3].split('/')[-1]
                SR2 = SR2_c
                GT2 = GT_2
                s = 1
                p += 1
            else:
                SR2 = torch.cat((SR2,SR2_c),1)
                GT2 = torch.cat((GT2,GT_2),1)
                s += 1

        ###last scan###  
        if torch.sum(GT2) != 0:

            acc += get_accuracy(SR2, GT2)
            SE += get_sensitivity(SR2, GT2)
            SP += get_specificity(SR2, GT2)
            DC += get_DC(SR2, GT2)
        
        
        Cl_Acc2=cl_correct2/total
        
        score =  DC/ p

        
        
        print(
            '[val] Loss: %.4f, %.4f, \n Cl_Acc2: %.4f\n' % (
                 l1,l2,
                 Cl_Acc2))

        print(
            '[Seg]  Acc: %.4f, SE: %.4f, SP: %.4f, DC: %.4f\n'  % (
               acc/ p,  SE/ p, SP/ p,DC/ p))

        # Save Best model
        if score > best_score:
            best_score = score
            best_Net = model.state_dict()
            print('Best model score : %.4f \n' % (best_score))
            torch.save(best_Net, './models/model_mask_'+args.modal+'.ckpt')













