import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch

class conv_block_f(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block_f,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv_f(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv_f,self).__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
            nn.ConvTranspose2d(ch_in,ch_in,kernel_size=3,stride=2,padding=1, output_padding=1),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.LeakyReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
            # nn.ConvTranspose2d(ch_in,ch_in,kernel_size=3,stride=2,padding=1, output_padding=1),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class C_Net_s(nn.Module):
    def __init__(self, img_ch=1,feature_ch=16, output_ch=1):
        super(C_Net_s, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(img_ch, feature_ch)
        self.Conv2 = conv_block(feature_ch, feature_ch*2)
        self.Conv3 = conv_block(feature_ch*2, feature_ch*4)
        self.Conv4 = conv_block(feature_ch*4, feature_ch*8)
        self.Conv5 = conv_block(feature_ch*8, feature_ch*16)


        self.bott32 = nn.Conv2d(feature_ch*4, 2,kernel_size  = 1,bias = False)
        self.bott52 = nn.Conv2d(feature_ch*16, 2,kernel_size  = 1,bias = False)


        
        self.relu = nn.ReLU(True) 
        
    def CAM_G(self,x3,x5,bott3,bott5):


        cam0 = bott3(x3)
        cl0 = nn.functional.adaptive_avg_pool2d(cam0,(1,1))
        cl0 = cl0.view(-1, 2)

        
        cam0 = F.upsample(cam0, size=(120, 120), mode='bilinear')
        # cam0 = torch.sigmoid(cam0)

        B, C, H, W = cam0.shape
        cam0 = cam0.view(B, -1)
        cam0 = cam0-cam0.min(dim=1, keepdim=True)[0]
        cam0 = cam0/(cam0.max(dim=1, keepdim=True)[0] + 1e-9)
        cam0 = cam0.view(B, C, H, W)

        cam1 = bott5(x5)
        cl1 = nn.functional.adaptive_avg_pool2d(cam1,(1,1))
        cl1 = cl1.view(-1, 2)

        
        cam1 = F.upsample(cam1, size=(120, 120), mode='bilinear')
        # cam1 = torch.sigmoid(cam1)

        B, C, H, W = cam1.shape
        cam1 = cam1.view(B, -1)
        cam1 = cam1-cam1.min(dim=1, keepdim=True)[0]
        cam1 = cam1/(cam1.max(dim=1, keepdim=True)[0] + 1e-9)
        cam1 = cam1.view(B, C, H, W)
        
        cam = cam0*cam1

        cam = F.upsample(cam, size=(240, 240), mode='bilinear')


        return cl1,cl0, cam        
        

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)       
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)                
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)        
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)        
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)   



        cl02,cl12,cam2 = self.CAM_G(x3,x5,self.bott32,self.bott52)

        

        #classification

        return cl02,cl12,cam2    


class U_Net(nn.Module):
    def __init__(self, img_ch=4,feature_ch=64, output_ch=1):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 =  conv_block(img_ch, feature_ch)
        self.Conv2 =  conv_block(feature_ch, feature_ch*2)
        self.Conv3 =  conv_block(feature_ch*2, feature_ch*4)
        self.Conv4 =  conv_block(feature_ch*4, feature_ch*8)
        self.Conv5 = conv_block(feature_ch*8, feature_ch*16)
        


        self.Up5 = up_conv(ch_in=feature_ch*16, ch_out=feature_ch*8)
        self.Up_conv5 = conv_block(ch_in=feature_ch*16, ch_out=feature_ch*8)

        self.Up4 = up_conv(ch_in=feature_ch*8, ch_out=feature_ch*4)
        self.Up_conv4 = conv_block(ch_in=feature_ch*8, ch_out=feature_ch*4)

        self.Up3 = up_conv(ch_in=feature_ch*4, ch_out=feature_ch*2)
        self.Up_conv3 = conv_block(ch_in=feature_ch*4, ch_out=feature_ch*2)

        self.Up2 = up_conv(ch_in=feature_ch*2, ch_out=feature_ch)
        self.Up_conv2 = conv_block(ch_in=feature_ch*2, ch_out=feature_ch)
        
        self.Up1 = up_conv(ch_in=feature_ch, ch_out=feature_ch)
        
        self.Conv_1x1_d = nn.Conv2d(feature_ch, output_ch, kernel_size=1, stride=1, padding=0)

        
    def forward(self, x):


        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)   

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)  

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5) 


        # x5 = self.conv_d5(x5)
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        # x3 = self.conv_d3(x3)
        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        

        d1 = self.Conv_1x1_d(d2)


        return d1





class F_Net(nn.Module):
    def __init__(self, img_ch=4,feature_ch=64, output_ch=1):
        super(F_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 =  conv_block_f(img_ch, feature_ch)
        self.Conv2 =  conv_block_f(feature_ch, feature_ch*2)
        self.Conv3 =  conv_block_f(feature_ch*2, feature_ch*4)
        self.Conv4 =  conv_block_f(feature_ch*4, feature_ch*8)
        self.Conv5 = conv_block_f(feature_ch*8, feature_ch*16)
        
        self.Up5 = up_conv_f(ch_in=feature_ch*16, ch_out=feature_ch*8)
        self.Up_conv5 = conv_block_f(ch_in=feature_ch*16, ch_out=feature_ch*8)

        self.Up4 = up_conv_f(ch_in=feature_ch*8, ch_out=feature_ch*4)
        self.Up_conv4 = conv_block_f(ch_in=feature_ch*8, ch_out=feature_ch*4)

        self.Up3 = up_conv_f(ch_in=feature_ch*4, ch_out=feature_ch*2)
        self.Up_conv3 = conv_block_f(ch_in=feature_ch*4, ch_out=feature_ch*2)

        self.Up2 = up_conv_f(ch_in=feature_ch*2, ch_out=feature_ch)
        self.Up_conv2 = conv_block_f(ch_in=feature_ch*2, ch_out=feature_ch)

        
        self.Conv_1x1_d = nn.Conv2d(feature_ch, output_ch, kernel_size=1, stride=1, padding=0)

        self.conv_d3 = nn.Conv2d(feature_ch*4, feature_ch*4,kernel_size  = 3,stride=1,padding=1)
        self.conv_d5 = nn.Conv2d(feature_ch*16, feature_ch*16,kernel_size  = 3,stride=1,padding=1)

        # self.conv_cam3 = nn.Conv2d(feature_ch*4, feature_ch*4,kernel_size  = 7,stride=1,padding=3)
        # self.conv_cam5 = nn.Conv2d(feature_ch*16, feature_ch*16,kernel_size  = 7,stride=1,padding=3)
        self.bott32 = nn.Conv2d(feature_ch*4, 2,kernel_size = 1, bias = False)
        self.bott52 = nn.Conv2d(feature_ch*16, 2,kernel_size = 1, bias = False)



    # def CAM_G(self,x3,x5,bott3,bott5,conv3,conv5):
    def CAM_G(self,x3,x5,bott3,bott5):

        cam0 = bott3(x3)
        cl0 = nn.functional.adaptive_avg_pool2d(cam0,(1,1))
        cl0 = cl0.view(-1, 2)

        
        cam0 = F.upsample(cam0, size=(120, 120), mode='bilinear')
        # cam0 = torch.sigmoid(cam0)

        B, C, H, W = cam0.shape
        cam0 = cam0.view(B, -1)
        cam0 = cam0-cam0.min(dim=1, keepdim=True)[0]
        cam0 = cam0/(cam0.max(dim=1, keepdim=True)[0] + 1e-9)
        cam0 = cam0.view(B, C, H, W)

        cam1 = bott5(x5)
        cl1 = nn.functional.adaptive_avg_pool2d(cam1,(1,1))
        cl1 = cl1.view(-1, 2)

        
        cam1 = F.upsample(cam1, size=(120, 120), mode='bilinear')
        # cam1 = torch.sigmoid(cam1)

        B, C, H, W = cam1.shape
        cam1 = cam1.view(B, -1)
        cam1 = cam1-cam1.min(dim=1, keepdim=True)[0]
        cam1 = cam1/(cam1.max(dim=1, keepdim=True)[0] + 1e-9)
        cam1 = cam1.view(B, C, H, W)
        
        cam = cam0*cam1

        cam = F.upsample(cam, size=(240, 240), mode='bilinear')


        return cl1,cl0, cam   
        
        

    def forward(self, x):


        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)   

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)  

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5) 


        cl0,cl1,cam = self.CAM_G(x3,x5,self.bott32,self.bott52)


        # x5 = self.conv_d5(x5)
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        # x3 = self.conv_d3(x3)
        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        

        d1 = self.Conv_1x1_d(d2)


        return d1,cl0,cl1,cam,d2




def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


