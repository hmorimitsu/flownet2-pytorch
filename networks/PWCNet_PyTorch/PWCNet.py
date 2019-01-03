import torch
import torch.nn as nn
import numpy as np

from networks.correlation_package.correlation import Correlation

class PWCNet(nn.Module):
    """
    PWC net without dilation convolution

    """
    def __init__(self, args, batchNorm=False, div_flow=20):
        super(PWCNet,self).__init__()
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.rgb_max = args.rgb_max
        self.args = args

        self.conv1a  = self.conv(batchNorm, 3,   16, kernel_size=3, stride=2)
        self.conv1aa = self.conv(batchNorm, 16,  16, kernel_size=3, stride=1)
        self.conv1b  = self.conv(batchNorm, 16,  16, kernel_size=3, stride=1)
        self.conv2a  = self.conv(batchNorm, 16,  32, kernel_size=3, stride=2)
        self.conv2aa = self.conv(batchNorm, 32,  32, kernel_size=3, stride=1)
        self.conv2b  = self.conv(batchNorm, 32,  32, kernel_size=3, stride=1)
        self.conv3a  = self.conv(batchNorm, 32,  64, kernel_size=3, stride=2)
        self.conv3aa = self.conv(batchNorm, 64,  64, kernel_size=3, stride=1)
        self.conv3b  = self.conv(batchNorm, 64,  64, kernel_size=3, stride=1)
        self.conv4a  = self.conv(batchNorm, 64,  96, kernel_size=3, stride=2)
        self.conv4aa = self.conv(batchNorm, 96,  96, kernel_size=3, stride=1)
        self.conv4b  = self.conv(batchNorm, 96,  96, kernel_size=3, stride=1)
        self.conv5a  = self.conv(batchNorm, 96, 128, kernel_size=3, stride=2)
        self.conv5aa = self.conv(batchNorm, 128,128, kernel_size=3, stride=1)
        self.conv5b  = self.conv(batchNorm, 128,128, kernel_size=3, stride=1)
        self.conv6aa = self.conv(batchNorm, 128,196, kernel_size=3, stride=2)
        self.conv6a  = self.conv(batchNorm, 196,196, kernel_size=3, stride=1)
        self.conv6b  = self.conv(batchNorm, 196,196, kernel_size=3, stride=1)

        self.corr    = Correlation(pad_size=args.pwcnet_md, kernel_size=1, max_displacement=args.pwcnet_md, stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU = nn.LeakyReLU(0.1)
        
        nd = (2*args.pwcnet_md+1)**2
        dd = np.cumsum([128,128,96,64,32])

        od = nd
        self.conv6_0 = self.conv(batchNorm, od,      128, kernel_size=3, stride=1)
        self.conv6_1 = self.conv(batchNorm, od+dd[0],128, kernel_size=3, stride=1)
        self.conv6_2 = self.conv(batchNorm, od+dd[1],96,  kernel_size=3, stride=1)
        self.conv6_3 = self.conv(batchNorm, od+dd[2],64,  kernel_size=3, stride=1)
        self.conv6_4 = self.conv(batchNorm, od+dd[3],32,  kernel_size=3, stride=1)        
        self.predict_flow6 = self.predict_flow(od+dd[4])
        self.deconv6 = self.deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat6 = self.deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+128+4
        self.conv5_0 = self.conv(batchNorm, od,      128, kernel_size=3, stride=1)
        self.conv5_1 = self.conv(batchNorm, od+dd[0],128, kernel_size=3, stride=1)
        self.conv5_2 = self.conv(batchNorm, od+dd[1],96,  kernel_size=3, stride=1)
        self.conv5_3 = self.conv(batchNorm, od+dd[2],64,  kernel_size=3, stride=1)
        self.conv5_4 = self.conv(batchNorm, od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow5 = self.predict_flow(od+dd[4]) 
        self.deconv5 = self.deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat5 = self.deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+96+4
        self.conv4_0 = self.conv(batchNorm, od,      128, kernel_size=3, stride=1)
        self.conv4_1 = self.conv(batchNorm, od+dd[0],128, kernel_size=3, stride=1)
        self.conv4_2 = self.conv(batchNorm, od+dd[1],96,  kernel_size=3, stride=1)
        self.conv4_3 = self.conv(batchNorm, od+dd[2],64,  kernel_size=3, stride=1)
        self.conv4_4 = self.conv(batchNorm, od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow4 = self.predict_flow(od+dd[4]) 
        self.deconv4 = self.deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat4 = self.deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+64+4
        self.conv3_0 = self.conv(batchNorm, od,      128, kernel_size=3, stride=1)
        self.conv3_1 = self.conv(batchNorm, od+dd[0],128, kernel_size=3, stride=1)
        self.conv3_2 = self.conv(batchNorm, od+dd[1],96,  kernel_size=3, stride=1)
        self.conv3_3 = self.conv(batchNorm, od+dd[2],64,  kernel_size=3, stride=1)
        self.conv3_4 = self.conv(batchNorm, od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow3 = self.predict_flow(od+dd[4]) 
        self.deconv3 = self.deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat3 = self.deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+32+4
        self.conv2_0 = self.conv(batchNorm, od,      128, kernel_size=3, stride=1)
        self.conv2_1 = self.conv(batchNorm, od+dd[0],128, kernel_size=3, stride=1)
        self.conv2_2 = self.conv(batchNorm, od+dd[1],96,  kernel_size=3, stride=1)
        self.conv2_3 = self.conv(batchNorm, od+dd[2],64,  kernel_size=3, stride=1)
        self.conv2_4 = self.conv(batchNorm, od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow2 = self.predict_flow(od+dd[4])

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

    def conv(self, batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
        if batchNorm:
            return nn.Sequential(
                    nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                                padding=padding, dilation=dilation, bias=False),
                    nn.BatchNorm2d(out_planes),
                    nn.LeakyReLU(0.1))
        else:
            return nn.Sequential(
                    nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                                padding=padding, dilation=dilation, bias=True),
                    nn.LeakyReLU(0.1))

    def deconv(self, in_planes, out_planes, kernel_size=4, stride=2, padding=1):
        return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

    def predict_flow(self, in_planes):
        return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = torch.autograd.Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid = torch.stack([
            2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0,
            2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0], dim=1)

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size()))
        if x.is_cuda:
            mask = mask.cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        # if W==128:
            # np.save('mask.npy', mask.cpu().data.numpy())
            # np.save('warp.npy', output.cpu().data.numpy())
        
        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask


    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        
        x = (inputs - rgb_mean) / self.rgb_max
        im1 = x[:,:,0,:,:]
        im2 = x[:,:,1,:,:]
        
        c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))
        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c24 = self.conv4b(self.conv4aa(self.conv4a(c23)))
        c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        c25 = self.conv5b(self.conv5aa(self.conv5a(c24)))
        c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))
        c26 = self.conv6b(self.conv6a(self.conv6aa(c25)))


        corr6 = self.corr(c16, c26) 
        corr6 = self.leakyRELU(corr6)   


        x = torch.cat((self.conv6_0(corr6), corr6),1)
        x = torch.cat((self.conv6_1(x), x),1)
        x = torch.cat((self.conv6_2(x), x),1)
        x = torch.cat((self.conv6_3(x), x),1)
        x = torch.cat((self.conv6_4(x), x),1)
        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)

        
        warp5 = self.warp(c25, up_flow6*0.625)
        corr5 = self.corr(c15, warp5) 
        corr5 = self.leakyRELU(corr5)
        x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)
        x = torch.cat((self.conv5_0(x), x),1)
        x = torch.cat((self.conv5_1(x), x),1)
        x = torch.cat((self.conv5_2(x), x),1)
        x = torch.cat((self.conv5_3(x), x),1)
        x = torch.cat((self.conv5_4(x), x),1)
        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)

       
        warp4 = self.warp(c24, up_flow5*1.25)
        corr4 = self.corr(c14, warp4)  
        corr4 = self.leakyRELU(corr4)
        x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)
        x = torch.cat((self.conv4_0(x), x),1)
        x = torch.cat((self.conv4_1(x), x),1)
        x = torch.cat((self.conv4_2(x), x),1)
        x = torch.cat((self.conv4_3(x), x),1)
        x = torch.cat((self.conv4_4(x), x),1)
        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)


        warp3 = self.warp(c23, up_flow4*2.5)
        corr3 = self.corr(c13, warp3) 
        corr3 = self.leakyRELU(corr3)
        

        x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
        x = torch.cat((self.conv3_0(x), x),1)
        x = torch.cat((self.conv3_1(x), x),1)
        x = torch.cat((self.conv3_2(x), x),1)
        x = torch.cat((self.conv3_3(x), x),1)
        x = torch.cat((self.conv3_4(x), x),1)
        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)


        warp2 = self.warp(c22, up_flow3*5.0) 
        corr2 = self.corr(c12, warp2)
        corr2 = self.leakyRELU(corr2)
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x),1)
        x = torch.cat((self.conv2_1(x), x),1)
        x = torch.cat((self.conv2_2(x), x),1)
        x = torch.cat((self.conv2_3(x), x),1)
        x = torch.cat((self.conv2_4(x), x),1)
        flow2 = self.predict_flow2(x)
        
        if self.training:
            return flow2,flow3,flow4,flow5,flow6
        else:
            return self.upsample1(flow2*self.div_flow)


class PWCDCNet(PWCNet):
    """
    PWC-DC net. add dilation convolution and densenet connections

    """
    def __init__(self, args, batchNorm=False, div_flow=20):
        super(PWCDCNet,self).__init__(args, batchNorm, div_flow)
        nd = (2*args.pwcnet_md+1)**2
        dd = np.cumsum([128,128,96,64,32])
        od = nd+32+4
        self.dc_conv1 = self.conv(batchNorm, od+dd[4], 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv2 = self.conv(batchNorm, 128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc_conv3 = self.conv(batchNorm, 128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc_conv4 = self.conv(batchNorm, 128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc_conv5 = self.conv(batchNorm, 96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = self.conv(batchNorm, 64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv7 = self.predict_flow(32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')


    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        
        x = (inputs - rgb_mean) / self.rgb_max
        im1 = x[:,:,0,:,:]
        im2 = x[:,:,1,:,:]
        
        c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))
        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c24 = self.conv4b(self.conv4aa(self.conv4a(c23)))
        c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        c25 = self.conv5b(self.conv5aa(self.conv5a(c24)))
        c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))
        c26 = self.conv6b(self.conv6a(self.conv6aa(c25)))


        corr6 = self.corr(c16, c26) 
        corr6 = self.leakyRELU(corr6)   


        x = torch.cat((self.conv6_0(corr6), corr6),1)
        x = torch.cat((self.conv6_1(x), x),1)
        x = torch.cat((self.conv6_2(x), x),1)
        x = torch.cat((self.conv6_3(x), x),1)
        x = torch.cat((self.conv6_4(x), x),1)
        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)

        
        warp5 = self.warp(c25, up_flow6*0.625)
        corr5 = self.corr(c15, warp5) 
        corr5 = self.leakyRELU(corr5)
        x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)
        x = torch.cat((self.conv5_0(x), x),1)
        x = torch.cat((self.conv5_1(x), x),1)
        x = torch.cat((self.conv5_2(x), x),1)
        x = torch.cat((self.conv5_3(x), x),1)
        x = torch.cat((self.conv5_4(x), x),1)
        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)

       
        warp4 = self.warp(c24, up_flow5*1.25)
        corr4 = self.corr(c14, warp4)  
        corr4 = self.leakyRELU(corr4)
        x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)
        x = torch.cat((self.conv4_0(x), x),1)
        x = torch.cat((self.conv4_1(x), x),1)
        x = torch.cat((self.conv4_2(x), x),1)
        x = torch.cat((self.conv4_3(x), x),1)
        x = torch.cat((self.conv4_4(x), x),1)
        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)


        warp3 = self.warp(c23, up_flow4*2.5)
        corr3 = self.corr(c13, warp3) 
        corr3 = self.leakyRELU(corr3)
        

        x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
        x = torch.cat((self.conv3_0(x), x),1)
        x = torch.cat((self.conv3_1(x), x),1)
        x = torch.cat((self.conv3_2(x), x),1)
        x = torch.cat((self.conv3_3(x), x),1)
        x = torch.cat((self.conv3_4(x), x),1)
        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)


        warp2 = self.warp(c22, up_flow3*5.0) 
        corr2 = self.corr(c12, warp2)
        corr2 = self.leakyRELU(corr2)
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x),1)
        x = torch.cat((self.conv2_1(x), x),1)
        x = torch.cat((self.conv2_2(x), x),1)
        x = torch.cat((self.conv2_3(x), x),1)
        x = torch.cat((self.conv2_4(x), x),1)
        flow2 = self.predict_flow2(x)
 
        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 += self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))
        
        if self.training:
            return flow2,flow3,flow4,flow5,flow6
        else:
            return self.upsample1(flow2*self.div_flow)
