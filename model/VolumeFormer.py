import torch
import math
from torch import nn
from einops.layers.torch import Rearrange

class PixelShuffle3d(torch.nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    '''
    def __init__(self, scale_d=8, scale_h=8, scale_w=8):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale_d = scale_d
        self.scale_h=scale_h
        self.scale_w=scale_w

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // (self.scale_d*self.scale_h*self.scale_w)

        out_depth = in_depth * self.scale_d
        out_height = in_height * self.scale_h
        out_width = in_width * self.scale_w

        input_view = input.contiguous().view(batch_size, nOut, self.scale_d, self.scale_h, self.scale_w, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)


# class MixerBlock(torch.nn.Module):

#     def __init__(self, in_channel, in_size, scale=2, patch_size=[4,6,6]):
#         super(MixerBlock,self).__init__()
#         # self.out_size=[in_size[0]//scale, in_size[1]//scale, in_size[2]//scale]
#         self.num_token=in_size[0]//patch_size[0]*in_size[1]//patch_size[1]*in_size[2]//patch_size[2]
#         self.token_dim=in_channel//16*(patch_size[0]//scale*patch_size[1]//scale*patch_size[2]//scale)
        
#         self.embedding = torch.nn.Conv3d(in_channel, self.token_dim, patch_size, patch_size)
#         self.token_mix = torch.nn.Sequential(
#             Rearrange('b c d h w -> b c (d h w)'),
#             torch.nn.LayerNorm(self.num_token), 
#             torch.nn.Linear(self.num_token, self.num_token//2),
#             torch.nn.GELU(),
#             torch.nn.Linear(self.num_token//2, self.num_token),
#             Rearrange('b c (d h w) -> b c d h w', d=in_size[0]//patch_size[0],h=in_size[1]//patch_size[1],w=in_size[2]//patch_size[2])
#         )
#         self.pixelshuffle=PixelShuffle3d(patch_size[0]//scale,patch_size[1]//scale,patch_size[2]//scale)
#         self.channel_mix = torch.nn.Sequential(
#             Rearrange('b c d h w -> b (d h w) c'),
#             torch.nn.LayerNorm(self.token_dim//(patch_size[0]//scale*patch_size[1]//scale*patch_size[2]//scale)),
#             torch.nn.Linear(self.token_dim//(patch_size[0]//scale*patch_size[1]//scale*patch_size[2]//scale), in_channel),
#             Rearrange('b (d h w) c -> b c d h w',d=in_size[0]//scale, h=in_size[1]//scale, w=in_size[2]//scale)
#         )
        
#         # self.beta=torch.Tensor([1],requires_grad=True)

#     def forward(self, x):
#         ebd=self.embedding(x)
#         tkm=self.token_mix(ebd)
#         rev_tkm=self.pixelshuffle(tkm)
#         return self.channel_mix(rev_tkm)


# class NonLocalPooling(torch.nn.Module):
#     def __init__(self, in_channel, in_size, scale=2, patch_size=[4,6,6]):
#         super().__init__() 
#         self.mlp=MixerBlock(in_channel, in_size, scale, patch_size)
#         self.max_pool=torch.nn.MaxPool3d(scale)
#         self.alpha=torch.tensor(1.0,requires_grad=True).cuda()
       
#     def forward(self,x):
#         maxpool=self.max_pool(x)
#         nonlocalpool=self.mlp(x)
#         return maxpool*(self.alpha).expand_as(maxpool)+nonlocalpool*(1-self.alpha).expand_as(nonlocalpool)

# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=8):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool3d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)

# class AutoDecomposition_noalter(torch.nn.Module):
#     def __init__(self,in_ch=1,inter_ch=0,out_ch=32,kernel=2,scale=2):
#         super(AutoDecomposition_noalter,self).__init__()
#         self.down=torch.nn.Conv3d(in_ch,in_ch*scale**3,kernel,scale,0)
#     def forward(self,x):
#         return self.down(x),x

class AutoDecomposition(torch.nn.Module):
    def __init__(self,in_ch=1,inter_ch=None,out_ch=None,kernel=2,scale=2):
        super(AutoDecomposition,self).__init__()
        if out_ch is None:
            if inter_ch is None:
                out_ch=in_ch*scale**3
            else:
                out_ch=inter_ch*scale**3

        self.inter_ch=inter_ch
        if self.inter_ch is None:
            self.down=torch.nn.Conv3d(in_ch,out_ch,kernel,scale,0)
        else:
            assert inter_ch>in_ch, "if enabled, inter_ch must be larger than in_ch"
            self.alter=torch.nn.Conv3d(in_ch,inter_ch-in_ch,3,1,1)
            self.down=torch.nn.Conv3d(inter_ch,out_ch,kernel,scale,0)

    def forward(self,x):
        if self.inter_ch is None:
            return self.down(x),x
        else:
            x=torch.cat((x,self.alter(x)),dim=1)
            return self.down(x),x

class SELayer(torch.nn.Module):
    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool3d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channel, channel // reduction, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(channel // reduction, channel, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        b, c,_, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class AutoMerge(torch.nn.Module):
    def __init__(self,in_ch=32,inter_ch=4,out_ch=1,kernel=4,scale=2):
        super(AutoMerge,self).__init__()
        self.att=SELayer(channel=in_ch)
        self.up=torch.nn.ConvTranspose3d(in_ch, inter_ch, kernel_size=(kernel,kernel,kernel), 
                                    stride=(2,2,2), padding=(1, 1, 1), output_padding=0, bias=True)
        
    def forward(self,x): 
        x = self.att(x)
        x = self.up(x)
        return x


class AutoMerge_noalter(torch.nn.Module):
    def __init__(self,in_ch=8,inter_ch=1,out_ch=1,kernel=4,scale=2):
        super(AutoMerge_noalter,self).__init__()
        self.att=SELayer(channel=in_ch)
        self.up=torch.nn.ConvTranspose3d(in_ch, inter_ch, kernel_size=(kernel,kernel,kernel), 
                                    stride=(2,2,2), padding=(1, 1, 1), output_padding=0, bias=True)
        
    def forward(self,x): 
        x = self.att(x)
        x = self.up(x)
        return x


class DoubleConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, out_ch, 3, padding=1),
            torch.nn.InstanceNorm3d(out_ch),  
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv3d(out_ch, out_ch, 3, padding=1),
            torch.nn.InstanceNorm3d(out_ch),
        )

        self.residual_upsampler = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
            torch.nn.InstanceNorm3d(out_ch))

        self.relu = torch.nn.LeakyReLU(inplace=True)

    def forward(self, input):
        return self.relu(self.conv(input)+self.residual_upsampler(input))

class SHMLP(torch.nn.Module):

    def __init__(self, d_model=512, S=256, num_head=4):
        super().__init__()
        # self.embedding = torch.nn.Conv3d()
        self.mk = torch.nn.Linear(d_model // num_head, S, bias=False)
        self.mv = torch.nn.Linear(S, d_model // num_head, bias=False)
        self.softmax = torch.nn.Softmax(dim=-2)
        self.linear_out = torch.nn.Linear(d_model, d_model)
        self.num_head = num_head

    def forward(self, queries):
        b, c, d, h, w = queries.size()
        queries = queries.view(b, c, -1).permute(0, 2, 1).contiguous()  # (b, n, c)
        queries = queries.view(b, -1, self.num_head, int(c // self.num_head)).permute(0, 2, 1, 3).contiguous()  # (b, head, n, c)
        attn = self.mk(queries)  # bs,head n s
        attn = self.softmax(attn)  # bs,head n s
        attn = attn/torch.sum(attn, dim=-1, keepdim=True) # bs,head,n,S
        out = self.mv(attn) # bs,head, n, d_model // head

        out = out.permute(0, 2, 1, 3).contiguous().view(b, -1, c)
        out = self.linear_out(out).permute(0, 2, 1).contiguous().view(b, c, d, h, w)
        return out


class LWSA(torch.nn.Module):
    def __init__(self, in_channel=128, num_head=4):
        super(LWSA, self).__init__()
        self.inter_channel = in_channel//2   # 256
        self.k_map = torch.nn.Conv3d(in_channel, self.inter_channel, 1, 1, 0)
        self.q_map = torch.nn.Conv3d(in_channel, self.inter_channel, 1, 1, 0)
        # self.v_map = torch.nn.Conv3d(in_channel, self.inter_channel, 1, 1, 0)
        # self.out_expand=torch.nn.Conv3d(self.inter_channel, in_channel, 1, 1, 0)
        self.v_skip1 = torch.nn.Conv3d(in_channel,in_channel//2,3,1,1,groups=in_channel//2)
        self.v_skip2 = torch.nn.Conv3d(in_channel,in_channel//2,5,1,2,groups=in_channel//2)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.num_head = num_head
        # self.proj = torch.nn.Linear(in_channel, in_channel)

    def forward(self, x):
        b, c, d, h, w = x.size()
        q = self.q_map(x).view(b, self.inter_channel, -1).permute(0, 2, 1).contiguous()
        q = q.view(b, -1, self.num_head, self.inter_channel // self.num_head).contiguous()  # (b, head, n, c)
        q = q.view(b, -1, self.inter_channel // self.num_head)

        k = self.k_map(x).view(b, self.inter_channel, -1).permute(0, 2, 1).contiguous()
        k = k.view(b, -1, self.num_head, self.inter_channel // self.num_head).contiguous()  # (b, head, c, n)
        k = k.view(b, -1, self.inter_channel // self.num_head).permute(0, 2, 1).contiguous()

        v = x.view(b, c, -1).permute(0, 2, 1).contiguous()  # move channel to the back  # (b, n, c)
        v = v.view(b, -1, self.num_head, c // self.num_head).contiguous()  # (b, head, n, c)
        v = v.view(b, -1, c // self.num_head)

        # v = self.v_map(x).view(b, c, -1).permute(0, 2, 1).contiguous()  # move channel to the back  # (b, n, c)
        # v = v.view(b, -1, self.num_head, self.inter_channel // self.num_head).contiguous()  # (b, head, n, c)
        # v = v.view(b, -1, self.inter_channel // self.num_head)

        qk = torch.matmul(q, k)  # (b, head, n, n)
        qk = qk / math.sqrt(int(self.inter_channel // self.num_head))
        qk = self.softmax(qk)   # (b, head, n, n)

        out = torch.matmul(qk, v)
        out=out.view(b, d, h, w, v.shape[-1] * self.num_head)

        out = out.permute(0, 4, 1, 2, 3).contiguous()
        # out = self.out_expand(out)
        skip = torch.cat((self.v_skip1(x),self.v_skip2(x)),dim=1)
        return out+torch.nn.Tanh()(skip)
        # return self.out_expand(out)


class EAModule(torch.nn.Module):
    def __init__(self, in_channel=256, ms=True):
        super(EAModule, self).__init__()
        self.SlayerNorm = nn.InstanceNorm3d(in_channel, eps=1e-6)
        self.ElayerNorm = nn.InstanceNorm3d(in_channel, eps=1e-6)
        self.LWSA = LWSA(in_channel)
        self.SHMLP = SHMLP(in_channel)

    def forward(self, x):
        x_n = self.SlayerNorm(x)
        x = x + self.LWSA(x_n)
        x_n = self.ElayerNorm(x)
        x = x + self.SHMLP(x_n)
        return x

'''
class function_1(nn.Module):
    def __init__(self):
        super(function_1, self).__init__()
        self.local_LWSA = local_LWSA()
'''

class VolumeFormer_ras_noshare(torch.nn.Module):
    def __init__(self, in_channel=1,inter_channel=None, out_channel=1,scale=2):
        super(VolumeFormer_ras_noshare, self).__init__()
        self.down = AutoDecomposition(in_channel,inter_channel,None)
        if inter_channel is None:
            inter_channel=in_channel
        self.conv1 = DoubleConv(inter_channel*scale**3, 64)
        self.pool1 = torch.nn.MaxPool3d(2)  #112
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = torch.nn.MaxPool3d(2)  #56
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = torch.nn.MaxPool3d(2)  #28

        self.conv4 = DoubleConv(256, 512) 

        self.ea1 = EAModule(512)
        self.ea_pool1 = torch.nn.MaxPool3d(2)
        self.ea2 = EAModule(512)
        self.ea_pool2 = torch.nn.MaxPool3d(2)
        self.ea3 = EAModule(512)
        self.ea4 = EAModule(512)
        self.ea5 = EAModule(512)

        self.up_ea1 = torch.nn.ConvTranspose3d(512, 512, 2, stride=2)
        self.up_ea2 = torch.nn.ConvTranspose3d(512, 512, 2, stride=2)

        self.up4 = torch.nn.ConvTranspose3d(512, 512, 2, stride=2)
        self.up_conv4 = DoubleConv(512, 256)
        self.up3 = torch.nn.ConvTranspose3d(256, 256, 2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)
        self.up2 = torch.nn.ConvTranspose3d(128, 128, 2, stride=2)
        self.up_conv2 = DoubleConv(128, 64)
        self.out_conv = torch.nn.Conv3d(64, inter_channel*scale**3, 3, 1, 1)
        self.up = AutoMerge(inter_channel*scale**3,inter_channel,out_channel)
        self.choose_out=torch.nn.Sequential(
            torch.nn.Conv3d(inter_channel,out_channel,1,1,0),
        )

    def forward(self, x):
        x, x_for_skip=self.down(x)
        c1 = self.conv1(x)
        c1p = self.pool1(c1)
        c2 = self.conv2(c1p)
        c2p = self.pool2(c2)
        c3 = self.conv3(c2p)
        c3p = self.pool3(c3)
        c4 = self.conv4(c3p)

        ea1 = self.ea1(c4)
        ea1_p = self.ea_pool1(ea1)
        ea2 = self.ea2(ea1_p)
        ea2_p = self.ea_pool2(ea2)
        ea3 = self.ea3(ea2_p)

        up_ea3 = self.up_ea1(ea3)+ea2
        up_ea3 = self.ea4(up_ea3)
        up_ea2 = self.up_ea2(up_ea3)+ea1
        up_ea2 = self.ea5(up_ea2)

        up4 = self.up4(up_ea2+c4)
        up4_conv = self.up_conv4(up4)
        up3 = self.up3(up4_conv+c3)
        up3_conv = self.up_conv3(up3)
        up2 = self.up2(up3_conv+c2)
        up2_conv = self.up_conv2(up2)

        out = self.out_conv(up2_conv+c1)
        out = self.up(out)
        final_out = self.choose_out(out+x_for_skip)
        return torch.sigmoid(final_out)

class VolumeFormer_ras_noalter_noshare(torch.nn.Module):
    def __init__(self, in_channel=1,inter_channel=None, out_channel=1,scale=2):
        super(VolumeFormer_ras_noalter_noshare, self).__init__()
        self.down = AutoDecomposition(in_channel,inter_channel,None)
        if inter_channel is None:
            inter_channel=in_channel
        self.conv1 = DoubleConv(inter_channel*scale**3, 64)
        self.pool1 = torch.nn.MaxPool3d(2)  #112
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = torch.nn.MaxPool3d(2)  #56
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = torch.nn.MaxPool3d(2)  #28

        self.conv4 = DoubleConv(256, 512) 

        self.ea1 = EAModule(512)
        self.ea_pool1 = torch.nn.MaxPool3d(2)
        self.ea2 = EAModule(512)
        self.ea_pool2 = torch.nn.MaxPool3d(2)
        self.ea3 = EAModule(512)
        self.ea4 = EAModule(512)
        self.ea5 = EAModule(512)
        
        self.up_ea1 = torch.nn.ConvTranspose3d(512, 512, 2, stride=2)
        self.up_ea2 = torch.nn.ConvTranspose3d(512, 512, 2, stride=2)
        # self.squeeze_ea1 = torch.nn.Conv3d(512, 256, 1, 1, 0)
        # self.squeeze_ea2 = torch.nn.Conv3d(512, 256, 1, 1, 0)

        self.up4 = torch.nn.ConvTranspose3d(512, 512, 2, stride=2)
        self.up_conv4 = DoubleConv(512, 256)
        self.up3 = torch.nn.ConvTranspose3d(256, 256, 2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)
        self.up2 = torch.nn.ConvTranspose3d(128, 128, 2, stride=2)
        self.up_conv2 = DoubleConv(128, 64)
        self.out_conv = torch.nn.Conv3d(64, inter_channel*scale**3, 3, 1, 1)
        self.up = AutoMerge(inter_channel*scale**3,inter_channel,out_channel)
        self.choose_out=torch.nn.Sequential(
            torch.nn.Conv3d(inter_channel,out_channel,1,1,0),
        )

    def forward(self, x):
        x, x_for_skip=self.down(x)
        c1 = self.conv1(x)
        c1p = self.pool1(c1)
        c2 = self.conv2(c1p)
        c2p = self.pool2(c2)
        c3 = self.conv3(c2p)
        c3p = self.pool3(c3)
        c4 = self.conv4(c3p)

        ea1 = self.ea1(c4)
        ea1_p = self.ea_pool1(ea1)
        ea2 = self.ea2(ea1_p)
        ea2_p = self.ea_pool2(ea2)
        ea3 = self.ea3(ea2_p)

        up_ea3 = self.up_ea1(ea3)+ea2
        up_ea3 = self.ea4(up_ea3)
        up_ea2 = self.up_ea2(up_ea3)+ea1
        up_ea2 = self.ea5(up_ea2)

        up4 = self.up4(up_ea2+c4)
        up4_conv = self.up_conv4(up4)
        up3 = self.up3(up4_conv+c3)
        up3_conv = self.up_conv3(up3)
        up2 = self.up2(up3_conv+c2)
        up2_conv = self.up_conv2(up2)

        out = self.out_conv(up2_conv+c1)
        out = self.up(out)
        final_out = self.choose_out(out+x_for_skip)
        return torch.sigmoid(final_out)



class VolumeFormer_ras_noalter(torch.nn.Module):
    def __init__(self, in_channel=1,inter_channel=None, out_channel=1,scale=2):
        super(VolumeFormer_ras_noalter, self).__init__()
        self.down = AutoDecomposition(in_channel,inter_channel,None)
        if inter_channel is None:
            inter_channel=in_channel
        self.conv1 = DoubleConv(inter_channel*scale**3, 64)
        self.pool1 = torch.nn.MaxPool3d(2)  #112
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = torch.nn.MaxPool3d(2)  #56
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = torch.nn.MaxPool3d(2)  #28

        self.conv4 = DoubleConv(256, 512) 

        self.ea = EAModule(512)
        self.decoder_ea = EAModule(512)
        self.ea_pool1 = torch.nn.MaxPool3d(2)
        self.ea_pool2 = torch.nn.MaxPool3d(2)
        self.up_ea1 = torch.nn.ConvTranspose3d(512, 512, 2, stride=2)
        self.up_ea2 = torch.nn.ConvTranspose3d(512, 512, 2, stride=2)
        # self.squeeze_ea1 = torch.nn.Conv3d(512, 256, 1, 1, 0)
        # self.squeeze_ea2 = torch.nn.Conv3d(512, 256, 1, 1, 0)

        self.up4 = torch.nn.ConvTranspose3d(512, 512, 2, stride=2)
        self.up_conv4 = DoubleConv(512, 256)
        self.up3 = torch.nn.ConvTranspose3d(256, 256, 2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)
        self.up2 = torch.nn.ConvTranspose3d(128, 128, 2, stride=2)
        self.up_conv2 = DoubleConv(128, 64)
        self.out_conv = torch.nn.Conv3d(64, inter_channel*scale**3, 3, 1, 1)
        self.up = AutoMerge(inter_channel*scale**3,inter_channel,out_channel)
        self.choose_out=torch.nn.Sequential(
            torch.nn.Conv3d(inter_channel,out_channel,1,1,0),
        )

    def forward(self, x):
        x, x_for_skip=self.down(x)
        c1 = self.conv1(x)
        c1p = self.pool1(c1)
        c2 = self.conv2(c1p)
        c2p = self.pool2(c2)
        c3 = self.conv3(c2p)
        c3p = self.pool3(c3)
        c4 = self.conv4(c3p)

        ea1 = self.ea(c4)
        ea1_p = self.ea_pool1(ea1)
        ea2 = self.ea(ea1_p)
        ea2_p = self.ea_pool2(ea2)
        ea3 = self.ea(ea2_p)

        up_ea3 = self.up_ea1(ea3)+ea2
        up_ea3 = self.decoder_ea(up_ea3)
        up_ea2 = self.up_ea2(up_ea3)+ea1
        up_ea2 = self.decoder_ea(up_ea2)

        up4 = self.up4(up_ea2+c4)
        up4_conv = self.up_conv4(up4)
        up3 = self.up3(up4_conv+c3)
        up3_conv = self.up_conv3(up3)
        up2 = self.up2(up3_conv+c2)
        up2_conv = self.up_conv2(up2)

        out = self.out_conv(up2_conv+c1)
        out = self.up(out)
        final_out = self.choose_out(out+x_for_skip)
        return torch.sigmoid(final_out)

class VolumeFormer_ras_noformer(torch.nn.Module):
    def __init__(self, in_channel=1,inter_channel=None, out_channel=1,scale=2):
        super(VolumeFormer_ras_noformer, self).__init__()
        self.down = AutoDecomposition(in_channel,inter_channel,None)
        if inter_channel is None:
            inter_channel=in_channel
        self.conv1 = DoubleConv(inter_channel*scale**3, 64)
        self.pool1 = torch.nn.MaxPool3d(2)  #112
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = torch.nn.MaxPool3d(2)  #56
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = torch.nn.MaxPool3d(2)  #28

        self.conv4 = DoubleConv(256, 512) 

        self.ea = DoubleConv(512,512)
        self.decoder_ea = DoubleConv(512,512)
        self.ea_pool1 = torch.nn.MaxPool3d(2)
        self.ea_pool2 = torch.nn.MaxPool3d(2)
        self.up_ea1 = torch.nn.ConvTranspose3d(512, 512, 2, stride=2)
        self.up_ea2 = torch.nn.ConvTranspose3d(512, 512, 2, stride=2)
        # self.squeeze_ea1 = torch.nn.Conv3d(512, 256, 1, 1, 0)
        # self.squeeze_ea2 = torch.nn.Conv3d(512, 256, 1, 1, 0)

        self.up4 = torch.nn.ConvTranspose3d(512, 512, 2, stride=2)
        self.up_conv4 = DoubleConv(512, 256)
        self.up3 = torch.nn.ConvTranspose3d(256, 256, 2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)
        self.up2 = torch.nn.ConvTranspose3d(128, 128, 2, stride=2)
        self.up_conv2 = DoubleConv(128, 64)
        self.out_conv = torch.nn.Conv3d(64, inter_channel*scale**3, 3, 1, 1)
        self.up = AutoMerge(inter_channel*scale**3,inter_channel,out_channel)
        self.choose_out=torch.nn.Sequential(
            torch.nn.Conv3d(inter_channel,out_channel,1,1,0),
        )

    def forward(self, x):
        x, x_for_skip=self.down(x)
        c1 = self.conv1(x)
        c1p = self.pool1(c1)
        c2 = self.conv2(c1p)
        c2p = self.pool2(c2)
        c3 = self.conv3(c2p)
        c3p = self.pool3(c3)
        c4 = self.conv4(c3p)

        ea1 = self.ea(c4)
        ea1_p = self.ea_pool1(ea1)
        ea2 = self.ea(ea1_p)
        ea2_p = self.ea_pool2(ea2)
        ea3 = self.ea(ea2_p)

        up_ea3 = self.up_ea1(ea3)+ea2
        up_ea3 = self.decoder_ea(up_ea3)
        up_ea2 = self.up_ea2(up_ea3)+ea1
        up_ea2 = self.decoder_ea(up_ea2)

        up4 = self.up4(up_ea2+c4)
        up4_conv = self.up_conv4(up4)
        up3 = self.up3(up4_conv+c3)
        up3_conv = self.up_conv3(up3)
        up2 = self.up2(up3_conv+c2)
        up2_conv = self.up_conv2(up2)

        out = self.out_conv(up2_conv+c1)
        out = self.up(out)
        final_out = self.choose_out(out+x_for_skip)
        return torch.sigmoid(final_out)

class VolumeFormer_ras_hd_noformer(torch.nn.Module):
    def __init__(self, in_channel=1,inter_channel=None, out_channel=1,scale=2):
        super(VolumeFormer_ras_hd_noformer, self).__init__()
        self.down = PixelUnshuffle3d(in_channel,inter_ch=inter_channel,scale_d=scale,scale_h=scale,scale_w=scale)
        if inter_channel is None:
            inter_channel=in_channel
        self.conv1 = DoubleConv(inter_channel*scale**3, 64)
        self.pool1 = torch.nn.MaxPool3d(2)  #112
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = torch.nn.MaxPool3d(2)  #56
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = torch.nn.MaxPool3d(2)  #28

        self.conv4 = DoubleConv(256, 512) 

        self.ea = DoubleConv(512,512)
        self.decoder_ea = DoubleConv(512,512)
        self.ea_pool1 = torch.nn.MaxPool3d(2)
        self.ea_pool2 = torch.nn.MaxPool3d(2)
        self.up_ea1 = torch.nn.ConvTranspose3d(512, 512, 2, stride=2)
        self.up_ea2 = torch.nn.ConvTranspose3d(512, 512, 2, stride=2)
        # self.squeeze_ea1 = torch.nn.Conv3d(512, 256, 1, 1, 0)
        # self.squeeze_ea2 = torch.nn.Conv3d(512, 256, 1, 1, 0)

        self.up4 = torch.nn.ConvTranspose3d(512, 512, 2, stride=2)
        self.up_conv4 = DoubleConv(512, 256)
        self.up3 = torch.nn.ConvTranspose3d(256, 256, 2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)
        self.up2 = torch.nn.ConvTranspose3d(128, 128, 2, stride=2)
        self.up_conv2 = DoubleConv(128, 64)
        self.out_conv = torch.nn.Conv3d(64, inter_channel*scale**3, 3, 1, 1)
        self.up = PixelShuffle3d()
        self.choose_out=torch.nn.Sequential(
            torch.nn.Conv3d(inter_channel,out_channel,1,1,0),
        )

    def forward(self, x):
        x, x_for_skip=self.down(x)
        c1 = self.conv1(x)
        c1p = self.pool1(c1)
        c2 = self.conv2(c1p)
        c2p = self.pool2(c2)
        c3 = self.conv3(c2p)
        c3p = self.pool3(c3)
        c4 = self.conv4(c3p)

        ea1 = self.ea(c4)
        ea1_p = self.ea_pool1(ea1)
        ea2 = self.ea(ea1_p)
        ea2_p = self.ea_pool2(ea2)
        ea3 = self.ea(ea2_p)

        up_ea3 = self.up_ea1(ea3)+ea2
        up_ea3 = self.decoder_ea(up_ea3)
        up_ea2 = self.up_ea2(up_ea3)+ea1
        up_ea2 = self.decoder_ea(up_ea2)

        up4 = self.up4(up_ea2+c4)
        up4_conv = self.up_conv4(up4)
        up3 = self.up3(up4_conv+c3)
        up3_conv = self.up_conv3(up3)
        up2 = self.up2(up3_conv+c2)
        up2_conv = self.up_conv2(up2)

        out = self.out_conv(up2_conv+c1)
        out = self.up(out)
        final_out = self.choose_out(out+x_for_skip)
        return torch.sigmoid(final_out)


# class VolumeFormer_ras_deconv4(torch.nn.Module):
#     def __init__(self, in_channel=4,inter_channel=None, out_channel=1,scale=2):
#         super(VolumeFormer_ras_deconv4, self).__init__()
#         self.down = AutoDecomposition(in_channel,inter_channel,None)
#         if inter_channel is None:
#             inter_channel=in_channel
#         self.conv1 = DoubleConv(inter_channel*scale**3, 64)
#         self.pool1 = torch.nn.MaxPool3d(2)  #112
#         self.conv2 = DoubleConv(64, 128)
#         self.pool2 = torch.nn.MaxPool3d(2)  #56
#         self.conv3 = DoubleConv(128, 256)
#         self.pool3 = torch.nn.MaxPool3d(2)  #28

#         self.conv4 = DoubleConv(256, 512) 

#         self.ea = EAModule(512)
#         self.decoder_ea = EAModule(512)
#         self.ea_pool1 = torch.nn.MaxPool3d(2)
#         self.ea_pool2 = torch.nn.MaxPool3d(2)
#         self.up_ea1 = torch.nn.ConvTranspose3d(512, 512, 4, stride=2,padding=1)
#         self.up_ea2 = torch.nn.ConvTranspose3d(512, 512, 4, stride=2,padding=1)
#         # self.squeeze_ea1 = torch.nn.Conv3d(512, 256, 1, 1, 0)
#         # self.squeeze_ea2 = torch.nn.Conv3d(512, 256, 1, 1, 0)

#         self.up4 = torch.nn.ConvTranspose3d(512, 512, 4, stride=2,padding=1)
#         self.up_conv4 = DoubleConv(512, 256)
#         self.up3 = torch.nn.ConvTranspose3d(256, 256, 4, stride=2,padding=1)
#         self.up_conv3 = DoubleConv(256, 128)
#         self.up2 = torch.nn.ConvTranspose3d(128, 128, 4, stride=2,padding=1)
#         self.up_conv2 = DoubleConv(128, 64)
#         self.out_conv = torch.nn.Conv3d(64, inter_channel*scale**3, 3, 1, 1)
#         self.up = AutoMerge(inter_channel*scale**3,inter_channel,out_channel)
#         self.choose_out=torch.nn.Sequential(
#             torch.nn.Conv3d(inter_channel,out_channel,1,1,0),
#         )

#     def forward(self, x):
#         x, x_for_skip=self.down(x)
#         c1 = self.conv1(x)
#         c1p = self.pool1(c1)
#         c2 = self.conv2(c1p)
#         c2p = self.pool2(c2)
#         c3 = self.conv3(c2p)
#         c3p = self.pool3(c3)
#         c4 = self.conv4(c3p)

#         ea1 = self.ea(c4)
#         ea1_p = self.ea_pool1(ea1)
#         ea2 = self.ea(ea1_p)
#         ea2_p = self.ea_pool2(ea2)
#         ea3 = self.ea(ea2_p)

#         up_ea3 = self.up_ea1(ea3)+ea2
#         up_ea3 = self.decoder_ea(up_ea3)
#         up_ea2 = self.up_ea2(up_ea3)+ea1
#         up_ea2 = self.decoder_ea(up_ea2)

#         up4 = self.up4(up_ea2+c4)
#         up4_conv = self.up_conv4(up4)
#         up3 = self.up3(up4_conv+c3)
#         up3_conv = self.up_conv3(up3)
#         up2 = self.up2(up3_conv+c2)
#         up2_conv = self.up_conv2(up2)

#         out = self.out_conv(up2_conv+c1)
#         out = self.up(out)
#         final_out = self.choose_out(out+x_for_skip)
#         return torch.sigmoid(final_out)

class VolumeFormer_ras(torch.nn.Module):
    def __init__(self, in_channel=4,inter_channel=None, out_channel=1,scale=2):
        super(VolumeFormer_ras, self).__init__()
        self.down = AutoDecomposition(in_channel,inter_channel,None)
        if inter_channel is None:
            inter_channel=in_channel
        self.conv1 = DoubleConv(inter_channel*scale**3, 64)
        self.pool1 = torch.nn.MaxPool3d(2)  #112
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = torch.nn.MaxPool3d(2)  #56
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = torch.nn.MaxPool3d(2)  #28

        self.conv4 = DoubleConv(256, 512) 

        self.ea = EAModule(512)
        self.decoder_ea = EAModule(512)
        self.ea_pool1 = torch.nn.MaxPool3d(2)
        self.ea_pool2 = torch.nn.MaxPool3d(2)
        self.up_ea1 = torch.nn.ConvTranspose3d(512, 512, 2, stride=2)
        self.up_ea2 = torch.nn.ConvTranspose3d(512, 512, 2, stride=2)
        # self.squeeze_ea1 = torch.nn.Conv3d(512, 256, 1, 1, 0)
        # self.squeeze_ea2 = torch.nn.Conv3d(512, 256, 1, 1, 0)

        self.up4 = torch.nn.ConvTranspose3d(512, 512, 2, stride=2)
        self.up_conv4 = DoubleConv(512, 256)
        self.up3 = torch.nn.ConvTranspose3d(256, 256, 2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)
        self.up2 = torch.nn.ConvTranspose3d(128, 128, 2, stride=2)
        self.up_conv2 = DoubleConv(128, 64)
        self.out_conv = torch.nn.Conv3d(64, inter_channel*scale**3, 3, 1, 1)
        self.up = AutoMerge(inter_channel*scale**3,inter_channel,out_channel)
        self.choose_out=torch.nn.Sequential(
            torch.nn.Conv3d(inter_channel,out_channel,1,1,0),
        )

    def forward(self, x):
        x, x_for_skip=self.down(x)
        c1 = self.conv1(x)
        c1p = self.pool1(c1)
        c2 = self.conv2(c1p)
        c2p = self.pool2(c2)
        c3 = self.conv3(c2p)
        c3p = self.pool3(c3)
        c4 = self.conv4(c3p)

        ea1 = self.ea(c4)
        ea1_p = self.ea_pool1(ea1)
        ea2 = self.ea(ea1_p)
        ea2_p = self.ea_pool2(ea2)
        ea3 = self.ea(ea2_p)

        up_ea3 = self.up_ea1(ea3)+ea2
        up_ea3 = self.decoder_ea(up_ea3)
        up_ea2 = self.up_ea2(up_ea3)+ea1
        up_ea2 = self.decoder_ea(up_ea2)

        up4 = self.up4(up_ea2+c4)
        up4_conv = self.up_conv4(up4)
        up3 = self.up3(up4_conv+c3)
        up3_conv = self.up_conv3(up3)
        up2 = self.up2(up3_conv+c2)
        up2_conv = self.up_conv2(up2)

        out = self.out_conv(up2_conv+c1)
        out = self.up(out)
        final_out = self.choose_out(out+x_for_skip)
        return torch.sigmoid(final_out)


# class VolumeFormer_ras_nonlocalpooling(torch.nn.Module):
#     def __init__(self, in_channel=1,inter_channel=None, out_channel=1,scale=2):
#         super(VolumeFormer_ras_nonlocalpooling, self).__init__()
#         self.down = AutoDecomposition(in_channel,inter_channel,None)
#         if inter_channel is None:
#             inter_channel=in_channel
#         self.conv1 = DoubleConv(inter_channel*scale**3, 64)
#         self.pool1 = NonLocalPooling(64,[64,96,96],2,[4,6,6])  #112
#         self.conv2 = DoubleConv(64, 128)
#         self.pool2 = NonLocalPooling(128,[32,48,48],2,[4,6,6])  #56
#         self.conv3 = DoubleConv(128, 256)
#         self.pool3 = NonLocalPooling(256,[16,24,24],2,[2,2,2])  #28

#         self.conv4 = DoubleConv(256, 512) 

#         self.ea = EAModule(512)
#         self.decoder_ea = EAModule(512)
#         self.ea_pool1 = NonLocalPooling(512,[8,12,12],2,[2,2,2])
#         self.ea_pool2 = NonLocalPooling(512,[4,6,6],2,[2,2,2]) 
#         self.up_ea1 = torch.nn.ConvTranspose3d(512, 512, 2, stride=2)
#         self.up_ea2 = torch.nn.ConvTranspose3d(512, 512, 2, stride=2)
#         # self.squeeze_ea1 = torch.nn.Conv3d(512, 256, 1, 1, 0)
#         # self.squeeze_ea2 = torch.nn.Conv3d(512, 256, 1, 1, 0)

#         self.up4 = torch.nn.ConvTranspose3d(512, 512, 2, stride=2)
#         self.up_conv4 = DoubleConv(512, 256)
#         self.up3 = torch.nn.ConvTranspose3d(256, 256, 2, stride=2)
#         self.up_conv3 = DoubleConv(256, 128)
#         self.up2 = torch.nn.ConvTranspose3d(128, 128, 2, stride=2)
#         self.up_conv2 = DoubleConv(128, 64)
#         self.out_conv = torch.nn.Conv3d(64, inter_channel*scale**3, 3, 1, 1)
#         self.up = AutoMerge(inter_channel*scale**3,inter_channel,out_channel)
#         self.choose_out=torch.nn.Sequential(
#             torch.nn.Conv3d(inter_channel,out_channel,1,1,0),
#         )

#     def forward(self, x):
#         x, x_for_skip=self.down(x)
#         c1 = self.conv1(x)
#         c1p = self.pool1(c1)
#         c2 = self.conv2(c1p)
#         c2p = self.pool2(c2)
#         c3 = self.conv3(c2p)
#         c3p = self.pool3(c3)
#         c4 = self.conv4(c3p)

#         ea1 = self.ea(c4)
#         ea1_p = self.ea_pool1(ea1)
#         ea2 = self.ea(ea1_p)
#         ea2_p = self.ea_pool2(ea2)
#         ea3 = self.ea(ea2_p)

#         up_ea3 = self.up_ea1(ea3)+ea2
#         up_ea3 = self.decoder_ea(up_ea3)
#         up_ea2 = self.up_ea2(up_ea3)+ea1
#         up_ea2 = self.decoder_ea(up_ea2)

#         up4 = self.up4(up_ea2+c4)
#         up4_conv = self.up_conv4(up4)
#         up3 = self.up3(up4_conv+c3)
#         up3_conv = self.up_conv3(up3)
#         up2 = self.up2(up3_conv+c2)
#         up2_conv = self.up_conv2(up2)

#         out = self.out_conv(up2_conv+c1)
#         out = self.up(out)
#         final_out = self.choose_out(out+x_for_skip)
#         return torch.sigmoid(final_out)


class VolumeFormer_ras_small(torch.nn.Module):
    def __init__(self, in_channel=1,inter_channel=None, out_channel=1,scale=2):
        super(VolumeFormer_ras_small, self).__init__()
        self.down = AutoDecomposition(in_channel,inter_channel,None)
        if inter_channel is None:
            inter_channel=in_channel
        self.conv1 = DoubleConv(inter_channel*scale**3, 32)
        self.pool1 = torch.nn.MaxPool3d(2)  #112
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = torch.nn.MaxPool3d(2)  #56
        self.conv3 = DoubleConv(64, 128)
        self.pool3 = torch.nn.MaxPool3d(2)  #28

        self.conv4 = DoubleConv(128, 256) 

        self.ea = EAModule(256)
        self.decoder_ea = EAModule(256)
        self.ea_pool1 = torch.nn.MaxPool3d(2)
        self.ea_pool2 = torch.nn.MaxPool3d(2)
        self.up_ea1 = torch.nn.ConvTranspose3d(256, 256, 2, stride=2)
        self.up_ea2 = torch.nn.ConvTranspose3d(256, 256, 2, stride=2)
        # self.squeeze_ea1 = torch.nn.Conv3d(512, 256, 1, 1, 0)
        # self.squeeze_ea2 = torch.nn.Conv3d(512, 256, 1, 1, 0)

        self.up4 = torch.nn.ConvTranspose3d(256, 256, 2, stride=2)
        self.up_conv4 = DoubleConv(256, 128)
        self.up3 = torch.nn.ConvTranspose3d(128, 128, 2, stride=2)
        self.up_conv3 = DoubleConv(128, 64)
        self.up2 = torch.nn.ConvTranspose3d(64, 64, 2, stride=2)
        self.up_conv2 = DoubleConv(64, 32)
        self.out_conv = torch.nn.Conv3d(32, inter_channel*scale**3, 3, 1, 1)
        self.up = AutoMerge(inter_channel*scale**3,inter_channel,out_channel)
        self.choose_out=torch.nn.Sequential(
            torch.nn.Conv3d(inter_channel,out_channel,1,1,0),
        )

    def forward(self, x):
        x, x_for_skip=self.down(x)
        c1 = self.conv1(x)
        c1p = self.pool1(c1)
        c2 = self.conv2(c1p)
        c2p = self.pool2(c2)
        c3 = self.conv3(c2p)
        c3p = self.pool3(c3)
        c4 = self.conv4(c3p)

        ea1 = self.ea(c4)
        ea1_p = self.ea_pool1(ea1)
        ea2 = self.ea(ea1_p)
        ea2_p = self.ea_pool2(ea2)
        ea3 = self.ea(ea2_p)

        up_ea3 = self.up_ea1(ea3)+ea2
        up_ea3 = self.decoder_ea(up_ea3)
        up_ea2 = self.up_ea2(up_ea3)+ea1
        up_ea2 = self.decoder_ea(up_ea2)

        up4 = self.up4(up_ea2+c4)
        up4_conv = self.up_conv4(up4)
        up3 = self.up3(up4_conv+c3)
        up3_conv = self.up_conv3(up3)
        up2 = self.up2(up3_conv+c2)
        up2_conv = self.up_conv2(up2)

        out = self.out_conv(up2_conv+c1)
        out = self.up(out)
        final_out = self.choose_out(out+x_for_skip)
        return torch.sigmoid(final_out)



# class VolumeFormer_ras_small_nonlocalpooling(torch.nn.Module):
#     def __init__(self, in_channel=1,inter_channel=None, out_channel=1,scale=2):
#         super(VolumeFormer_ras_small_nonlocalpooling, self).__init__()
#         self.down = AutoDecomposition(in_channel,inter_channel,None)
#         if inter_channel is None:
#             inter_channel=in_channel
#         self.conv1 = DoubleConv(inter_channel*scale**3, 32)
#         self.pool1 = NonLocalPooling(32,[64,96,96],2,[4,6,6])  #112
#         self.conv2 = DoubleConv(32, 64)
#         self.pool2 = NonLocalPooling(64,[32,48,48],2,[4,6,6])  #56
#         self.conv3 = DoubleConv(64, 128)
#         self.pool3 = NonLocalPooling(128,[16,24,24],2,[2,2,2])  #28

#         self.conv4 = DoubleConv(128, 256) 

#         self.ea = EAModule(256)
#         self.decoder_ea = EAModule(256)
#         self.ea_pool1 = NonLocalPooling(256,[8,12,12],2,[2,2,2])
#         self.ea_pool2 = NonLocalPooling(256,[4,6,6],2,[2,2,2])
#         self.up_ea1 = torch.nn.ConvTranspose3d(256, 256, 2, stride=2)
#         self.up_ea2 = torch.nn.ConvTranspose3d(256, 256, 2, stride=2)
#         # self.squeeze_ea1 = torch.nn.Conv3d(512, 256, 1, 1, 0)
#         # self.squeeze_ea2 = torch.nn.Conv3d(512, 256, 1, 1, 0)

#         self.up4 = torch.nn.ConvTranspose3d(256, 256, 2, stride=2)
#         self.up_conv4 = DoubleConv(256, 128)
#         self.up3 = torch.nn.ConvTranspose3d(128, 128, 2, stride=2)
#         self.up_conv3 = DoubleConv(128, 64)
#         self.up2 = torch.nn.ConvTranspose3d(64, 64, 2, stride=2)
#         self.up_conv2 = DoubleConv(64, 32)
#         self.out_conv = torch.nn.Conv3d(32, inter_channel*scale**3, 3, 1, 1)
#         self.up = AutoMerge(inter_channel*scale**3,inter_channel,out_channel)
#         self.choose_out=torch.nn.Sequential(
#             torch.nn.Conv3d(inter_channel,out_channel,1,1,0),
#         )

#     def forward(self, x):
#         x, x_for_skip=self.down(x)
#         c1 = self.conv1(x)
#         c1p = self.pool1(c1)
#         c2 = self.conv2(c1p)
#         c2p = self.pool2(c2)
#         c3 = self.conv3(c2p)
#         c3p = self.pool3(c3)
#         c4 = self.conv4(c3p)

#         ea1 = self.ea(c4)
#         ea1_p = self.ea_pool1(ea1)
#         ea2 = self.ea(ea1_p)
#         ea2_p = self.ea_pool2(ea2)
#         ea3 = self.ea(ea2_p)

#         up_ea3 = self.up_ea1(ea3)+ea2
#         up_ea3 = self.decoder_ea(up_ea3)
#         up_ea2 = self.up_ea2(up_ea3)+ea1
#         up_ea2 = self.decoder_ea(up_ea2)

#         up4 = self.up4(up_ea2+c4)
#         up4_conv = self.up_conv4(up4)
#         up3 = self.up3(up4_conv+c3)
#         up3_conv = self.up_conv3(up3)
#         up2 = self.up2(up3_conv+c2)
#         up2_conv = self.up_conv2(up2)

#         out = self.out_conv(up2_conv+c1)
#         out = self.up(out)
#         final_out = self.choose_out(out+x_for_skip)
#         return torch.sigmoid(final_out)

# class PixelShuffle3d(torch.nn.Module):
#     '''
#     This class is a 3d version of pixelshuffle.
#     '''
#     def __init__(self, scale_d=2, scale_h=2, scale_w=2):
#         '''
#         :param scale: upsample scale
#         '''
#         super().__init__()
#         self.scale_d = scale_d
#         self.scale_h=scale_h
#         self.scale_w=scale_w

#     def forward(self, input):
#         batch_size, channels, in_depth, in_height, in_width = input.size()
#         nOut = channels // (self.scale_d*self.scale_h*self.scale_w)

#         out_depth = in_depth * self.scale_d
#         out_height = in_height * self.scale_h
#         out_width = in_width * self.scale_w

#         input_view = input.contiguous().view(batch_size, nOut, self.scale_d, self.scale_h, self.scale_w, in_depth, in_height, in_width)

#         output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

#         return output.view(batch_size, nOut, out_depth, out_height, out_width)

class PixelUnshuffle3d(torch.nn.Module):
    '''
    This class is a 3d version of pixelunshuffle.
    '''
    def __init__(self, in_ch, inter_ch=None, scale_d=2, scale_h=2, scale_w=2):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale_d = scale_d
        self.scale_h=scale_h
        self.scale_w=scale_w
        self.inter_ch=inter_ch
        self.alter=None
        if not self.inter_ch is None:
            assert inter_ch>in_ch, "if enabled, inter_ch must be larger than in_ch"
            self.alter=torch.nn.Conv3d(in_ch,inter_ch-in_ch,3,1,1)

    def forward(self, input):
        if not self.inter_ch is None:
            alter=self.alter(input)
            input=torch.cat((alter,input),dim=1)

        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels * self.scale_d*self.scale_h*self.scale_w

        out_depth = in_depth // self.scale_d
        out_height = in_height // self.scale_h
        out_width = in_width // self.scale_w

        input_view = input.contiguous().view(batch_size, channels, out_depth, self.scale_d, out_height, self.scale_h, out_width, self.scale_w)

        output = input_view.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width),input


class VolumeFormer_hd(torch.nn.Module):
    def __init__(self, in_channel=1,inter_channel=None, out_channel=1,scale=2):
        super(VolumeFormer_hd, self).__init__()
        self.down = PixelUnshuffle3d(in_channel,inter_ch=inter_channel,scale_d=scale,scale_h=scale,scale_w=scale)
        if inter_channel is None:
            inter_channel=in_channel
        self.conv1 = DoubleConv(inter_channel*scale**3, 64)
        self.pool1 = torch.nn.MaxPool3d(2)  #112
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = torch.nn.MaxPool3d(2)  #56
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = torch.nn.MaxPool3d(2)  #28

        self.conv4 = DoubleConv(256, 512) 

        self.ea = EAModule(512)
        self.decoder_ea = EAModule(512)
        self.ea_pool1 = torch.nn.MaxPool3d(2)
        self.ea_pool2 = torch.nn.MaxPool3d(2)
        self.up_ea1 = torch.nn.ConvTranspose3d(512, 512, 2, stride=2)
        self.up_ea2 = torch.nn.ConvTranspose3d(512, 512, 2, stride=2)
        # self.squeeze_ea1 = torch.nn.Conv3d(512, 256, 1, 1, 0)
        # self.squeeze_ea2 = torch.nn.Conv3d(512, 256, 1, 1, 0)

        self.up4 = torch.nn.ConvTranspose3d(512, 512, 2, stride=2)
        self.up_conv4 = DoubleConv(512, 256)
        self.up3 = torch.nn.ConvTranspose3d(256, 256, 2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)
        self.up2 = torch.nn.ConvTranspose3d(128, 128, 2, stride=2)
        self.up_conv2 = DoubleConv(128, 64)
        self.out_conv = torch.nn.Conv3d(64, inter_channel*scale**3, 3, 1, 1)
        self.up = PixelShuffle3d()
        self.choose_out=torch.nn.Sequential(
            torch.nn.Conv3d(inter_channel,out_channel,1,1,0),
        )

    def forward(self, x):
        x, x_for_skip=self.down(x)
        c1 = self.conv1(x)
        c1p = self.pool1(c1)
        c2 = self.conv2(c1p)
        c2p = self.pool2(c2)
        c3 = self.conv3(c2p)
        c3p = self.pool3(c3)
        c4 = self.conv4(c3p)

        ea1 = self.ea(c4)
        ea1_p = self.ea_pool1(ea1)
        ea2 = self.ea(ea1_p)
        ea2_p = self.ea_pool2(ea2)
        ea3 = self.ea(ea2_p)

        up_ea3 = self.up_ea1(ea3)+ea2
        up_ea3 = self.decoder_ea(up_ea3)
        up_ea2 = self.up_ea2(up_ea3)+ea1
        up_ea2 = self.decoder_ea(up_ea2)

        up4 = self.up4(up_ea2+c4)
        up4_conv = self.up_conv4(up4)
        up3 = self.up3(up4_conv+c3)
        up3_conv = self.up_conv3(up3)
        up2 = self.up2(up3_conv+c2)
        up2_conv = self.up_conv2(up2)

        out = self.out_conv(up2_conv+c1)
        out = self.up(out)
        final_out = self.choose_out(out+x_for_skip)
        return torch.sigmoid(final_out)


# class VolumeFormer(torch.nn.Module):
#     def __init__(self, in_channel=1,inter_channel=4, out_channel=1,scale=2):
#         super(VolumeFormer, self).__init__()
#         self.down = AutoDecomposition(in_channel,inter_channel,None)
#         self.conv1 = DoubleConv(inter_channel*scale**3, 64)
#         self.pool1 = torch.nn.MaxPool3d(2)  #112
#         self.conv2 = DoubleConv(64, 128)
#         self.pool2 = torch.nn.MaxPool3d(2)  #56
#         self.conv3 = DoubleConv(128, 256)
#         self.pool3 = torch.nn.MaxPool3d(2)  #28

#         self.conv4 = DoubleConv(256, 512) 

#         self.ea = EAModule(512)
#         self.decoder_ea = EAModule(512)
#         self.ea_pool1 = torch.nn.MaxPool3d(2)
#         self.ea_pool2 = torch.nn.MaxPool3d(2)
#         self.up_ea1 = torch.nn.ConvTranspose3d(512, 256, 2, stride=2)
#         self.up_ea2 = torch.nn.ConvTranspose3d(512, 256, 2, stride=2)
#         self.squeeze_ea1 = torch.nn.Conv3d(512, 256, 1, 1, 0)
#         self.squeeze_ea2 = torch.nn.Conv3d(512, 256, 1, 1, 0)

#         self.up4 = torch.nn.ConvTranspose3d(1024, 512, 2, stride=2)
#         self.up_conv4 = DoubleConv(512, 256)
#         self.up3 = torch.nn.ConvTranspose3d(512, 256, 2, stride=2)
#         self.up_conv3 = DoubleConv(256, 128)
#         self.up2 = torch.nn.ConvTranspose3d(256, 128, 2, stride=2)
#         self.up_conv2 = DoubleConv(128, 64)
#         self.out_conv = torch.nn.Conv3d(128, inter_channel*scale**3, 3, 1, 1)
#         self.up = AutoMerge(inter_channel*scale**3,inter_channel,out_channel)
#         self.choose_out=torch.nn.Sequential(
#             torch.nn.Conv3d(2*inter_channel,inter_channel,3,1,1),
#             torch.nn.Conv3d(inter_channel,out_channel,1,1,0),
#         )

#     def forward(self, x):
#         x, x_for_skip=self.down(x)
#         c1 = self.conv1(x)
#         c1p = self.pool1(c1)
#         c2 = self.conv2(c1p)
#         c2p = self.pool2(c2)
#         c3 = self.conv3(c2p)
#         c3p = self.pool3(c3)
#         c4 = self.conv4(c3p)

#         ea1 = self.ea(c4)
#         ea1_p = self.ea_pool1(ea1)
#         ea2 = self.ea(ea1_p)
#         ea2_p = self.ea_pool2(ea2)
#         ea3 = self.ea(ea2_p)

#         up_ea3 = torch.cat((self.up_ea1(ea3), self.squeeze_ea1(ea2)), dim=1)
#         up_ea3 = self.decoder_ea(up_ea3)
#         up_ea2 = torch.cat((self.up_ea2(up_ea3), self.squeeze_ea2(ea1)), dim=1)
#         up_ea2 = self.decoder_ea(up_ea2)

#         up4 = self.up4(torch.cat((up_ea2, c4), dim=1))
#         up4_conv = self.up_conv4(up4)
#         up3 = self.up3(torch.cat((up4_conv, c3), dim=1))
#         up3_conv = self.up_conv3(up3)
#         up2 = self.up2(torch.cat((up3_conv, c2), dim=1))
#         up2_conv = self.up_conv2(up2)

#         out = self.out_conv(torch.cat((up2_conv, c1), dim=1))
#         out = self.up(out)
#         final_out = self.choose_out(torch.cat((out, x_for_skip), dim=1))
#         return torch.sigmoid(final_out)


class VolumeFormer_noshare(torch.nn.Module):
    def __init__(self, in_channel=1, out_channel=1, scale=2):
        super(VolumeFormer_noshare, self).__init__()
        self.down = AutoDecomposition(in_channel,in_channel*scale**3)
        self.conv1 = DoubleConv(in_channel*scale**3, 64)
        self.pool1 = torch.nn.MaxPool3d(2)  #112
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = torch.nn.MaxPool3d(2)  #56
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = torch.nn.MaxPool3d(2)  #28

        self.conv4 = DoubleConv(256, 512) # adding this for no reason

        self.ea1 = EAModule(512)
        self.ea_pool1 = torch.nn.MaxPool3d(2)
        self.ea2 = EAModule(512)
        self.ea_pool2 = torch.nn.MaxPool3d(2)
        self.ea3 = EAModule(512)
        self.ea4 = EAModule(512)
        self.ea5 = EAModule(512)

        self.up_ea1 = torch.nn.ConvTranspose3d(512, 256, 2, stride=2)
        self.squeeze_ea1 = torch.nn.Conv3d(512, 256, 1, 1, 0)
        self.up_ea2 = torch.nn.ConvTranspose3d(512, 256, 2, stride=2)
        self.squeeze_ea2 = torch.nn.Conv3d(512, 256, 1, 1, 0)

        self.up4 = torch.nn.ConvTranspose3d(1024, 512, 2, stride=2)
        self.up_conv4 = DoubleConv(512, 256)
        self.up3 = torch.nn.ConvTranspose3d(512, 256, 2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)
        self.up2 = torch.nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.up_conv2 = DoubleConv(128, 64)
        self.out_conv = torch.nn.Conv3d(128, out_channel*scale**3, 3, 1, 1)  #input 128 because concatenated with c1
        self.up = AutoMerge(out_channel*scale**3,out_channel) #input 128 because concatenated with c1

    def forward(self, x):
        x=self.down(x)
        c1 = self.conv1(x)
        c1p = self.pool1(c1)
        c2 = self.conv2(c1p)
        c2p = self.pool2(c2)
        c3 = self.conv3(c2p)
        c3p = self.pool3(c3)
        c4 = self.conv4(c3p)

        ea1 = self.ea1(c4)
        ea1_p = self.ea_pool1(ea1)
        ea2 = self.ea2(ea1_p)
        ea2_p = self.ea_pool2(ea2)
        ea3 = self.ea3(ea2_p)

        up_ea3 = torch.cat((self.up_ea1(ea3), self.squeeze_ea1(ea2)), dim=1)
        up_ea3 = self.ea4(up_ea3)
        up_ea2 = torch.cat((self.up_ea2(up_ea3), self.squeeze_ea2(ea1)), dim=1)
        up_ea2 = self.ea5(up_ea2)

        up4 = self.up4(torch.cat((up_ea2, c4), dim=1))
        up4_conv = self.up_conv4(up4)
        up3 = self.up3(torch.cat((up4_conv, c3), dim=1))
        up3_conv = self.up_conv3(up3)
        up2 = self.up2(torch.cat((up3_conv, c2), dim=1))
        up2_conv = self.up_conv2(up2)

        out = self.out_conv(torch.cat((up2_conv, c1), dim=1))
        out = self.up(out)
        return nn.Sigmoid()(out)