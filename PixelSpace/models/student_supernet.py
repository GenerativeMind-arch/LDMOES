import torch.nn as nn
import torch
from torch.nn import functional as F
from models.diffusion import *
import numpy as np
import random
import copy



# search space
KERNEL_SIZE = [1,3,5]
CHANNEL_NUM = [0.5, 0.8, 1.0, 1.2]


max_ch_mul = CHANNEL_NUM[-1]

#保证通道数是32的倍数
def round_ch_num(ch_num):
    new_ch_num = round(ch_num/32) *32
    return new_ch_num

'''
ResBlock with variable kernel size
'''

class MixResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,dropout, temb_channels=512,):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = Normalize(in_channels)
        self.conv1 = MixOps(max_in_ch= in_channels, max_out_ch = round_ch_num(out_channels*max_ch_mul) )
        # self.temb_proj = torch.nn.Linear(temb_channels, int(out_channels*max_ch_mul) )
        # self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.temb_proj = torch.nn.Linear(temb_channels, round_ch_num(max_ch_mul * out_channels))
        # TODO: group norm with variable channel nums, search channel
        # self.norm2 = Normalize(out_channels)
        self.norm2_mixed = nn.ModuleList()
        for ch_num in CHANNEL_NUM:
            self.norm2_mixed.append(Normalize(round_ch_num(out_channels * ch_num)))

        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = MixOps(max_in_ch= round_ch_num(out_channels*max_ch_mul), max_out_ch=out_channels)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0)

    # forward index = (kernel_size, channel_num)
    def forward(self, x, temb, forward_op =None):

        if forward_op is None:
            # forward_op = np.random.randint(0,len(KERNEL_SIZE), 2)
            kernel_index = list(np.random.randint(0,len(KERNEL_SIZE), 2))
            ch_num_index = list(np.random.randint(0,len(CHANNEL_NUM), 1)) *2  #，它被乘以 2 (* 2)，所以最终 ch_num_index 的长度是 2，但两个值的大小应该是一样的
            forward_op = list(zip(kernel_index, ch_num_index))

        # ch_num = random.choice(CHANNEL_NUM)
        # ch_num = 1.0
        ch_num_index = forward_op[0][1]
        ch_num = CHANNEL_NUM[ch_num_index]

        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        in_ch = self.in_channels
        out_ch = round_ch_num(self.out_channels * ch_num)
        # h = self.conv1(h, forward_index = forward_op[0], in_ch=in_ch, out_ch=out_ch)
        h = self.conv1(h, forward_index = forward_op[0][0], in_ch=in_ch, out_ch=out_ch)


        h = h + self.temb_proj(nonlinearity(temb))[:, 0:out_ch, None, None]  #None 填充的操作是为了广播
        # h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        # h = self.norm2(h)
        h = self.norm2_mixed[ch_num_index](h)
        h = nonlinearity(h)
        h = self.dropout(h)

        in_ch = out_ch
        out_ch = self.out_channels
        # h = self.conv2(h, forward_index = forward_op[1], in_ch=in_ch, out_ch=out_ch)
        h = self.conv2(h, forward_index = forward_op[1][0], in_ch=in_ch, out_ch=out_ch)
       

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class MixOps(nn.Module):
    def __init__(self, max_out_ch, max_in_ch):
        super(MixOps, self).__init__()
        self._mix_ops = nn.ModuleList()
        for k_size in KERNEL_SIZE:
            ops =  ChConv(k_size, max_in_ch, max_out_ch)
            self._mix_ops.append(ops)
    
    def forward(self, x, forward_index=0, in_ch=0, out_ch=0):
        # Single-path
        return self._mix_ops[forward_index](x, in_ch, out_ch)
        # k_size_index, ch_num_index = forward_index
        # return self._mix_ops[k_size_index](x, in_ch, out_ch)


class ChConv(nn.Module):
    def __init__(self, kernel_size, max_in_ch, max_out_ch):
        super(ChConv, self).__init__()
        self.max_in_ch = max_in_ch
        self.max_out_ch = max_out_ch
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - 1) // 2
        self.conv = nn.Conv2d(max_in_ch,max_out_ch, kernel_size=self.kernel_size, stride=1, padding=self.padding)

    def forward(self, x, in_ch, out_ch):
        w = self.conv.weight[0:out_ch,0:in_ch,:,:]
        b = self.conv.bias[0:out_ch]
        out = F.conv2d(x, w, b, stride=1, padding=self.padding) 
        return out


class SuperStudent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv
        num_timesteps = config.diffusion.num_diffusion_timesteps
        
        if config.model.type == 'bayesian':
            self.logvar = nn.Parameter(torch.zeros(num_timesteps))
        
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.layers = 0
        self.stage_num =  2*len(config.model.ch_mult) +2

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,self.temb_ch),
            torch.nn.Linear(self.temb_ch,self.temb_ch),])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,self.ch,kernel_size=3,stride=1,padding=1)
        curr_res = resolution
        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(MixResnetBlock(in_channels=block_in,out_channels=block_out,  temb_channels=self.temb_ch,dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = MixResnetBlock(in_channels=block_in,out_channels=block_in,temb_channels=self.temb_ch,dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = MixResnetBlock(in_channels=block_in,out_channels=block_in,temb_channels=self.temb_ch,dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(MixResnetBlock(in_channels=block_in+skip_in,out_channels=block_out,temb_channels=self.temb_ch,dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,out_ch,kernel_size=3,stride=1,padding=1)

    def forward(self, x, t, guides=None, guide_hs=None, start=0, forward_op =None, sample=False):
        if sample:
            return self.foward_all_stage(x, t, forward_op=forward_op)
        else:
            # train
            return self.forward_stage(x, t, guides, guide_hs, start, forward_op)

    #trian:
    def forward_stage(self, x, t, guides=None, guide_hs=None,start=0, forward_op =None):

        assert x.shape[2] == x.shape[3] == self.resolution
        block_idx = 0
        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        outputs = []
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):

            if block_idx < start:
                block_idx += 1
                continue
            else:
                if i_level !=0:
                    hs[-1] = self.down[i_level-1].downsample( guides[block_idx-1] )
                for i_block in range(self.num_res_blocks):
                    if forward_op is not None:
                        h = self.down[i_level].block[i_block](hs[-1], temb, forward_op=forward_op[2*i_block:2*(i_block+1)])
                    else:
                        h = self.down[i_level].block[i_block](hs[-1], temb)
                    if len(self.down[i_level].attn) > 0:
                        h = self.down[i_level].attn[i_block](h)
                    hs.append(h)

                # guided features after the block <stage>
                outputs.append(h)
                return outputs

                # if i_level != self.num_resolutions-1:
                #     hs.append(self.down[i_level].downsample(hs[-1]))
       
        # middle
        # h = hs[-1]
        if block_idx == start:
            h = guides[block_idx-1]
            if forward_op is not None:
                h = self.mid.block_1(h, temb, forward_op[0:2])
            else:
                h = self.mid.block_1(h, temb)
            h = self.mid.attn_1(h)
            if forward_op is not None:  
                h = self.mid.block_2(h, temb, forward_op[2:4])
            else:
                h = self.mid.block_2(h, temb)
            outputs.append(h)
            return outputs

        block_idx += 1 
            
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            if block_idx < start:
                block_idx += 1
                continue
            else:
                h = guides[block_idx-1]
                if i_level != self.num_resolutions -1:
                    h = self.up[i_level+1].upsample(h)
                for i_block in range(self.num_res_blocks+1):
                    # h = self.up[i_level].block[i_block](
                    #     torch.cat([h, hs.pop()], dim=1), temb)
                    hs_idx = -1-( (self.num_resolutions-1-i_level)*(self.num_res_blocks+1)  + i_block)
                    if forward_op is not None:
                        h = self.up[i_level].block[i_block](torch.cat([h, guide_hs[hs_idx]], dim=1), temb, 
                                                            forward_op=forward_op[2*i_block:2*(i_block+1)])
                    else:
                        h = self.up[i_level].block[i_block](torch.cat([h, guide_hs[hs_idx]], dim=1), temb)
                    if len(self.up[i_level].attn) > 0:
                        h = self.up[i_level].attn[i_block](h)
                outputs.append(h)
                return outputs
                # if i_level != 0:
                #     h = self.up[i_level].upsample(h)

        assert block_idx == start
        # end
        h = guides[block_idx-1]
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        outputs.append(h)

        return outputs

    #sample:
    def foward_all_stage(self, x, t, forward_op =[]):
        assert x.shape[2] == x.shape[3] == self.resolution

        num_res_blocks= self.config.model.num_res_blocks
        num_resolutions = len(self.config.model.ch_mult)
        layer_num = [num_res_blocks*2] * num_resolutions + [2*2] + [(num_res_blocks+1)*2] * num_resolutions + [0]
        assert len(forward_op) == sum(layer_num)
        def pop_forward_op(forward_op):
            a = forward_op.pop(0)
            b = forward_op.pop(0)
            return [a,b]

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb, forward_op=pop_forward_op(forward_op))
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

       # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb, forward_op=pop_forward_op(forward_op))
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb, forward_op=pop_forward_op(forward_op))

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb, forward_op=pop_forward_op(forward_op))
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        assert len(forward_op) == 0
        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h



class StandAloneNet(nn.Module):
    def __init__(self, config, forward_op):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv
        num_timesteps = config.diffusion.num_diffusion_timesteps
        
        if config.model.type == 'bayesian':
            self.logvar = nn.Parameter(torch.zeros(num_timesteps))
        
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.layers = 0
        self.stage_num =  2*len(config.model.ch_mult) +2

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,self.temb_ch),
            torch.nn.Linear(self.temb_ch,self.temb_ch),])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,self.ch,kernel_size=3,stride=1,padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        block_in = None

        self.forward_op = copy.deepcopy(forward_op)
        def pop_forward_op(forward_op):
            a = forward_op.pop(0)
            b = forward_op.pop(0)
            # return [ KERNEL_SIZE[a],KERNEL_SIZE[b] ]
            return [a,b]
        # down
        for i_level in range(self.num_resolutions): 
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                
                block.append(ResnetBlock(in_channels=block_in,out_channels=block_out,temb_channels=self.temb_ch,dropout=dropout,kernel_sizes=pop_forward_op(forward_op) ))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,out_channels=block_in,temb_channels=self.temb_ch,dropout=dropout,kernel_sizes=pop_forward_op(forward_op))
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,out_channels=block_in,temb_channels=self.temb_ch, dropout=dropout,kernel_sizes=pop_forward_op(forward_op))

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,out_channels=block_out,temb_channels=self.temb_ch,dropout=dropout,kernel_sizes=pop_forward_op(forward_op) ))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,out_ch,kernel_size=3,stride=1,padding=1)


    def forward(self, x, t, guides=None, guide_hs=None, sample=False, only_last=False):
        if sample or only_last:
            return self.foward_all_stage(x, t)
        else:
            # train
            return self.forward_stage_para(x, t, guides, guide_hs)
        # return self.foward_all_stage(x, t)


    def forward_stage(self, x, t, guides=None, guide_hs=None,start=0):

        assert x.shape[2] == x.shape[3] == self.resolution
        block_idx = 0
        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        outputs = []
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            if block_idx < start:
                block_idx += 1
                continue
            else:
                if i_level !=0:
                    hs[-1] = self.down[i_level-1].downsample( guides[block_idx-1] )
                for i_block in range(self.num_res_blocks):
                    h = self.down[i_level].block[i_block](hs[-1], temb)
                    if len(self.down[i_level].attn) > 0:
                        h = self.down[i_level].attn[i_block](h)
                    hs.append(h)

                # guided features after the block <stage>
                outputs.append(h)
                return outputs

        #middle
        if block_idx == start:
            h = guides[block_idx-1]
            h = self.mid.block_1(h, temb)
            h = self.mid.attn_1(h)
            h = self.mid.block_2(h, temb)
            outputs.append(h)
            return outputs

        block_idx += 1 
            
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            if block_idx < start:
                block_idx += 1
                continue
            else:
                h = guides[block_idx-1]
                if i_level != self.num_resolutions -1:
                    h = self.up[i_level+1].upsample(h)
                for i_block in range(self.num_res_blocks+1):
                    # h = self.up[i_level].block[i_block](
                    #     torch.cat([h, hs.pop()], dim=1), temb)
                    hs_idx = -1-( (self.num_resolutions-1-i_level)*(self.num_res_blocks+1)  + i_block)
                    h = self.up[i_level].block[i_block](torch.cat([h, guide_hs[hs_idx]], dim=1), temb)
                    if len(self.up[i_level].attn) > 0:
                        h = self.up[i_level].attn[i_block](h)
                outputs.append(h)
                return outputs

        assert block_idx == start
        # end
        h = guides[block_idx-1]
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        outputs.append(h)

        return outputs

    #
    def forward_stage_para(self, x, t, guides=None, guide_hs=None):
    
        assert x.shape[2] == x.shape[3] == self.resolution
        block_idx = 0
        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        outputs = []
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):

            if i_level != 0:
                hs.append( self.down[i_level-1].downsample( guides[block_idx-1]) )
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)

            # guided features after the block <stage>
            outputs.append(h)
            block_idx += 1

        #middle
        h = guides[block_idx-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        outputs.append(h)
        block_idx += 1

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            h = guides[block_idx-1]
            if i_level != self.num_resolutions -1:
                h = self.up[i_level+1].upsample(h)
            for i_block in range(self.num_res_blocks+1):
                hs_idx = -1-( (self.num_resolutions-1-i_level)*(self.num_res_blocks+1)  + i_block)
                h = self.up[i_level].block[i_block](torch.cat([h, guide_hs[hs_idx]], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            outputs.append(h)
            block_idx += 1
        
        # end
        h = guides[block_idx-1]
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        outputs.append(h)
        block_idx += 1
      
        return outputs

    def foward_all_stage(self, x, t):
        assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h


    def inherit_from_teacher(self, teacher):

        def pop_forward_op(forward_op):
            a = forward_op.pop(0)
            b = forward_op.pop(0)
            return [a,b]


        def inherit(net_a, net_b, ops):
            if ops[0] ==1:
                net_a.conv1.load_state_dict(net_b.conv1.state_dict())
            if ops[1] ==1:
                net_a.conv2.load_state_dict(net_b.conv2.state_dict())


        forward_op = copy.deepcopy(self.forward_op)

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                inherit(self.down[i_level].block[i_block], teacher.down[i_level].block[i_block],
                        ops =pop_forward_op(forward_op))

        inherit(self.mid.block_1, teacher.mid.block_1, ops =pop_forward_op(forward_op))
        inherit(self.mid.block_2, teacher.mid.block_2, ops =pop_forward_op(forward_op))


        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                inherit(self.up[i_level].block[i_block], teacher.up[i_level].block[i_block],
                        ops =pop_forward_op(forward_op))
        