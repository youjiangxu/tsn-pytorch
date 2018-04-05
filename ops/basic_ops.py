import torch
import math
from torch.autograd import Variable

class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.autograd.Function):

    def __init__(self, consensus_type, dim=1):
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output

    def backward(self, grad_output):
        if self.consensus_type == 'avg':
            grad_in = grad_output.expand(self.shape) / float(self.shape[self.dim])
        elif self.consensus_type == 'identity':
            grad_in = grad_output
        else:
            grad_in = None

        return grad_in


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim)(input)



class VideoSeqvlad(torch.autograd.Function):

    def __init__(self, timesteps, num_centers, redu_dim=None):
        '''
            num_centers: set the number of centers for sevlad
            redu_dim: reduce channels for input tensor
        '''
        self.num_centers = num_centers
        self.redu_dim = redu_dim
        self.timesteps = timesteps

        self.in_shape = None
        self.out_shape = self.num_centers*self.redu_dim
        self.batch_size = None
        print('## in SeqVLADModule ##',self.num_centers, self.redu_dim)
        


        self.centers = torch.Tensor(self.num_centers, self.redu_dim) # weight : out, in , h, w
        self.centers = torch.nn.init.xavier_uniform(self.centers, gain=1) 
        self.centers = torch.nn.Parameter(self.centers, requires_grad=True)

        self.share_w = torch.Tensor(self.num_centers, self.redu_dim, 1, 1) # weight : out, in , h, w
        self.share_w = torch.nn.init.xavier_normal(self.share_w, gain=1) 
        self.share_w = torch.nn.Parameter(self.share_w, requires_grad=True)

        self.share_b = torch.Tensor(self.num_centers,) # weight : out, in , h, w
        self.share_b = torch.nn.init.uniform(self.share_b) 
        self.share_b = torch.nn.Parameter(self.share_b, requires_grad=True)


        self.U_r = torch.Tensor(self.num_centers, self.num_centers, 3, 3) # weight : out, in , h, w
        self.U_r = torch.nn.init.xavier_normal(self.U_r, gain=1) 
        self.U_r = torch.nn.Parameter(self.U_r, requires_grad=True)

        self.U_z = torch.Tensor(self.num_centers, self.num_centers, 3, 3) # weight : out, in , h, w
        self.U_z = torch.nn.init.xavier_normal(self.U_z, gain=1) 
        self.U_z = torch.nn.Parameter(self.U_z, requires_grad=True)

        self.U_h = torch.Tensor(self.num_centers, self.num_centers, 3, 3) # weight : out, in , h, w
        self.U_h = torch.nn.init.xavier_normal(self.U_h, gain=1) 
        self.U_h = torch.nn.Parameter(self.U_h, requires_grad=True)


    def forward(self, input_tensor):
        '''
        input_tensor: N*timesteps, C, H, W
        '''
        self.in_shape = input_tensor.size()
        self.batch_size = self.in_shape[0]//self.timesteps


        print(self.in_shape)
        if self.redu_dim == None:
            self.redu_dim = self.in_shape[1]
        elif self.redu_dim < self.in_shape[1]:
            print('## reduction dim ##')
            self.redu_conv = torch.nn.conv(self.in_shape[1], self.redu_dim, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
            self.redu_relu = torch.nn.relu(inplace=True)

            input_tensor = self.redu_relu(self.redu_conv(input_tensor))

        self.out_shape = self.num_centers*self.redu_dim


        ## wx_plus_b : N*timesteps, redu_dim, H, W
        wx_plus_b = torch.nn.functional.conv2d(input_tensor, self.share_w, bias=self.share_b, stride=1, padding=1, dilation=1, groups=1) 
        wx_plus_b = wx_plus_b.view(self.batch_size, self.timesteps, self.num_centers, self.in_shape[2], self.in_shape[3])
        ## reshape 


        ## init hidden states
        ## h_tm1 = N, num_centers, H, W
        h_tm1 = torch.Tensor(self.batch_size, self.num_centers, self.in_shape[2], self.in_shape[3])
        h_tm1 = torch.nn.init.constant_(h_tm1, 0) 


        ## prepare the input tensor shape
        ## output
        assignments = []

        for i in range(self.timesteps):
            wx_plus_b_at_t = wx_plus_b[:,i,:,:,:]

            Uz_h = torch.nn.functional.conv2d(h_tm1, self.U_z, bias=None, stride=1, padding=1) 
            z = torch.nn.functional.sigmoid(wx_plus_b_at_t+Uz_h)

            Ur_h = torch.nn.functional.conv2d(h_tm1, self.U_r, bias=None, stride=1, padding=1) 
            r = torch.nn.functional.sigmoid(wx_plus_b_at_t+Ur_h)

            Uh_h = torch.nn.functional.conv2d(r*h_tm1, self.U_h, bias=None, stride=1, padding=1)
            hh = torch.nn.functional.tanh(wx_plus_b_at_t+Uh_h)

            h = (1 - z) * hh + z*h_tm1
            assignments.append(h)
            h_tm1 = h

        ## timesteps, batch_size , num_centers, h, w

        assignments = torch.stack(assignments, axis=0)
        print('assignments shape', assignments.size())

        ## batch_size, timesteps, num_centers, h, w
        assignments = torch.transpose(assignments, 0, 1) 
        print('transposed assignments shape', assignments.size())

        ## assignments: batch_size, timesteps, num_centers, h*w
        assignments = assignments.view(self.batch_size*self.timesteps, self.num_centers, self.in_shape[2]*self.in_shape[3])

        ## alpha *c 
        ## a_sum: batch_size, timesteps, num_centers, 1
        a_sum = torch.sum(assignments, -1, keepdim=True)

        ## a: batch_size*timesteps, num_centers, redu_dim
        a = a_sum * self.centers.view(1, self.num_centers, self.redu_dim)

        ## alpha* input_tensor
        ## fea_assign: batch_size, timesteps, num_centers, h, w ==> batch_size*timesteps, num_centers, h*w 
        # fea_assign = assignments.view(self.batch_size*self.timesteps, self.num_centers, self.in_shape[2]*self.in_shape[3])

        ## input_tensor: batch_size, timesteps, redu_dim, h, w  ==> batch_size*timesteps, redu_dim, h*w  ==>  batch_size*timesteps, h*w, redu_dim 
        input_tensor = input_tensor.view(self.batch_size*self.timesteps, self.redu_dim, self.in_shape[2]*self.in_shape[3])
        input_tensor = torch.transpose(input_tensor, 1, 2)

        ## x: batch_size*timesteps, num_centers, redu_dim
        x  = torch.matmul(assignments, input_tensor)


        ## batch_size*timesteps, num_centers, redu_dim
        vlad = x - a 

        ## batch_size*timesteps, num_centers, redu_dim ==> batch_size, timesteps, num_centers, redu_dim
        vlad = vlad.view(self.batch_size, self.timesteps, self.num_centers, self.redu_dim)

        ## batch_size, num_centers, redu_dim 
        vlad = torch.sum(vlad, 1, keepdim=False)

        ## intor normalize
        vlad = torch.nn.functional.normalize(vlad, p=2, dim=2)

        ## l2-normalize
        vlad = vlad.view(self.batch_size, self.num_centers*self.redu_dim)
        vlad = torch.nn.functional.normalize(vlad, p=2, dim=1)
        return input_tensor

    def backward(self, grad_output):
        # if self.consensus_type == 'avg':
        #     grad_in = grad_output.expand(self.shape) / float(self.shape[self.dim])
        # elif self.consensus_type == 'identity':
        #     grad_in = grad_output
        # else:
        #     grad_in = None

        grad_in = grad_output

        return grad_in

class SeqVLADModule(torch.nn.Module):

    def __init__(self, timesteps, num_centers, redu_dim=None):
        '''
            num_centers: set the number of centers for sevlad
            redu_dim: reduce channels for input tensor
        '''
        super(SeqVLADModule, self).__init__()
        self.num_centers = num_centers
        self.redu_dim = redu_dim
        self.timesteps = timesteps
        print('## in SeqVLADModule ##',self.num_centers, self.redu_dim)


    def forward(self, input_tensor):
        return VideoSeqvlad(self.timesteps, self.num_centers, self.redu_dim)(input_tensor)


