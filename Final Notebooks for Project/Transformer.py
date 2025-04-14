
import math
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
class Attention(nn.Module):
    def __init__(self, d_model=4,hid_dim=1,Qbias=False,Vbias=True,Mask=False,OnlyAtt=True):
        super(Attention, self).__init__()
        self.dk=hid_dim
        self.Q=nn.Linear(d_model,hid_dim,bias=Qbias)
        self.K=nn.Linear(d_model,hid_dim,bias=Qbias)
        self.V=nn.Linear(d_model,hid_dim,bias=Vbias)
        self.O=nn.Linear(hid_dim,d_model)
        self.Mask=Mask
        self.OnlyAtt=OnlyAtt
    def forward(self, src):
        Q=self.Q(src)
       
        K=self.K(src)
    
        O1=torch.matmul(Q,torch.transpose(K,dim0=1,dim1=2))/np.sqrt(self.dk)

        mask = torch.ones_like(O1, dtype=torch.bool)  # Initialize mask as all True (allowed)
        mask[:, :, 0] = False  # First token does not attend to any token
    
        # Apply the mask by setting attention scores to a very low value where masked
        O1.masked_fill_(~mask, float('-inf'))  
       
        att=torch.softmax(O1,dim=2)

        att_out=torch.matmul(att,src)
        
        V=self.V(att_out)
        #V=att_out
        if self.OnlyAtt:
            return V[:,0]
        else:
            return self.O(V)


class MHA(nn.Module):
    def __init__(self,d_model=1,Qbias=False,Vbias=True,n_heads=1,hid_dim=1,Mask=False,OnlyAtt=False):
        super(MHA,self).__init__()
        self.attention_layers = nn.ModuleList([
            Attention(
                d_model=d_model,
                Qbias=Qbias,
                Vbias=Vbias,
                Mask=Mask,
                OnlyAtt=OnlyAtt,
                hid_dim=hid_dim

            )
             for _ in range(n_heads)
        ])

    def forward(self,src):
        
        output=torch.zeros(size=src.shape).to(device)
        
        for head in self.attention_layers:
            output+=head(src)

        return src+output
    


class MLP(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(MLP,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,output_dim)
        )
    def forward(self,x):
        return self.model(x)

class TLayerSpec(nn.Module):
    def __init__(self,d_model,Qbias=False,Vbias=True,n_heads=1,dimff=8,d_out=1,mask=None,hid_dim=1):
        super(TLayerSpec,self).__init__()
        self.attention=MHA(d_model,Qbias=Qbias,Vbias=Vbias,n_heads=n_heads,hid_dim=hid_dim)
        self.mlp=MLP(d_model,dimff,d_out)

    def forward(self,src):
        attn_out=self.attention(src)
        n1=src+attn_out
        mlp1=self.mlp(n1)
        return mlp1
    
class MyTransformerSpec(nn.Module):
    def __init__(self,
                 Qbias=False,Vbias=True,n_heads=1,
                 layers=1,
                 dimFeedForward=[[1,16,1]],hid_dim=1):
        super(MyTransformerSpec,self).__init__()
        self.tlayers= nn.ModuleList([
            TLayerSpec(d_model=dimFeedForward[i][0],
                Qbias=Qbias,Vbias=Vbias,n_heads=n_heads,
                dimff=dimFeedForward[i][1],d_out=dimFeedForward[i][2],hid_dim=hid_dim)
                for i in range(layers)
        ])
        self.layers=layers
        

    def forward(self, x, X):
        # Concatenate x (external token) with token matrix X along dimension 1.
        # x: (batch_size, 1, d_model) and X: (batch_size, seq_len, d_model)
        ce = torch.cat([x, X], dim=1).to(torch.float32)  # Now shape: (batch_size, total_seq_len, d_model)
        
   
        # Pass the mask into each transformer layer
        for i in range(self.layers):
            ce = self.tlayers[i](ce)
        
        return ce[:,0,:]
   