#Classical Attention
class Attention(nn.Module):
    def __init__(self, d_model=4,hid_dim=1,o_dim=1,Qbias=False,Vbias=True,Mask=False,OnlyAtt=True):
        super(Attention, self).__init__()
        self.dk=hid_dim
        self.Q=nn.Linear(d_model,d_model,bias=Qbias)
        self.K=nn.Linear(d_model,d_model,bias=Qbias)
        self.V=nn.Linear(d_model,d_model,bias=Vbias)
        self.Mask=Mask
        self.OnlyAtt=OnlyAtt
    def forward(self, src):
        Q=self.Q(src)
       
        K=self.K(src)
    
        O=torch.matmul(Q,torch.transpose(K,dim0=1,dim1=2))/np.sqrt(self.dk)

        if self.Mask:
            #n
            seq_len=src.size(1)
            #Upper triangular Matrix of Trues
            mask = torch.triu(torch.ones(seq_len, seq_len, device=src.device), diagonal=1).bool()

            O=O.masked_fill(mask,float('-inf'))
            
        att=torch.softmax(O,dim=2)
       
        att_out=torch.matmul(att,src)
        
        V=self.V(att_out)
        #V=att_out
        if self.OnlyAtt:
            return V[:,0]
        else:
            return V

import math

#Original Multi-headed Attention Transformer
class MHAttention(nn.Module):
    def __init__(self, d_model=4,Qbias=False,Vbias=True,Mask=False,OnlyAtt=True,n_heads=1):
        super(MHAttention, self).__init__()
        self.dk=d_model//n_heads
        self.Q=nn.Linear(d_model,d_model,bias=Qbias)
        self.K=nn.Linear(d_model,d_model,bias=Qbias)
        self.V=nn.Linear(d_model,d_model,bias=Vbias)
        self.O=nn.Linear(d_model,d_model,bias=False)
        self.Mask=Mask
        self.OnlyAtt=OnlyAtt
        self.n_heads=n_heads
        self.d_model=d_model
    def forward(self, src):
        
        batch_size, seq_len, _ = src.shape

        # Compute Q, K, V
        Q = self.Q(src).view(batch_size, seq_len, self.n_heads, self.dk).transpose(1, 2)  # (B, H, S, D_k)
        K = self.K(src).view(batch_size, seq_len, self.n_heads, self.dk).transpose(1, 2)  
        V = self.V(src).view(batch_size, seq_len, self.n_heads, self.dk).transpose(1, 2)  

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dk)  # (B, H, S, S)

        # Apply mask if needed
        if self.Mask:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=src.device), diagonal=1).bool()
            scores.masked_fill_(mask, float('-inf'))

        # Compute attention weights
        attn = torch.softmax(scores, dim=-1)  # (B, H, S, S)
        
        # Apply attention to values
        context = torch.matmul(attn, V)  # (B, H, S, D_k)

        # Reshape & project output
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)  # (B, S, D)
        output = self.O(context)

        return output if not self.OnlyAtt else output[:, 0]
    
#In Context Learning MHA
class MHA(nn.Module):
    def __init__(self,d_model=1,Qbias=False,Vbias=True,nheads=1):
        super(MHA,self).__init__()
        self.attention_layers = nn.ModuleList([
            Attention(
                d_model=d_model,
                Qbias=Qbias,
                Vbias=Vbias

            )
             for _ in range(nheads)
        ])

    def forward(self,src):
        
        output=torch.zeros(size=[src.shape[0],1]).to(device)
        for head in self.attention_layers:
            output+=head(src)

        return output



class AttentionApproximator(nn.Module):
    def __init__(self,d_model=1,Qbias=False,Vbias=True,i_dim=1,o_dim=1):
        super(AttentionApproximator,self).__init__()
        self.Attention=Attention(d_model,Qbias=Qbias,Vbias=Vbias,OnlyAtt=True)
        self.embedding=nn.Linear(i_dim,d_model)
        self.O=nn.Linear(d_model,o_dim)

    def forward(self,src):
        x=self.embedding(src)
        x=self.Attention(x)
        x=self.O(x)

        return x
    

class MHAttentionApproximator(nn.Module):
    def __init__(self,d_model=1,Qbias=False,Vbias=True,i_dim=1,o_dim=1,n_heads=1):
        super(MHAttentionApproximator,self).__init__()
        self.MHAttention=MHAttention(d_model,Qbias=Qbias,Vbias=Vbias,OnlyAtt=True,n_heads=n_heads)
        self.embedding=nn.Linear(i_dim,d_model)
        self.O=nn.Linear(d_model,o_dim)

    def forward(self,src):
        x=self.embedding(src)
        x=self.MHAttention(x)
        x=self.O(x)

        return x