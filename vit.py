
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
  def __init__(self,img_size,patch_size,in_channels,embed_dim):
    super().__init__()
    self.patch_size=patch_size
    self.proj=nn.Conv2d(in_channels=in_channels,out_channels=embed_dim,kernel_size=patch_size,stride=patch_size)
    num_patches=(img_size//patch_size) ** 2
    self.cls_token=nn.Parameter(torch.randn(1,1,embed_dim))
    self.pos_embed=nn.Parameter(torch.randn(1,1+num_patches,embed_dim))
  def forward(self, x:torch.Tensor):
    B=x.size(0)
    x=self.proj(x)
    x=x.flatten(2).transpose(1,2)  
    cls_token=self.cls_token.expand(B, -1, -1) 
    x=torch.cat((cls_token,x),dim=1)
    x=x+self.pos_embed
    return x 


class MLP(nn.Module):
  def __init__(self,in_features,hidden_features,drop_rate):
    super().__init__()
    self.fc1=nn.Linear(in_features=in_features,out_features=hidden_features)
    self.fc2=nn.Linear(in_features=hidden_features,out_features=in_features)
    self.dropout=nn.Dropout(drop_rate)

  def forward(self,x):
      x=self.dropout(F.gelu(self.fc1(x)))
      x=self.dropout(self.fc2(x))
      return x


class TransformerEncoderLayer(nn.Module):
  def __init__(self,embed_dim,num_heads,mlp_dim,drop_rate):
    super().__init__()
    self.norm1=nn.LayerNorm(embed_dim)
    self.attn=nn.MultiheadAttention(embed_dim,num_heads,dropout=drop_rate,batch_first=True)
    self.norm2=nn.LayerNorm(embed_dim)
    self.mlp=MLP(embed_dim,mlp_dim,drop_rate)
  def forward(self,x):
    norm_x=self.norm1(x)
    x=x+self.attn(norm_x,norm_x,norm_x)[0]
    x=x+self.mlp(self.norm2(x))
    return x 



class VisionTransformer(nn.Module):
  def __init__(self,img_size,patch_size,in_channels,num_classes,embed_dim,num_heads,depth,mlp_dim,drop_rate):
    super().__init__()
    self.patch_embed=PatchEmbedding(img_size,patch_size,in_channels,embed_dim)
    self.encoder=nn.Sequential(
        *[TransformerEncoderLayer(embed_dim,num_heads,mlp_dim,drop_rate) for _ in range(depth)]
    ) 
    self.norm=nn.LayerNorm(embed_dim)
    self.head=nn.Linear(embed_dim,num_classes)
  def forward(self,x):
    x=self.patch_embed(x)
    x=self.encoder(x)
    x=self.norm(x)
    cls_token=x[:,0]
    return self.head(cls_token)  
