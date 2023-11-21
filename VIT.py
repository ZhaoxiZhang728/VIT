# Created by zhaoxizh@unc.edu at 15:42 2023/11/18 using PyCharm

import torch.nn as nn
import torch
import torch.nn.functional as F
class Patch_embedding(nn.Module):
    def __init__(self, patch_shape:tuple, channels,hidden_dim):
        super().__init__()


        self.conv = nn.Conv2d(in_channels=channels,
                              out_channels=hidden_dim,
                              stride=patch_shape,
                              kernel_size=patch_shape
        )

    def forward(self,X) -> torch.Tensor:
        # the input size of X should be [batch, channel, height,width]
        # returning size is [batch, number of patches, hidden_channel]
        return torch.flatten(self.conv(X), start_dim=2, end_dim=3).transpose(1,2)


class Mulit_head_attention(nn.Module):
    def __init__(self, num_heads,num_input,num_output):
        super().__init__()
        self.num_heads = num_heads
        self.weight_q = nn.Linear(num_input,num_output,bias=True)
        self.weight_k = nn.Linear(num_input,num_output,bias=True)
        self.weight_v = nn.Linear(num_input,num_output,bias=True)

        # for output of attention
        self.weight_o = nn.Linear(num_input,num_output,bias=True)

    def forward(self,queries,keys,values,valid_lens):
        queries = self.transpose_qkv(self.weight_q(queries))
        keys = self.transpose_qkv(self.weight_k(keys))
        values =self.transpose_qkv(self.weight_v(values))

        output = self.attention(queries,keys,values,valid_lens)

        return self.transpose_output(output)


    def attention(self,q,k,v,valid_lens):
        k = torch.transpose(input=k,dim0=1,dim1=2)
        attenion_score = torch.bmm(q, k) / torch.sqrt(valid_lens)

        attenion_weight = F.softmax(attenion_score,dim=-1)

        return torch.bmm(attenion_weight,v)

    def transpose_qkv(self,X):
        # X.shape [batch_size,number of patch, number of features]

        X = X.reshape(X.shape[0],X.shape[1],self.num_heads,-1)
        # X.shape [batch_size,number of patch,number of heads, number of features]

        X = X.permute(0,2,1,3)
        # X.shape [batch_size,number of heads,number of patch, number of features]

        # X.shape [batch_size * number of heads,number of patch, number of features]
        return X.reshape(-1,X.shape[2],X.shape[3])

    def transpose_output(self, X):
        """Reverse the operation of transpose_qkv."""
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        #[batch_size,number of patch, number of features]
        return X.reshape(X.shape[0], X.shape[1], -1)



class Vit_Block(nn.Module):
    def __init__(self,hidden_dim,num_heads,valid_lens):
        super().__init__()
        self.valid_lens = valid_lens
        self.norm = nn.LayerNorm(hidden_dim)
        self.attention = Mulit_head_attention(num_heads,hidden_dim,hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.Dropout(0.5),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim,hidden_dim)
        )

    def forward(self,x):
        x = self.norm(x)
        output = self.attention(x,x,x,self.valid_lens)
        output = output + x
        y = output
        output = self.norm(output)
        output = y+ self.mlp(output)

        return output

class Vit(nn.Module):
    def __init__(self,img_shape:tuple, patch_shape:tuple, channels,hidden_dim,dim,num_class,num_heads,block_num):
        super().__init__()

        self.num_patch = (img_shape[0] // patch_shape[0]) * (img_shape[1] // patch_shape[1])
        self.patch_size = patch_shape
        self.cls_token = nn.Parameter(torch.randn(1,1,hidden_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1,self.num_patch+1,hidden_dim)) # .repeat(batch_size,1,1)

        self.patch_embedding = Patch_embedding(
                                               patch_shape = patch_shape,
                                               channels = channels,
                                               hidden_dim=hidden_dim
                                               )
        vit_blocks = []
        for _ in range(block_num):
            vit_blocks.append(Vit_Block(hidden_dim,num_heads,torch.tensor(self.patch_size[0])))

        self.module = nn.Sequential(*vit_blocks)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim*(self.num_patch+1),dim),
            nn.GELU(),
            nn.Linear(dim,num_class)
        )

    def forward(self,img):

        re = self.patch_embedding(img)
        cls_tokens = self.cls_token.repeat(img.shape[0], 1, 1)

        re = torch.concatenate((re,cls_tokens),dim=1)

        re = re + self.pos_embedding

        a_re = self.module(re)
        flatten_re = torch.flatten(a_re,start_dim=1,end_dim=2)

        return self.mlp(flatten_re)


if __name__ == '__main__':
    img = torch.rand((10,3,256,256))
    vit = Vit(img_shape=(256,256),patch_shape=(32,32),channels=3,hidden_dim=128,dim=84,num_class=10,num_heads=16,block_num=10)
    re = vit(img)
    print(re.shape)