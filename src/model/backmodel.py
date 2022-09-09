import torch
from torch import nn


class FM(nn.Module):
    '''
    FM (Factorization Machine)
    '''
    def __init__(self, w1, v, point_wise=True):
        super().__init__()
        self.point_wise = point_wise
        self.W0 = torch.randn(1)
        self.W1 = w1 # W1: (n, 1) first-order linear part embedding
        self.V = v   # V: (n, embedding_size) second-order interaction embedding
    
    def forward(self, input):
        # input: (batch_size, n), n is the num of fields
        linear_part = self.W1(input).sum(dim=-2) + self.W0 # (batch_size, 1)
        x = self.V(input) # (batch_size, n, embedding_size)
        interaction_part_1 = torch.pow(x.sum(dim=-2), 2) # (batch_size, embedding_size)
        interaction_part_2 = torch.pow(x, 2).sum(dim=-2)
        product_part = interaction_part_1 - interaction_part_2
        if self.point_wise:
            product_part = product_part.sum(dim=-1, keepdim=True) # (batch_size, 1)
        output = linear_part + 0.5 * product_part
        return output


class PNN(nn.Module):
    '''
    PNN: embedding layer + product layer
    '''
    def __init__(self, w1, v):
        super().__init__()
        self.fm = FM(w1, v, point_wise=False)
        self.V = v
    
    def forward(self, input):
        embed_input = self.V(input).sum(dim=-2) # (batch_size, embedding_size)
        fm_input = self.fm(input) # (batch_size, embedding_size)
        embed_len = torch.sqrt(torch.pow(embed_input, 2).sum(dim=-1, keepdim=True) + 1e-8)
        fm_len = torch.sqrt(torch.pow(fm_input, 2).sum(dim=-1, keepdim=True) + 1e-8)
        fm_input = fm_input / fm_len * embed_len
        output = torch.cat([fm_input, embed_input], dim=-1)
        return output


if __name__ == "__main__":
    n, k = 5, 3
    w1 = nn.Embedding.from_pretrained(torch.randn(n, 1))
    v = nn.Embedding.from_pretrained(torch.randn(n, k))
    pnn = PNN(w1, v)
    input = torch.rand(10, 10) * n
    input = input.int()
    output = pnn(input)
    print(output)