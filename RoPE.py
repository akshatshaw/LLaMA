# import torch
# class rope(torch.nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.head_dim =  args.dim
#         self.seq_len = args.max_seq_len
#         self.device = args.device
#         self.theta = args.theta
#         self.freqs_complex=self.precompute_theta_pos_frequencies()
        
#     def forward(self,x):
#         return self.apply_rotary_embeddings(x)
    
#     def precompute_theta_pos_frequencies(self):
#         assert self.head_dim % 2 == 0, "Dimension must be divisible by 2"
#         # According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
#         theta_numerator = torch.arange(0, self.head_dim, 2).float()
#         theta = 1.0 / (self.theta ** (theta_numerator / self.head_dim)).to(self.device) 
        
#         # Construct the positions (the "m" parameter)
#         m = torch.arange(self.seq_len, device = self.device)
        
#         # Multiply each theta by each position using the outer product.
#         freqs = torch.outer(m, theta).float()
        
#         # We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
#         freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
#         return freqs_complex

#     def apply_rotary_embeddings(self, x: torch.Tensor):
#         # Separate the last dimension pairs of two values, representing the real and imaginary parts of the complex number
#         # Two consecutive values will become a single complex number
#         # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2)
#         x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
#         # Reshape the freqs_complex tensor to match the shape of the x_complex tensor. So we need to add the batch dimension and the head dimension
#         # (Seq_Len, Head_Dim/2) --> (1, Seq_Len, 1, Head_Dim/2)
#         freqs_complex = self.freqs_complex.unsqueeze(0).unsqueeze(2)
#         # Multiply each complex number in the x_complex tensor by the corresponding complex number in the freqs_complex tensor
#         # Which results in the rotation of the complex number as shown in the Figure 1 of the paper
#         # (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
#         x_rotated = x_complex * freqs_complex
#         # Convert the complex number back to the real number
#         # (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
#         x_out = torch.view_as_real(x_rotated)
#         # (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
#         x_out = x_out.reshape(*x.shape)
#         return x_out.type_as(x).to(self.device)


# import torch

# class rope:
#     def __init__(self, head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
#         self.head_dim = head_dim
#         self.seq_len = seq_len
#         self.device = device
#         self.theta = theta
#         self.freqs_complex = self.precompute_theta_pos_frequencies()

#     def precompute_theta_pos_frequencies(self):
#         assert self.head_dim % 2 == 0, "Dimension must be divisible by 2"
        
#         theta_numerator = torch.arange(0, self.head_dim, 2).float()
#         theta = 1.0 / (self.theta ** (theta_numerator / self.head_dim)).to(self.device)
#         m = torch.arange(self.seq_len, device=self.device)
#         freqs = torch.outer(m, theta).float()
#         freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        
#         return freqs_complex

#     def apply_rotary_embeddings(self, x: torch.Tensor):
#         x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
#         freqs_complex = self.freqs_complex.unsqueeze(0).unsqueeze(2)
#         x_rotated = x_complex * freqs_complex
#         x_out = torch.view_as_real(x_rotated)
#         x_out = x_out.reshape(*x.shape)
#         return x_out.type_as(x).to(self.device)
    


import torch
class rope(torch.nn.Module):
    def __init__(self, head_dim, seq_len, device="cuda", theta=10000):
        super().__init__()
        self.head_dim=head_dim
        self.seq_len=seq_len
        self.device=device
        self.theta=theta
        self.freqs_complex=self.precompute_theta_pos_freq(head_dim,seq_len,device,theta)
    def forward(self,x):
        return self.rotary_pos_embeds(x,self.device)
    def precompute_theta_pos_freq(self,head_dim,seq,device,theta):
        assert head_dim%2==0 #dimension should be divisible by 2
        #theta_i=10000^(-2(i-1)/dim) for i={1,2,...,dim/2}
        theta_num=torch.arange(0,head_dim,2).float()
        theta=1./(theta**(theta_num/head_dim)).to(device)
        #write the m positions
        m=torch.arange(seq,device=device)
        #multiply m with every theta
        freqs=torch.outer(m,theta).float()
        #convert this into complex form (polar)
        freqs_complex=torch.polar(torch.ones_like(freqs),freqs)
        return freqs_complex

    def rotary_pos_embeds(self, x,device):
        x_complex=torch.view_as_complex(x.float().reshape(*x.shape[:-1],-1,2))
        freqs_complex=self.freqs_complex.unsqueeze(0).unsqueeze(1)
        # print('-'*50)
        # print(x_complex.shape,freqs_complex.shape)
        # print('-'*50)
        x_rotated=x_complex*freqs_complex
        x_out=torch.view_as_real(x_rotated)
        x_out=x_out.reshape(*x.shape)
        return x_out.type_as(x).to(device)