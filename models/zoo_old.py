import torch
import torch.nn as nn
from .vit import VisionTransformer

def tensor_prompt(a, b, c=None):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    nn.init.uniform_(p)
    return p

class Generator(nn.Module):
    #mlp with 2 hidden layers
    def __init__(self,noise_dim=32,num_classes=200):
        super(Generator, self).__init__()
        self.embedding = nn.Embedding(num_classes, noise_dim) #(num_classes, embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(2*noise_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 768),
        )
    def forward(self, y,eps):
        z = self.embedding(y)
        z = torch.cat((eps, z), 1)
        return self.net(z)

class DualPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768,prompt_type='l2p'):
        super().__init__()
        self.task_count_f = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.expand_and_freeze = False
        self.n_tasks = n_tasks
        self.prompt_type = prompt_type
        print('prompt_type : ',prompt_type)
        self._init_smart(emb_d, prompt_param)

        # init frequency table
        for e in self.e_layers:
            setattr(self, f'freq_curr_{e}',torch.nn.Parameter(torch.zeros(self.e_pool_size,), requires_grad=False))
            setattr(self, f'freq_past_{e}',torch.nn.Parameter(torch.zeros(self.e_pool_size,), requires_grad=False))

        # g prompt init
        for g in self.g_layers:
            p = tensor_prompt(self.g_p_length, emb_d)
            setattr(self, f'g_p_{g}',p)

        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)

    def _init_smart(self, emb_d, prompt_param):
        
        self.top_k = 1
        self.task_id_bootstrap = True

        # prompt locations
        self.g_layers = [0,1]
        self.e_layers = [2,3,4]

        # prompt pool size
        self.g_p_length = prompt_param[2]
        self.e_p_length = prompt_param[1]
        self.e_pool_size = prompt_param[0]

    def process_frequency(self):
        self.task_count_f += 1
        if not self.task_id_bootstrap:
            for e in self.e_layers:
                f_ = getattr(self, f'freq_curr_{e}')
                f_ = f_ / torch.sum(f_)
                setattr(self, f'freq_past_{e}',torch.nn.Parameter(f_, requires_grad=False))


    def forward(self, x_querry, l, x_block, train=False, task_id=None, hepco=False):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape 
            
            if self.expand_and_freeze:
                K = getattr(self,f'e_k_{l}')
                p = getattr(self,f'e_p_{l}')

                # freeze/control past tasks
                pt = self.e_pool_size / self.n_tasks
                s = int(self.task_count_f * pt)
                f = int((self.task_count_f + 1) * pt)
                
                if train:
                    if self.task_count_f > 0:
                        K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                        p = torch.cat((p[:s].detach().clone(),p[s:f]), dim=0)
                    else:
                        K = K[s:f]
                        p = p[s:f]
                else:
                    K = K[0:f]
                    p = p[0:f]
                
            else:
                K = getattr(self,f'e_k_{l}') 
                p = getattr(self,f'e_p_{l}') 
            

            # cosine similarity to match keys/querries
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_querry, dim=1).detach()
            cos_sim = torch.einsum('bj,kj->bk', q, n_K)
            
            if train:

                # prompting
                if self.task_id_bootstrap:
                    loss = 1.0 - cos_sim[:,task_id].mean()  # the cosine similarity is always le 1
                    P_ = p[task_id].expand(len(x_querry),-1,-1)
                else:
                    if self.task_count_f > 0:
                        f_ = getattr(self, f'freq_past_{l}')
                        f_tensor = f_.expand(B,-1)
                        cos_sim_scaled = cos_sim
                    else:
                        cos_sim_scaled = cos_sim
                    top_k = torch.topk(cos_sim_scaled, self.top_k, dim=1)
                    k_idx = top_k.indices
                    loss = 0
                    if self.prompt_type == 'l2p':
                        loss = (1.0 - cos_sim[:,k_idx]).sum()
                        P_ = p[k_idx][:,0] # replace with p[k_idx] to choose top k
                    elif self.prompt_type =='weighted_l2p' :
                        cos_sim_scaled = cos_sim_scaled.unsqueeze(2).unsqueeze(3) 
                        P_ = torch.sum(torch.mul(cos_sim_scaled,p),dim=1)

                    # update frequency
                    f_ = getattr(self, f'freq_curr_{l}')
                    f_to_add = torch.bincount(k_idx.flatten().detach(),minlength=self.e_pool_size)
                    f_ += f_to_add
            else:
                top_k = torch.topk(cos_sim, self.top_k, dim=1)
                if self.prompt_type == 'l2p':
                    k_idx = top_k.indices
                    P_ = p[k_idx][:,0] 
                elif self.prompt_type =='weighted_l2p' :
                    cos_sim = cos_sim.unsqueeze(2).unsqueeze(3) 
                    P_ = torch.sum(torch.mul(cos_sim,p),dim=1)
                
            # select prompts
            i = int(self.e_p_length/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]
        # g prompts
        g_valid = False
        if l in self.g_layers:
            g_valid = True
            j = int(self.g_p_length/2)
            p = getattr(self,f'g_p_{l}') # 0 based indexing here
            P_ = p.expand(len(x_querry),-1,-1)
            Gk = P_[:,:j,:]
            Gv = P_[:,j:,:]

        # combine prompts for prefix tuning
        if e_valid and g_valid:
            Pk = torch.cat((Ek, Gk), dim=1)
            Pv = torch.cat((Ev, Gv), dim=1)
            p_return = [Pk, Pv]
        elif e_valid:
            p_return = [Ek, Ev]
        elif g_valid:
            p_return = [Gk, Gv]
            loss = 0
        else:
            p_return = None
            loss = 0

        if hepco:
            if train:
                return P_
            else:
                return P_
        # return
        if train:
            return p_return, loss, x_block #p_return, loss, x_block  : uncomment
        else:
            return p_return, 0, x_block

class L2P(DualPrompt):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768,prompt_type='l2p'):
        super().__init__(emb_d, n_tasks, prompt_param, key_dim,prompt_type)

    def _init_smart(self, emb_d, prompt_param):
        self.top_k = 5
        self.task_id_bootstrap = False

        # prompt locations
        self.g_layers = []
        if prompt_param[2] > 0:
            self.e_layers = [0,1,2,3,4]
        else:
            self.e_layers = [0]

        if prompt_param[3] == 3:
            self.expand_and_freeze = True


        # prompt pool size
        self.g_p_length = -1
        self.e_p_length = prompt_param[1]
        self.e_pool_size = prompt_param[0]





class ResNetZoo(nn.Module):
    def __init__(self, num_classes=10, pt=False, mode=1, prompt_flag=False, prompt_param=None,prompt_type='l2p'):
        super(ResNetZoo, self).__init__()

        # get last layer
        self.last = nn.Linear(512, num_classes)
        self.prompt_flag = prompt_flag
        self.task_id = None

        # get feature encoder
        if mode == 0:
            if pt:
                zoo_model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                            num_heads=12, ckpt_layer=0,
                                            drop_path_rate=0
                                            )
                from timm.models import vit_base_patch16_224
                load_dict = vit_base_patch16_224(pretrained=True).state_dict()
                del load_dict['head.weight']; del load_dict['head.bias']
                zoo_model.load_state_dict(load_dict)

                self.last = nn.Linear(768, num_classes)

        # create prompting module
        if self.prompt_flag == 'l2p':
            self.prompt = L2P(768, prompt_param[0], prompt_param[1],prompt_type=prompt_type) 
            

        elif self.prompt_flag == 'dual':
            self.prompt = DualPrompt(768, prompt_param[0], prompt_param[1])

        else:
            self.prompt = None
        
        self.feat = zoo_model
        

    def forward(self, x, pen=False, train=False,z=None):

        if z is not None:
            out = self.last(z)
            return out
        if self.prompt is not None:
            with torch.no_grad():
                q, _ = self.feat(x)
                q = q[:,0,:]
            out, prompt_loss = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id)
            out = out[:,0,:]
        else:
            out, _ = self.feat(x)
            out = out[:,0,:]
        out = out.view(out.size(0), -1)
        if not pen:
            out = self.last(out)
        if self.prompt is not None and train:
            return out, prompt_loss
        else:
            return out
            
def vit_pt_imnet(out_dim, block_division = None, prompt_flag = 'None', prompt_param=None,prompt_type='l2p'):
    return ResNetZoo(num_classes=out_dim, pt=True, mode=0, prompt_flag=prompt_flag, prompt_param=prompt_param, prompt_type = prompt_type) ## : Jump to ResnetZoo init