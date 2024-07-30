from __future__ import print_function
import torch
import models
from .default import count_parameters
from .kd import LWF, get_one_hot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.schedulers import CosineSchedule

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1) * 1.0e3  # scale up
        distance_negative = (anchor - negative).pow(2).sum(1) * 1.0e1  # scale up
        # print(distance_positive)
        losses = torch.relu((distance_positive - distance_negative).sum() + self.margin)
        return losses.mean()

class DualPrompt(LWF):

    def __init__(self, learner_config):
        self.prompt_param = learner_config['prompt_param']  ## : prompt param are supplied as cl args
        super(DualPrompt, self).__init__(learner_config)  
        self.triplet_loss = TripletLoss(margin=1)
    # update model - add dual prompt loss   
    def update_model(self, inputs, targets, target_KD = None, loss_type = None, server_model = None, lambda_prox = 0.01,prev_task_model = None):

        # logits
        try :
            logits, prompt_loss = self.model(inputs, train=True) 
        except:
            print(inputs.shape,inputs.device)
        tasks_till_now = [j for sub in self.tasks_real[:self.task_count+1] for j in sub]
        
        total_loss = torch.zeros((1,), requires_grad=True).cuda()
        penalty = torch.tensor(0., requires_grad=True).cuda()
        if loss_type == "fedprox":
            for (n,w), (n_t,w_t) in zip(server_model.named_parameters(),self.model.named_parameters()):
                if 'prompt' in n or 'last' in n:
                    penalty += torch.pow(torch.norm(w.detach() - w_t), 2)
            total_loss += (lambda_prox / 2.) * penalty
        elif loss_type == "triplet":
            with torch.no_grad():
                server_logits, _ = server_model(inputs, train=True)
                prev_task_logits, _ = prev_task_model(inputs, train=True)
            trip_loss = self.triplet_loss(logits, server_logits, prev_task_logits)
            total_loss += trip_loss
        # ce loss
        logits = logits[:,tasks_till_now]

        # standard ce
        logits[:,:self.last_valid_out_dim] = -float('inf')
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss += self.criterion(logits, targets.long(), dw_cls)
        
        total_loss = total_loss + self.mu * prompt_loss.sum()
        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), prompt_loss.sum().detach(), torch.zeros((1,), requires_grad=True).cuda().detach(), logits

    # sets model optimizers
    def init_optimizer(self):

        # parse optimizer args
        try :
            params_to_opt = list(self.model.module.prompt.parameters()) 
            if not self.config['freeze_last']:
                params_to_opt += list(self.model.module.last.parameters())
        except : 
            params_to_opt = list(self.model.prompt.parameters())
            if not self.config['freeze_last']:
                params_to_opt += list(self.model.last.parameters())

        print('*****************************************')
        print('*****************************************')
        print('*****************************************')
        print('Num param opt: ' + str(count_parameters(params_to_opt)))
        print('*****************************************')
        print('*****************************************')
        print('*****************************************')
        optimizer_arg = {'params':params_to_opt,
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        elif self.config['optimizer'] == 'Adam':
            optimizer_arg['betas'] = (self.config['momentum'],0.999)

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        
        # create schedules
        if self.schedule_type == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])
        elif self.schedule_type == 'decay':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=0.1)

    def create_model(self):
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet, CLIP ... etc) of model
        if 'clip' in cfg['model_name']:
            model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'dual', prompt_param=self.prompt_param,template_style=cfg['template_style']) ## : Jump to vit_pt_imnet in zoo_old
        else:
            model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'dual', prompt_param=self.prompt_param)
        return model

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        # self.model.prompt = self.model.prompt.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self


class L2P(DualPrompt):

    def __init__(self, learner_config):
        super(L2P, self).__init__(learner_config) ## : Jump to DualPrompt Initialization

    def create_model(self):
        cfg = self.config
        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        if 'clip' in cfg['model_name']:
            model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'l2p',prompt_param=self.prompt_param,template_style=cfg['template_style']) ## : Jump to vit_pt_imnet in zoo_old
        else:
            model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'l2p',prompt_param=self.prompt_param,prompt_type = cfg['prompt_type']) ## : Jump to vit_pt_imnet in zoo_old
        return model


class Finetune(LWF):

    def __init__(self, learner_config):
        super(Finetune, self).__init__(learner_config)

    def update_model(self, inputs, targets, target_KD = None):

        # get output
        logits = self.forward(inputs)

        # standard ce
        logits[:,:self.last_valid_out_dim] = -float('inf')
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), total_loss.detach(), torch.zeros((1,), requires_grad=True).cuda().detach(), logits

class Linear(Finetune):

    def __init__(self, learner_config):
        super(Linear, self).__init__(learner_config)

    # sets model optimizers
    def init_optimizer(self):

        # parse optimizer args
        print('*****************************************')
        print('*****************************************')
        print('*****************************************')
        print('Num param opt: ' + str(count_parameters(self.model.module.last.parameters())))
        print('*****************************************')
        print('*****************************************')
        print('*****************************************')
        optimizer_arg = {'params':self.model.module.last.parameters(),
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        elif self.config['optimizer'] == 'Adam':
            optimizer_arg['betas'] = (self.config['momentum'],0.999)

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        
        # create schedules
        if self.schedule_type == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])
        elif self.schedule_type == 'decay':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=0.1)


