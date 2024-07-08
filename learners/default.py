from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import models
from utils.metric import accuracy, AverageMeter, Timer
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy
from utils.schedulers import CosineSchedule


class NormalNN(nn.Module):
    '''
    Normal Neural Network with SGD for classification
    '''
    def __init__(self, learner_config):

        super(NormalNN, self).__init__()
        self.log = print
        self.config = learner_config
        self.out_dim = learner_config['out_dim'] 
        self.model = self.create_model()
        self.reset_optimizer = True
        self.overwrite = learner_config['overwrite']
        self.batch_size = learner_config['batch_size']
        self.previous_teacher = None
        self.tasks = learner_config['tasks']
        self.tasks_real = learner_config['tasks_real']
        self.top_k = learner_config['top_k']

        # replay memory parameters
        self.memory_size = self.config['memory']
        self.task_count = 0

        # class balancing
        self.dw = self.config['DW']
        if self.memory_size <= 0:
            self.dw = False

        # distillation
        self.DTemp = learner_config['temp']
        self.mu = learner_config['mu']
        self.beta = learner_config['beta']
        self.eps = learner_config['eps']

        # supervised criterion
        self.criterion_fn = nn.CrossEntropyLoss(reduction='none')
        
        # cuda gpu
        if learner_config['gpuid'][0] >= 0:
            self.cuda()
            self.gpu = True
        else:
            self.gpu = False
        
        # highest class index from past task
        self.last_valid_out_dim = 0 

        # highest class index from current task
        self.valid_out_dim = 0

        # set up schedules
        self.schedule_type = self.config['schedule_type']
        self.schedule = self.config['schedule']

        # initialize optimizer
        self.init_optimizer()

    ##########################################
    #           MODEL TRAINING               #
    ##########################################

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        
        # try to load model
        need_train = True
        # if not self.overwrite:
        if True:
            try:
                self.load_model(model_save_dir)
                need_train = False
            except:
                pass

        # trains
        if need_train:
            if self.reset_optimizer:  # Reset optimizer before learning each task
                self.log('Optimizer is reset!')
                self.init_optimizer()

            # data weighting
            self.data_weighting(train_dataset)
            
            # Evaluate the performance of current task
            self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=0,total=self.config['schedule'][-1]))
            if val_loader is not None:
                self.validation(val_loader)
        
            losses = AverageMeter()
            acc = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()
            for epoch in range(self.config['schedule'][-1]):
                self.epoch=epoch

                if epoch > 0: self.scheduler.step()
                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
                batch_timer.tic()
                for i, (x, y, task)  in enumerate(train_loader):

                    # verify in train mode
                    self.model.train()
                    # send data to gpu
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()
                    
                    # model update
                    loss, output= self.update_model(x, y)
                    # measure elapsed time
                    batch_time.update(batch_timer.toc())  
                    batch_timer.tic()
                    
                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                    losses.update(loss,  y.size(0)) 
                    batch_timer.tic()

                # eval update
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses,acc=acc))

                # Evaluate the performance of current task
                if val_loader is not None:
                    self.validation(val_loader)

                # reset
                losses = AverageMeter()
                acc = AverageMeter()
                
        self.model.eval()

        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset(self.memory_size, np.arange(self.last_valid_out_dim))

        # for eval
        if self.previous_teacher is not None:
            self.previous_previous_teacher = self.previous_teacher

        # new teacher
        teacher = Teacher(solver=self.model)
        self.previous_teacher = copy.deepcopy(teacher)

        try:
            return batch_time.avg
        except:
            return None

    def criterion(self, logits, targets, data_weights):
        """The loss criterion with any additional regularizations added
        The inputs and targets could come from single task 
        The network always makes the predictions with all its heads
        The criterion will match the head and task to calculate the loss.
        Parameters
        ----------
        logits : dict(torch.Tensor)
            Dictionary of predictions, e.g. outs from `forward`
        targets : torch.Tensor
            target labels
        Returns
        -------
        torch._Loss :
            the loss function with any modifications added
        """
        # loss_supervised = (self.criterion_fn(logits, targets.long()) * data_weights).mean()
        loss_supervised = self.criterion_fn(logits, targets.long()).mean()
        return loss_supervised 

    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        
        if dw_force is not None:
            dw_cls = dw_force
        elif self.dw:
            dw_cls = self.dw_k[targets.long()]
        else:
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        logits = self.forward(inputs)
        #past_tasks = [j for sub in self.tasks_real[:self.task_count] for j in sub]
        # print(logits.shape) 
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # : replace to params_to_opt ??
        self.optimizer.step()
        return total_loss.detach(), logits

    ##########################################
    #             MODEL EVAL                 #
    ##########################################


    def validation(self, dataloader, model=None, task_in = None, task_metric='acc', relabel_clusters = True, verbal = True, cka_flag = -1, task_global=False,t_idx=None):

        if model is None:
            if task_metric == 'acc':
                model = self.model
            else:
                return -1

        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        orig_mode = model.training
        model.eval()
        for i, (input, target, task) in enumerate(dataloader):

            if math.ceil(target.shape[0]/len(self.config['gpuid']))*(len(self.config['gpuid'])-1) >= target.shape[0]: 
                    print(target.shape[0])
                    continue
            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()
            if task_in is None:
                if t_idx is not None:
                    tasks_till_now = [j for sub in self.tasks_real[:t_idx+1] for j in sub]
                else:
                    tasks_till_now = [j for sub in self.tasks_real[:self.task_count+1] for j in sub]
                output = model.forward(input)[:, tasks_till_now]
                acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
            else:
                mask = target >= task_in[0]
                mask_ind = mask.nonzero().view(-1) 
                input, target = input[mask_ind], target[mask_ind]

                mask = target < task_in[-1]
                mask_ind = mask.nonzero().view(-1) 
                input, target = input[mask_ind], target[mask_ind]
                
                if len(target) > 1:
                    if cka_flag > -1:
                        output = model.forward(input)[:, :cka_flag]
                        acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
                    else:
                        if task_global:
                            tasks_till_now = [j for sub in self.tasks_real[:self.task_count] for j in sub]
                            output = model.forward(input)[:, tasks_till_now]
                            acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
                        else:
                            #tasks_till_now = [j for sub in self.tasks_real[:self.task_count] for j in sub]
                            if t_idx is not None:
                                output = model.forward(input)[:, self.tasks_real[t_idx]]
                            else:
                                output = model.forward(input)[:, self.tasks_real[self.task_count-1]]
                            acc = accumulate_acc(output, target-task_in[0], task, acc, topk=(self.top_k,))
            
        model.train(orig_mode)

        if verbal:
            self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
                    .format(acc=acc, time=batch_timer.toc()))
        return acc.avg

    def label_clusters(self, dataloader):
        pass

    ##########################################
    #             MODEL UTILS                #
    ##########################################

    # data weighting
    def data_weighting(self, dataset, num_seen=None):

        self.dw_k = torch.tensor(np.ones(self.valid_out_dim + 1, dtype=np.float32))
        # cuda
        if self.cuda:
            self.dw_k = self.dw_k.cuda()

    def save_model(self, filename):
        model_state = self.model.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        self.log('=> Saving class model to:', filename)
        torch.save(model_state, filename + 'class.pth')
        self.log('=> Save Done')

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename + 'class.pth'))
        self.log('=> Load Done')
        if self.gpu:
            self.model = self.model.cuda()
        self.model.eval()

    def load_model_other(self, filename, model):
        model.load_state_dict(torch.load(filename + 'class.pth'))
        if self.gpu:
            model = model.cuda()
        return model.eval()


    # sets model optimizers
    def init_optimizer(self):
        if self.config['freeze_encoder']:
            try :
                params_to_opt = list(self.model.module.last.parameters())
            except :
                params_to_opt = list(self.model.last.parameters())
        else:
            params_to_opt = list(self.model.parameters())
        # parse optimizer args
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
        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        if 'clip' in cfg['model_name'] :
            model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim,template_style=cfg['template_style']) ## : For L2P, jump to zoo_old/vit_pt_imnet
        else :
            model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim) ## : For L2P, jump to zoo_old/vit_pt_imnet

        return model

    def print_model(self):
        self.log(self.model)
        self.log('#parameter of model:', self.count_parameter())
    
    def reset_model(self):
        self.model.apply(weight_reset)

    def forward(self, x):
        tasks_till_now = [j for sub in self.tasks_real[:self.task_count+1] for j in sub]
        return self.model.forward(x)[:, tasks_till_now]

    def predict(self, inputs):
        self.model.eval()
        out = self.forward(inputs)
        return out
    
    def add_valid_output_dim(self, dim=0):
        # This function is kind of ad-hoc, but it is the simplest way to support incremental class learning
        self.log('Incremental class: Old valid output dimension:', self.valid_out_dim)
        self.valid_out_dim += dim
        self.log('Incremental class: New Valid output dimension:', self.valid_out_dim)
        return self.valid_out_dim

    def count_parameter(self):
        return sum(p.numel() for p in self.model.parameters())   

    def count_memory(self, dataset_size):
        return self.count_parameter() + self.memory_size * dataset_size[0]*dataset_size[1]*dataset_size[2]

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            print('Using multiple GPUs')
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.log("Running on:", device)
        return device

    def pre_steps(self):
        pass

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def accumulate_acc(output, target, task, meter, topk):
    current_acc = accuracy(output, target, topk)
    #print(f'Current Accuracy : ',current_acc)
    meter.update(current_acc, len(target))
    return meter

def loss_fn_kd(scores, target_scores, data_weights, allowed_predictions, T=2., soft_t = False):
    """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].
    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""


    log_scores_norm = F.log_softmax(scores[:, allowed_predictions] / T, dim=1)
    if soft_t:
        targets_norm = target_scores
    else:
        targets_norm = F.softmax(target_scores[:, allowed_predictions] / T, dim=1)

    # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
    KD_loss_unnorm = -(targets_norm * log_scores_norm)
    KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)                      #--> sum over classes
    # KD_loss_unnorm = KD_loss_unnorm * data_weights                  # data weighting
    KD_loss_unnorm = KD_loss_unnorm.mean()                          #--> average over batch

    # normalize
    KD_loss = KD_loss_unnorm # * T**2

    return KD_loss

##########################################
#            TEACHER CLASS               #
##########################################

class Teacher(nn.Module):

    def __init__(self, solver):

        super().__init__()
        self.solver = solver

    def generate_scores(self, x, allowed_predictions=None, threshold=None):

        # set model to eval()-mode
        mode = self.training
        self.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x)
        y_hat = y_hat[:, allowed_predictions]

        # set model back to its initial mode
        self.train(mode=mode)

        # threshold if desired
        if threshold is not None:
            # get predicted class-labels (indexed according to each class' position in [allowed_predictions]!)
            y_hat = F.softmax(y_hat, dim=1)
            ymax, y = torch.max(y_hat, dim=1)
            thresh_mask = ymax > (threshold)
            thresh_idx = thresh_mask.nonzero().view(-1)
            y_hat = y_hat[thresh_idx]
            y = y[thresh_idx]
            return y_hat, y, x[thresh_idx]

        else:
            # get predicted class-labels (indexed according to each class' position in [allowed_predictions]!)
            ymax, y = torch.max(y_hat, dim=1)

            return y_hat, y

    def generate_scores_pen(self, x):

        # set model to eval()-mode
        mode = self.training
        self.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x=x, pen=True)

        # set model back to its initial mode
        self.train(mode=mode)

        return y_hat

    def generate_scores_layer(self, x, layer):

        # set model to eval()-mode
        mode = self.training
        self.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x=x, pen=True, l = layer)

        # set model back to its initial mode
        self.train(mode=mode)

        return y_hat

def count_parameters(params):
    return sum(p.numel() for p in params if p.requires_grad)