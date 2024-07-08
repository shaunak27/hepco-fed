from __future__ import print_function
import math
import torch
import torch.nn as nn
from utils.metric import AverageMeter, Timer
import numpy as np
from .default import NormalNN, accumulate_acc, loss_fn_kd, Teacher
import copy
import matplotlib
matplotlib.use("Agg")
import math

class LWF(NormalNN):

    def __init__(self, learner_config):
        super(LWF, self).__init__(learner_config)
        self.previous_teacher = None
        self.replay = False
        self.past_tasks = []
        self.first_task = True
        self.first_block = True
        self.ce_loss = nn.BCELoss()
        self.init_task_param_reg = self.eps > 0

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None,prev_server=None,server=None,loss_type = None,lambda_prox = 0.01):
        
        # L2 from the start
        if self.init_task_param_reg: self.accumulate_block_memory(train_loader)
        
        # init teacher
        if self.replay:
            teacher = Teacher(solver=prev_server.model)
            self.previous_teacher = copy.deepcopy(teacher)

        # try to load model
        need_train = True
        # if not self.overwrite or self.task_count == 0:
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
        
            losses = [AverageMeter() for l in range(3)]
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
                    self.step = i

                    # verify in train mode
                    self.model.train()
                    if math.ceil(y.size(0)/len(self.config['gpuid']))*(len(self.config['gpuid'])-1) >= y.size(0): 
                        print(y.size(0))
                        continue
                    # send data to gpu
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()
                    
                    # if KD
                    if self.replay:
                        allowed_predictions =  [j for sub in self.tasks_real[:self.task_count] for j in sub]
                        y_hat, _ = self.previous_teacher.generate_scores(x, allowed_predictions=allowed_predictions)
                    else:
                        y_hat = None

                    # model update - training data
                    loss, loss_class, loss_distill, output= self.update_model(x, y, y_hat,loss_type = loss_type, server_model = server.model if server is not None else None,lambda_prox = lambda_prox)

                    # measure elapsed time
                    batch_time.update(batch_timer.toc()) 

                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                    losses[0].update(loss,  y.size(0)) 
                    losses[1].update(loss_class,  y.size(0)) 
                    losses[2].update(loss_distill,  y.size(0)) 
                    batch_timer.tic()

                # eval update
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses[0],acc=acc))
                self.log(' * Class Loss {loss.avg:.3f} | KD Loss {lossb.avg:.3f}'.format(loss=losses[1],lossb=losses[2]))

                # Evaluate the performance of current task
                if val_loader is not None:
                    self.validation(val_loader)

                # reset
                losses = [AverageMeter() for l in range(3)]
                acc = AverageMeter()

        self.model.eval()

        
        self.last_last_valid_out_dim = self.last_valid_out_dim
        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset(self.memory_size, np.arange(self.last_valid_out_dim))

        
        self.accumulate_block_memory(train_loader)

        try:
            return batch_time.avg
        except:
            return None

    def accumulate_block_memory(self, train_loader):
        pass

    def update_model(self, inputs, targets, target_KD = None,loss_type = None, server_model = None,lambda_prox = 0.01): 
        
        total_loss = torch.zeros((1,), requires_grad=True).cuda()

        if self.dw:
            dw_cls = self.dw_k[targets.long()]
        else:
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        tasks_till_now = [j for sub in self.tasks_real[:self.task_count+1] for j in sub]
        logits = self.forward(inputs)
        loss_class = self.criterion(logits, targets.long(), dw_cls)
        total_loss += loss_class

        # KD
        if target_KD is not None:
            dw_KD = self.dw_k[-1 * torch.ones(len(target_KD),).long()]
            logits_KD = logits
            loss_distill = loss_fn_kd(logits_KD, target_KD, dw_KD, np.arange(self.last_valid_out_dim).tolist(), self.DTemp)
            total_loss += self.mu * loss_distill
        else:
            loss_distill = torch.zeros((1,), requires_grad=True).cuda()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), loss_class.detach(), loss_distill.detach(), logits

    def combine_data(self, data):
        x, y = [],[]
        for i in range(len(data)):
            x.append(data[i][0])
            y.append(data[i][1])
        x, y = torch.cat(x), torch.cat(y)
        return x, y

    ##########################################
    #             MODEL UTILS                #
    ##########################################

    def combine_data(self, data):
        x, y = [],[]
        for i in range(len(data)):
            x.append(data[i][0])
            y.append(data[i][1])
        x, y = torch.cat(x), torch.cat(y)
        return x, y

class LWF_MC(LWF):

    def __init__(self, learner_config):
        super(LWF_MC, self).__init__(learner_config)
        

    def update_model(self, inputs, targets, target_KD = None,loss_type = None, server_model = None,lambda_prox = 0.01):

        # get output
        logits = self.forward(inputs)

        # class loss      
        if target_KD is not None:
            target_mod = get_one_hot(targets, self.valid_out_dim)
            target_mod[:, :self.last_valid_out_dim] = torch.sigmoid(target_KD)
            total_loss = self.ce_loss(torch.sigmoid(logits), target_mod)
        else:
            target_mod = get_one_hot(targets, self.valid_out_dim)
            total_loss = self.ce_loss(torch.sigmoid(logits), target_mod)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), total_loss.detach(), torch.zeros((1,), requires_grad=True).cuda().detach(), logits


def get_one_hot(target,num_class):
    one_hot=torch.zeros(target.shape[0],num_class).cuda()
    one_hot[range(target.shape[0]), target]=1
    return one_hot