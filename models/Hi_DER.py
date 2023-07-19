import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import DERNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
from utils.hierarchical_loss import HierarchicalLossNetwork
import torch 



hierarchy=  {
    0: {
        0 : [0]},
    1: {
        1 : [1],
        2 : [2,3]},
    2: {
        3 : [4]},
    3: {
        4 : [5],
        5 : [6,7]},
    4: {
        6 : [8]},
    5: {
        7 : [9],
        8 : [10,11]},
    6: {
        9 : [12]},
    7: {
        10 : [13],
        11 : [14,15]},
}

init_epoch = 100
init_lr = 0.0001
init_milestones = [50]
init_lr_decay = 0.1
init_weight_decay = 0.0005


epochs = 100
lrate = 0.0001
milestones = [50]
lrate_decay = 0.1
batch_size = 32
weight_decay = 2e-4
num_workers = 32
T = 2

EPSILON = 1e-8


class HiDER(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = DERNet(args, False)
        #self._network= torch.nn.DataParallel(DERNet(args, False))
    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test",
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self._L1_total_classes = 8
        self._L2_total_classes = 12 #max(set(t)) + 1
        self._network.update_fc(self._total_classes,self._L2_total_classes,self._L1_total_classes)

        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        if self._cur_task > 0:
            for i in range(self._cur_task):
                for p in self._network.convnets[i].parameters():
                    p.requires_grad = False

        logging.info("All params: {}".format(count_parameters(self._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(self._network, True))
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def train(self):
        self._network.train()
        if len(self._multiple_gpus) > 1 :
            self._network_module_ptr = self._network.module
        else:
            self._network_module_ptr = self._network
        self._network_module_ptr.convnets[-1].train()
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                self._network_module_ptr.convnets[i].eval()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            optimizer = Adam(filter(lambda p: p.requires_grad,self._network.parameters()), lr=init_lr,weight_decay=init_weight_decay,)
      
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = Adam(filter(lambda p: p.requires_grad,self._network.parameters()), lr=init_lr,weight_decay=init_weight_decay,)

            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
            if len(self._multiple_gpus) > 1:
                self._network.module.weight_align(
                    self._total_classes - self._known_classes
                )
            else:
                self._network.weight_align(self._total_classes - self._known_classes)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
    
        prog_bar = tqdm(range(init_epoch))
        
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            correct, total = 0, 0
       

            for i, (_, inputs, L3,L2,L1) in enumerate(train_loader):
                inputs, L3,L2,L1 = inputs.to(self._device), L3.to(self._device), L2.to(self._device), L1.to(self._device)
                
                out,level1,level2,level3= self._network(inputs)
                logits = out["logits"]
                
                HLN = HierarchicalLossNetwork(hierarchical_labels=hierarchy, device=self._device)

                L1_pred,L2_pred,L3_pred = level1,level2,level3
            
                prediction = [L1_pred,L2_pred,L3_pred]
                true_labels= [L1,L2,L3]


                dloss = HLN.calculate_dloss(prediction, true_labels)
                lloss = HLN.calculate_lloss(prediction, true_labels)

                loss = lloss + dloss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()     
                l3preds = torch.argmax(nn.Softmax(dim=1)(prediction[2]), dim=1)
                l2preds = torch.argmax(nn.Softmax(dim=1)(prediction[1]), dim=1)
                l1preds = torch.argmax(nn.Softmax(dim=1)(prediction[0]), dim=1)
                correct += l3preds.eq(L3.expand_as(l3preds)).cpu().sum()
                total += len(L3)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
    
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            losses_clf = 0.0
            losses_aux = 0.0
            correct, total = 0, 0
            for i, (_, inputs, L3,L2,L1) in enumerate(train_loader):
                inputs, L3,L2,L1 = inputs.to(self._device), L3.to(self._device), L2.to(self._device),L1.to(self._device)
                out,level1,level2,level3= self._network(inputs)
                logits, aux_logits = out["logits"], out["aux_logits"]
               
                HLN = HierarchicalLossNetwork(hierarchical_labels=hierarchy, device=self._device)


                L1_pred,L2_pred,L3_pred = level1, level2,level3
                prediction = [L1_pred,L2_pred,L3_pred]
                true_labels= [L1,L2,L3]

                
                dloss = HLN.calculate_dloss(prediction, true_labels)
                lloss = HLN.calculate_lloss(prediction, true_labels)
                
                loss_clf= lloss + dloss

                aux_L3 = L3.clone()
                aux_L3 = torch.where(
                    aux_L3 - self._known_classes + 1 > 0,
                    aux_L3 - self._known_classes + 1,
                    0,
                )
                loss_aux = F.cross_entropy(aux_logits, aux_L3)

                loss = loss_clf + loss_aux

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_aux += loss_aux.item()
                losses_clf += loss_clf.item()
                l3preds = torch.argmax(nn.Softmax(dim=1)(prediction[2]), dim=1)
                l2preds = torch.argmax(nn.Softmax(dim=1)(prediction[1]), dim=1)
                l1preds = torch.argmax(nn.Softmax(dim=1)(prediction[0]), dim=1)
             
                correct += l3preds.eq(L3.expand_as(l3preds)).cpu().sum()
                total += len(L3)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
 
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_aux / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_aux / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)