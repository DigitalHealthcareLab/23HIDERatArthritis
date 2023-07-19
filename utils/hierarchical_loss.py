'''Hierarchical Loss Network
'''


import torch
import torch.nn as nn
import torch



hierarchyL1= {
0:[0],
1:[1],
2:[2,3],
3:[4],
4:[5],
5:[6,7],
6:[8],
7:[9],
8:[10.11],
9:[12],
10:[13],
11:[14,15],
}

hierarchyL2= {
0:[0],
1:[1,2],
2:[3],
3:[4,5],
4:[6],
5:[7,8],
6:[9],
7:[10,11],
}

class HierarchicalLossNetwork:
    '''Logics to calculate the loss of the model.
    '''
    def __init__(self,hierarchical_labels, device='cpu', total_level=3, alpha=1, beta=0.8, p_loss=3):
        '''Param init.
        '''
        self.total_level = total_level

        self.alpha = alpha
        self.beta = beta
        self.p_loss = p_loss
        self.device = device
        self.L1_labels = list(range(8))
        self.L2_labels = list(range(12))
        self.L3_labels = list(range(16))

        self.hierarchical_labels = hierarchical_labels
        self.numeric_hierarchy = self.words_to_indices()

    def words_to_indices(self):
        '''Convert the classes from words to indices.
        '''
        numeric_hierarchy = {}
        for k, v in self.hierarchical_labels.items():
            numeric_hierarchy[self.L1_labels.index(k)] = {}
            for k2, v2 in v.items():
                numeric_hierarchy[self.L1_labels.index(k)][self.L2_labels.index(k2)] = [self.L3_labels.index(i) for i in v2]
        print("NH",numeric_hierarchy)
        return numeric_hierarchy

        
    def check_hierarchy(self,current_lvl_pred, prev_lvl_pred, prev_prev_lvl_pred):
        bool_tensor = []
        print(current_lvl_pred)
        print(prev_lvl_pred)
        print(prev_prev_lvl_pred)

        for i in range(prev_prev_lvl_pred.size(0)):
            if prev_lvl_pred[i].item() in hierarchyL2[prev_prev_lvl_pred[i].item()] and current_lvl_pred[i].item() in hierarchyL1[prev_lvl_pred[i].item()]  :
                bool_tensor.append(False)
            else:
                bool_tensor.append(True)
        return torch.FloatTensor(bool_tensor).to(self.device)


    def calculate_lloss(self, predictions, true_labels):
        '''Calculates the layer loss.
        '''
        lloss = 0

        for l in range(self.total_level):

            lloss += nn.CrossEntropyLoss()(predictions[l], true_labels[l])
            
        return 1 * lloss

    def calculate_dloss(self, predictions, true_labels):
        '''Calculate the dependence loss.
        '''

        dloss = 0
        for l in range(2, self.total_level):

            current_lvl_pred = torch.argmax(nn.Softmax(dim=1)(predictions[l]), dim=1)
            prev_lvl_pred = torch.argmax(nn.Softmax(dim=1)(predictions[l-1]), dim=1)
            prev_prev_lvl_pred = torch.argmax(nn.Softmax(dim=1)(predictions[l-2]), dim=1)

            D_l = self.check_hierarchy(current_lvl_pred, prev_lvl_pred, prev_prev_lvl_pred)
            print("ppp",prev_prev_lvl_pred)
            print("tl",true_labels)
            l_prev_prev = torch.where(prev_prev_lvl_pred == true_labels[l-2], torch.FloatTensor([0]).to(self.device), torch.FloatTensor([1]).to(self.device))
            l_prev = torch.where(prev_lvl_pred == true_labels[l-1], torch.FloatTensor([0]).to(self.device), torch.FloatTensor([1]).to(self.device))
            l_curr = torch.where(current_lvl_pred == true_labels[l], torch.FloatTensor([0]).to(self.device), torch.FloatTensor([1]).to(self.device))

            dloss += torch.sum(torch.pow(self.p_loss, D_l*l_prev_prev)*torch.pow(self.p_loss, D_l*l_prev)*torch.pow(self.p_loss, D_l*l_curr) - 1)

        return self.beta * dloss


            
    class ArgMax(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            idx = torch.argmax(input, 1)

            output = torch.zeros_like(input)
            output.scatter_(1, idx, 1)
            
            return output
        

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output