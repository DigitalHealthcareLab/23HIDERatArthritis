import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist

EPSILON = 1e-8
batch_size = 32


hierarchyL3=  {
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

hierarchyL22= {
0:0,
1:1,
2:1,
3:2,
4:3,
5:3,
6:4,
7:5,
8:5,
9:6,
10:7,
11:7,
}
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
hierarchyL12= {
0: 0,
1: 1,
2: 2,
3: 2,
4: 3,
5: 4,
6: 5,
7: 5,
8: 6,
9: 7,
10: 8,
11: 8,
12: 9,
13: 10,
14: 11,
15: 11,
}

hierarchyL31=  {
    0: {
        0 : [0]},
    1: {
        1 : [1,2],
        2 : [3,4]},
    2: {
        3 : [5]},
    3: {
        4 : [6,7],
        5 : [8,9]},
    4: {
        6 : [10]},
    5: {
        7 : [11,12],
        8 : [13,14]},
    6: {
        9 : [15]},
    7: {
        10 : [16,17],
        11 : [18,19]},
}
hierarchyL21= {
0:[0],
1:[1,2],
2:[3],
3:[4,5],
4:[6],
5:[7,8],
6:[9],
7:[10,11],
}

hierarchyL11= {
0:[0],
1:[1,2],
2:[3,4],
3:[5],
4:[6,7],
5:[8,9],
6:[10],
7:[11,12],
8:[13.14],
9:[15],
10:[16,17],
11:[18,19],
}



class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._L3_memory, self._L2_memory, self._L1_memory = np.array([]), np.array([]),np.array([]),np.array([])
        self.topk = 2

        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"][0]
        self._device1 = args["device"][1]
        self._device2 = args["device"][2]
        self._device3 = args["device"][3]
        self._multiple_gpus = args["device"]

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._L3_memory
        ), "Exemplar size error."
        return len(self._L3_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            
            #total_samples = sum([len(self._data_memory[i]) for i in range(self._total_classes)])
            #return total_samples // self._total_classes
            
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def build_rehearsal_memory(self, data_manager, per_class):
        per_class = self.samples_per_class
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))
        torch.save(self._network,"{}_{}.pt".format(filename, self._cur_task))




    def after_task(self):
        pass

    def _evaluate(self, L3_pred, L3_true):
        ret = {}
        grouped = accuracy(L3_pred.T[0], L3_true, self._known_classes)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (L3_pred.T == np.tile(L3_true, (self.topk, 1))).sum() * 100 / len(L3_true),
            decimals=2,
        )

        return ret

 

    def eval_task(self):

        L3_pred, L3_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(L3_pred, L3_true)

        if hasattr(self, "_class_means"):
            L3_pred, L3_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(L3_pred, L3_true)
        else:
            nme_accy = None
        
        return cnn_accy, nme_accy


    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._L3_memory,self._L2_memory,self._L1_memory)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, L3,L2,L1) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs,level0,level1,level2 = model(inputs)
                L1_pred,L2_pred,L3_pred = level0,level1,level2
            
                prediction = [L1_pred,L2_pred,L3_pred]
                true_labels= [L1,L2,L3]
                outputs=outputs["logits"]
            predicts=torch.argmax(nn.Softmax(dim=1)(prediction[2]), dim=1)
            #predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == L3).sum()
            total += len(L3)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def calculate_accuracy(predictions, labels):
        '''Calculates the accuracy of the prediction.
        '''

        num_data = labels.size()[0]
        predicted = torch.argmax(predictions, dim=1)

        correct_pred = torch.sum(predicted == labels)

        accuracy = correct_pred*(100/num_data)

        return accuracy.item()
   
    def _eval_cnn(self, loader):
        w=10
        self._network.eval()
        L3_pred, L3_true = [], []
        L2_pred, L2_true = [], []
        L1_pred, L1_true = [], []
        y_inputs=[]
        for _, (_, inputs, L3,L2,L1) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs,level0,level1,level2= self._network(inputs)
                L1_pred,L2_pred,L3_pred = level0,level1,level2
            
                prediction = [L1_pred,L2_pred,L3_pred]
                true_labels= [L1,L2,L3]
                outputs=outputs["logits"]
            #predicts=torch.argmax(nn.Softmax(dim=1)(prediction[0]), dim=1)
            L1_predicts=nn.Softmax(dim=1)(prediction[0])
            L2_predicts=nn.Softmax(dim=1)(prediction[1])
            fine_pedicts=nn.Softmax(dim=1)(prediction[2])#outputs
            #L2_pred=level1

            L1_pred2=[]
            L2_final_pred=[]
            L2_i_pred=[]
            for i in range(len(L1_predicts)):
                L12 = torch.argmax(nn.Softmax(dim=1)(prediction[0]), dim=1)
                L1_pred2.append(L12[i].tolist())
                L2_list = hierarchyL2[L12[i].tolist()]
                L2_list2 = []
                for t in L2_list:
                    if t in range(len(level1[i].tolist())):#len(outputs[i].tolist())
                        L2_list2.append(t)
                    else:
                        pass

                L1_temp_pred=[]
                for j in range(len(L2_pred[i])):
                    if j in L2_list2:
                        L1_temp_pred.append(L2_pred[i].tolist()[j]+w)
                    else:
                        L1_temp_pred.append(L2_pred[i].tolist()[j]-w)
                L2_i_pred.append(L1_temp_pred)
            
            
            L2_final_pred=torch.tensor(L2_i_pred)
            L1_pred2=torch.tensor(L1_pred2)
            

            L2_pred2=[]
            final_pred=[]
            i_pred=[]
            for i in range(len(L2_final_pred)):
                L22 = torch.argmax(nn.Softmax(dim=1)(L2_final_pred), dim=1)
                L2_pred2.append(L22[i].tolist())
                fine_list = hierarchyL1[L22[i].tolist()]
                fine_list2 = []
                for t in fine_list:
                    if t in range(len(outputs[i].tolist())):#len(outputs[i].tolist())
                        fine_list2.append(t)
                    else:
                        pass

                temp_pred=[]
                for j in range(len(L3_pred[i])):
                    if j in fine_list2:
                        temp_pred.append(L3_pred[i].tolist()[j]+w)
                    else:
                        temp_pred.append(L3_pred[i].tolist()[j]-w)
                i_pred.append(temp_pred)


            final_pred=torch.tensor(i_pred)#final_pred
            L2_pred2=torch.tensor(L2_pred2)


            predicts = torch.topk(
                nn.Softmax(dim=1)(final_pred), k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]

    
            L3_pred.append(predicts.cpu().numpy())
            L3_true.append(L3.cpu().numpy())
            y_inputs.append(inputs.cpu().numpy())

            L2_pred.append(L2_pred2.cpu().numpy())
            L2_true.append(L2.cpu().numpy())

            L1_pred.append(L1_pred2.cpu().numpy())
            L1_true.append(L1.cpu().numpy())
  

        return np.concatenate(L3_pred), np.concatenate(L3_true)  # [N, topk]
    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, L3_true,L2_true,L1_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, : self.topk], L3_true#,y_L2 # [N, topk]

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, L3,L2,L1 = [], [],[],[]
        for _, _inputs, _L3,_L2,_L1 in loader:
            _L3 = _L3.numpy()
            _L2 = _L2.numpy()
            _L1 = _L1.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.extract_vector(_inputs.to(self._device))
                )
            else:
                _vectors = tensor2numpy(
                    self._network.extract_vector(_inputs.to(self._device))
                )

            vectors.append(_vectors)
            L3.append(_L3)
            L2.append(_L2)
            L1.append(_L1)

        return np.concatenate(vectors), np.concatenate(L3) , np.concatenate(L2),np.concatenate(L1)

    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_L3 , dummy_L2,dummy_L1 = copy.deepcopy(self._data_memory), copy.deepcopy(
            self._L3_memory), copy.deepcopy(self._L2_memory), copy.deepcopy(
            self._L1_memory
        )
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._L3_memory,self._L2_memory,self._L1_memory = np.array([]), np.array([]), np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_L3 == class_idx)[0]
            dd, dt, dc, df = dummy_data[mask][:m], dummy_L3[mask][:m], dummy_L2[mask][:m], dummy_L1[mask][:m]
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._L3_memory = (
                np.concatenate((self._L3_memory, dt))
                if len(self._L3_memory) != 0
                else dt
            )
            self._L2_memory = (
                np.concatenate((self._L2_memory, dc))
                if len(self._L2_memory) != 0
                else dc
            )

            self._L1_memory = (
                np.concatenate((self._L1_memory, df))
                if len(self._L1_memory) != 0
                else df
            )
            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt,dc,df)
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ ,_,_= self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, L3,L2,L1, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ ,_,_= self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)
            exemplar_L3 = np.full(m, class_idx)
            exemplar_L2 = np.full(m, class_idx)
            exemplar_L1 = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._L3_memory = (
                np.concatenate((self._L3_memory, exemplar_L3))
                if len(self._L3_memory) != 0
                else exemplar_L3
            )

            self._L2_memory = (
                np.concatenate((self._L2_memory, exemplar_L2))
                if len(self._L2_memory) != 0
                else L2
            )

            self._L1_memory = (
                np.concatenate((self._L1_memory, exemplar_L1))
                if len(self._L1_memory) != 0
                else L1
            )
            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_L3,L2,L1),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _,_,_ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar_unified(self, data_manager, m):
        logging.info(
            "Constructing exemplars for new classes...({} per classes)".format(m)
        )
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained 
        for class_idx in range(self._known_classes):
            mask = np.where(self._L3_memory == class_idx)[0]
            class_data, class_L3,class_L2,class_L1 = (
                self._data_memory[mask],
                self._L3_memory[mask],
                self._L2_memory[mask],
                self._L1_memory[mask],
            )

            class_dset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(class_data, class_L3,class_L2,class_L1)
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _,_,_= self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, L3,L2,L1, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )

            vectors, _,_,_ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection
            
            selected_exemplars = np.array(selected_exemplars)
            exemplar_L3 = np.full(m, class_idx)
            exemplar_L2 = np.full(m, hierarchyL12[class_idx])
            exemplar_L1 = np.full(m, hierarchyL22[hierarchyL12[class_idx]])

            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._L3_memory = (
                np.concatenate((self._L3_memory, exemplar_L3))
                if len(self._L3_memory) != 0
                else exemplar_L3
            )
            self._L2_memory = (
                np.concatenate((self._L2_memory, exemplar_L2))
                if len(self._L2_memory) != 0
                else exemplar_L2
            )

            self._L1_memory = (
                np.concatenate((self._L1_memory, exemplar_L1))
                if len(self._L1_memory) != 0
                else exemplar_L1
            )
            # Exemplar mean

            exemplar_dset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_L3,exemplar_L2,exemplar_L1),
            )
            exemplar_loader = DataLoader(
                exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ ,_,_= self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        self._class_means = _class_means
