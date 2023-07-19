import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iOA
import torch
import random


class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment):
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(
        self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            x,y,y2,y3 = self._train_data, self._train_L3,self._train_L2,self._train_L1
        elif source == "test":
            x,y,y2,y3 = self._test_data, self._test_L3, self._test_L2,self._test_L1
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    #transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, L3,L2,L1 = [], [],[],[]
        for idx in indices:
            if m_rate is None:
                class_data, class_L3 , class_L2, class_L1  = self._select(
                    x,y,y2,y3, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_L3, class_L2 , class_L1  = self._select_rmm(
                    x,y,y2,y3, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            L3.append(class_L3)
            L2.append(class_L2)
            L1.append(class_L1)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_L3,appendent_L2,appendent_L1 = appendent
            data.append(appendent_data)
            L3.append(appendent_L3)
            L2.append(appendent_L2)
            L1.append(appendent_L1)

        data, L3, L2, L1 = np.concatenate(data), np.concatenate(L3),np.concatenate(L2),np.concatenate(L1)
    
        #suffix = datetime.datetime.now().strftime("%H%M%S")
        #output_path1= "".join([base,suffix,'data.pickle'])

        #with open(output_path1, 'wb') as f:
        #    pickle5.dump(np.concatenate(data), f)


        if ret_data:
            return data, L3,L2,L1, DummyDataset(data, L3,L2,L1, trsf, self.use_path)
        else:
            return DummyDataset(data, L3,L2,L1, trsf, self.use_path)

    def get_dataset_with_split(
        self, indices, source, mode, appendent=None, val_samples_per_class=0
    ):
        if source == "train":
            x, y,y2,y3 = self._train_data, self._train_L3, self._train_L2, self._train_L1
        elif source == "test":
            x, y,y2,y3 = self._test_data, self._test_L3, self._test_L2, self._test_L1
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_L3, train_L2, train_L1 = [], [],[],[]
        val_data, val_L3, val_L2, val_L1 = [], [],[],[]
        for idx in indices:
            class_data, class_L3,class_L2,class_L1 = self._select(
                x, y,y2,y3, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_L3.append(class_L3[val_indx])
            val_L2.append(class_L2[val_indx])
            val_L1.append(class_L1[val_indx])

            train_data.append(class_data[train_indx])
            train_L3.append(class_L3[train_indx])
            train_L2.append(class_L2[train_indx])
            train_L1.append(class_L1[train_indx])

        if appendent is not None:
            appendent_data, appendent_L3,appendent_L2,appendent_L1 = appendent
            for idx in range(0, int(np.max(appendent_L3)) + 1):
                append_data, append_L3,append_L2,append_L1 = self._select(
                    appendent_data, appendent_L3,appendent_L2,appendent_L1,low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_L3.append(append_L3[val_indx])
                val_L2.append(append_L2[val_indx])
                val_L1.append(append_L1[val_indx])

                train_data.append(append_data[train_indx])
                train_L3.append(append_L3[train_indx])
                train_L2.append(append_L2[train_indx])
                train_L1.append(append_L1[train_indx])

        train_data, train_L3, train_L2, train_L1 = np.concatenate(train_data), np.concatenate(
            train_L3) , np.concatenate(train_L2) , np.concatenate(train_L1)
        
        val_data, val_L3, val_L2,val_L1 = np.concatenate(val_data), np.concatenate(val_L3),
        np.concatenate(val_L2), np.concatenate(val_L1)

        return DummyDataset(
            train_data, train_L3,train_L2,train_L1, trsf, self.use_path
        ), DummyDataset(val_data, val_L3,val_L2,val_L1,  trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name)
        idata.download_data()

        # Data
        self._train_data, self._train_L3 , self._train_L2 , self._train_L1= idata.train_data, idata.train_L3, idata.train_L2, idata.train_L1
        self._test_data, self._test_L3, self._test_L2 ,self._test_L1= idata.test_data, idata.test_L3, idata.test_L2, idata.test_L1
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order
     
        order = [i for i in range(len(np.unique(self._train_L3)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        # Map indices
        self._train_L3 = _map_new_class_index(
            self._train_L3, self._class_order
        )
        self._test_L3 = _map_new_class_index(self._test_L3, self._class_order)

    def _select(self, x, y,y2,y3, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes] ,y2[idxes],y3[idxes]

    def _select_rmm(self, x, y,y2,y3, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes] ,y2[new_idxes],y3[new_idxes]

    def getlen(self, index):
        y = self._train_L3
        return np.sum(np.where(y == index))


class DummyDataset(Dataset):
    def __init__(self, images, L3,L2,L1, trsf, use_path=False):
        assert len(images) == len(L3), "Data size error!"
        self.images = images
        self.L3 = L3
        self.L2 = L2
        self.L1 = L1
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.L3[idx]
        L2= self.L2[idx]
        L1 = self.L1[idx]

        return idx, image, label,L2,L1


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name):
    name = dataset_name.lower()
    if name == "ioa":
        return iOA()  
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
