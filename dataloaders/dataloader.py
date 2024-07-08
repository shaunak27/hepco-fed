from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
import pickle
import torch
import torch.utils.data as data
from .utils import download_url, check_integrity
import json 
import time
from models.zeroshot import imr_classnames
VAL_HOLD = 0.1
class iDataset(data.Dataset):
    
    def __init__(self, root,
                train=True, transform=None,
                download_flag=False, lab=True, swap_dset = None, 
                tasks=None, seed=-1, rand_split=False, validation=False, kfolds=5,client_idx = -1):

        # process rest of args
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set
        self.validation = validation
        self.seed = seed
        self.t = -1
        self.tasks = tasks
        self.download_flag = download_flag

        # load dataset
        self.load()
        self.num_classes = len(np.unique(self.targets))

        # remap labels to match task order
        c = 0
        self.class_mapping = {}
        self.class_mapping[-1] = -1
        for task in self.tasks:
            for k in task:
                self.class_mapping[k] = c
                c += 1
        # targets as numpy.array
        self.data = np.asarray(self.data)
        self.targets = np.asarray(self.targets)

        # if validation
        if self.validation:
            
            # shuffle
            state = np.random.get_state()
            np.random.seed(self.seed)
            randomize = np.random.permutation(len(self.targets))
            self.data = self.data[randomize]
            self.targets = self.targets[randomize]
            np.random.set_state(state)

            # sample
            num_data_per_fold = int(len(self.targets) / kfolds)
            start = 0
            stop = num_data_per_fold
            locs_train = []
            locs_val = []
            for f in range(kfolds):
                if self.seed == f:
                    locs_val.extend(np.arange(start,stop))
                else:
                    locs_train.extend(np.arange(start,stop))
                start += num_data_per_fold
                stop += num_data_per_fold

            # train set
            if self.train:
                self.archive = []
                for task in self.tasks:
                    locs = np.isin(self.targets[locs_train], task).nonzero()[0]
                    self.archive.append((self.data[locs_train][locs].copy(), self.targets[locs_train][locs].copy()))

            # val set
            else:
                self.archive = []
                for task in self.tasks:
                    locs = np.isin(self.targets[locs_val], task).nonzero()[0]
                    self.archive.append((self.data[locs_val][locs].copy(), self.targets[locs_val][locs].copy()))

        # else
        else:
            self.archive = []
            for task in self.tasks:
                locs = np.isin(self.targets, task).nonzero()[0]
                self.archive.append((self.data[locs].copy(), self.targets[locs].copy()))

        if self.train:
            self.coreset = (np.zeros(0, dtype=self.data.dtype), np.zeros(0, dtype=self.targets.dtype))

    def __getitem__(self, index, simple = False):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, self.class_mapping[target], self.t


    def load_dataset(self, t, train=True):
        
        if train:
            self.data, self.targets = self.archive[t]
        else:
            self.data    = np.concatenate([self.archive[s][0] for s in range(t+1)], axis=0)
            self.targets = np.concatenate([self.archive[s][1] for s in range(t+1)], axis=0)
        self.t = t

        print(np.unique(self.targets))

    def append_coreset(self, only=False, interp=False):
        len_core = len(self.coreset[0])
        if self.train and (len_core > 0):
            if only:
                self.data, self.targets = self.coreset
            else:
                len_data = len(self.data)
                sample_ind = np.random.choice(len_core, len_data)
                self.data = np.concatenate([self.data, self.coreset[0][sample_ind]], axis=0)
                self.targets = np.concatenate([self.targets, self.coreset[1][sample_ind]], axis=0)

    def update_coreset(self, coreset_size, seen):
        num_data_per = coreset_size // len(seen)
        remainder = coreset_size % len(seen)
        data = []
        targets = []
        
        # random coreset management; latest classes take memory remainder
        # coreset selection without affecting RNG state
        state = np.random.get_state()
        np.random.seed(self.seed*10000+self.t)
        for k in reversed(seen):
            mapped_targets = [self.class_mapping[self.targets[i]] for i in range(len(self.targets))]
            locs = (mapped_targets == k).nonzero()[0]
            if (remainder > 0) and (len(locs) > num_data_per):
                num_data_k = num_data_per + 1
                remainder -= 1
            else:
                num_data_k = min(len(locs), num_data_per)
            locs_chosen = locs[np.random.choice(len(locs), num_data_k, replace=False)]
            data.append([self.data[loc] for loc in locs_chosen])
            targets.append([self.targets[loc] for loc in locs_chosen])
        self.coreset = (np.concatenate(list(reversed(data)), axis=0), np.concatenate(list(reversed(targets)), axis=0))
        np.random.set_state(state)

    def load(self):
        pass


    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class iCIFAR10(iDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the iDataset Dataset.
    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
    im_size=32
    nch=3

    def load(self):

        # download dataset
        if self.download_flag:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train or self.validation:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        self.course_targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
                if 'coarse_labels' in entry:
                    self.course_targets.extend(entry['coarse_labels'])
                
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self._load_meta()

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    

class iCIFAR100(iCIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the iCIFAR10 Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    im_size=32
    nch=3

class iIMAGENET_R(iDataset):
    
    base_folder = 'imagenet-r'
    im_size=224
    nch=3
    def load(self):
        self.data, self.targets = [], []
        images_path = os.path.join(self.root, self.base_folder)
        data_dict = get_data(images_path)
        y = 0
        cwd = os.getcwd()
        mapper_path = os.path.join(cwd,'imr_class_reverse_map.json')
        mapper = json.load(open(mapper_path)) 
        ordered_keys = [mapper[key] for key in imr_classnames]
        for key in ordered_keys:
            num_y = len(data_dict[key])
            self.data.extend([data_dict[key][i] for i in np.arange(0,num_y)])
            self.targets.extend([y for i in np.arange(0,num_y)])
            y += 1

        n_data = len(self.targets)
        index_sample = [i for i in range(n_data)]
        import random
        random.seed(0)
        random.shuffle(index_sample)
        if self.train or self.validation:
            index_sample = index_sample[:int(0.8*n_data)]
        else:
            index_sample = index_sample[int(0.8*n_data):]

        self.data = [self.data[i] for i in index_sample]
        self.targets = [self.targets[i] for i in index_sample]

    def __getitem__(self, index, simple = False):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class
        """
        img_path, target = self.data[index], self.targets[index]
        img = jpg_image_to_array(img_path)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, self.class_mapping[target], self.t

    
    def parse_archives(self) -> None:
        if not check_integrity(os.path.join(self.root, META_FILE)):
            parse_devkit_archive(self.root)

        if not os.path.isdir(self.split_folder):
            if self.split == 'train':
                parse_train_archive(self.root)
            elif self.split == 'val':
                parse_val_archive(self.root)

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

class IMBALANCEINR(iIMAGENET_R):

    def __init__(self, root, train=True, transform=None, download_flag=False, percent= 0.1, 
    imb_type='exp', imb_factor=0.01,seed=-1,tasks=None,validation=False,rand_split=False,client_idx=0):
        self.imb_type = imb_type
        self.imb_factor = imb_factor
        self.percent = percent
        self.client_idx = client_idx
        super(IMBALANCEINR, self).__init__(root, train, transform, download_flag,tasks=tasks,seed=seed,validation=validation,rand_split=rand_split)

    def load_dataset(self, t, train=True, label_counts=None, seed=-1, cutoff=False, cutoff_ratio = 0):
        
        if train:
            self.data, self.targets = self.archive[t]
            self.cls_num = len(np.unique(self.targets))
            if self.client_idx >= 0 and self.train:
                img_num_list = (np.array(self.get_img_num_per_cls(self.cls_num, self.imb_type, self.imb_factor, cutoff_ratio)) * self.percent).astype(int)
                if cutoff :
                    cutoff_index = int(cutoff_ratio * self.cls_num)
                    img_num_list = np.append(img_num_list,np.array([0]*cutoff_index))
                #print(img_num_list)
                classes = self.gen_imbalanced_data(img_num_list,seed=seed) 
                for c,i in zip(classes,img_num_list):
                    label_counts[c] = i
                return label_counts


        else:
            self.data    = np.concatenate([self.archive[s][0] for s in range(t+1)], axis=0)
            self.targets = np.concatenate([self.archive[s][1] for s in range(t+1)], axis=0)
        self.t = t

        #print(np.unique(self.targets))

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor,cutoff_ratio=0):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        new_cls_num = cls_num - int(cls_num * cutoff_ratio)
        if imb_type == 'exp':
            for cls_idx in range(new_cls_num):
                num = img_max * (imb_factor**(cls_idx / (new_cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls,seed=-1):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        t = 1000 * time.time() # current time in milliseconds
        if seed >= 0:
            np.random.seed(seed)
        else:
            np.random.seed(int(t) % 2**32)
        np.random.shuffle(classes) # random shuffle class 
        print(classes)
        # classes = (classes + client_idx)%len(img_num_per_cls)
        self.num_per_cls_dict = dict()
        class_len = []
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            
            self.num_per_cls_dict[the_class] = len(idx) #the_img_num # uncomment
            if the_img_num != 0:
                selec_idx = idx[:int(len(idx)*self.percent)] # uncomment
            else:
                selec_idx = idx[:the_img_num]
            class_len.append(len(selec_idx))
            new_data.extend([self.data[x] for x in selec_idx])
            new_targets.extend([the_class, ] * len(selec_idx))
        #new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        print(class_len)
        return classes
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

class IMBALANCECIFAR(iCIFAR100):

    def __init__(self, root, train=True, transform=None, download_flag=False, percent= 0.1, 
    imb_type='exp', imb_factor=0.01,seed=-1,tasks=None,validation=False,rand_split=False,client_idx=0):
        self.imb_type = imb_type
        self.imb_factor = imb_factor
        self.percent = percent
        self.client_idx = client_idx
        super(IMBALANCECIFAR, self).__init__(root, train, transform, download_flag,tasks=tasks,seed=seed,validation=validation,rand_split=rand_split)
        # np.random.seed(rand_number)

    def load_dataset(self, t, train=True, label_counts=None, seed=-1, cutoff=False, cutoff_ratio = 0):
        
        if train:
            self.data, self.targets = self.archive[t]
            self.cls_num = len(np.unique(self.targets))
            if self.client_idx >= 0 and self.train:
                img_num_list = (np.array(self.get_img_num_per_cls(self.cls_num, self.imb_type, self.imb_factor, cutoff_ratio)) * self.percent).astype(int)
                if cutoff :
                    cutoff_index = int(cutoff_ratio * self.cls_num)
                    img_num_list = np.append(img_num_list,np.array([0]*cutoff_index))
                #print(img_num_list)
                classes = self.gen_imbalanced_data(img_num_list,seed=seed) 
                for c,i in zip(classes,img_num_list):
                    label_counts[c] = i
                return label_counts


        else:
            self.data    = np.concatenate([self.archive[s][0] for s in range(t+1)], axis=0)
            self.targets = np.concatenate([self.archive[s][1] for s in range(t+1)], axis=0)
        self.t = t

        #print(np.unique(self.targets))

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor,cutoff_ratio=0):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        new_cls_num = cls_num - int(cls_num * cutoff_ratio)
        if imb_type == 'exp':
            for cls_idx in range(new_cls_num):
                num = img_max * (imb_factor**(cls_idx / (new_cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls,seed=-1):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        t = 1000 * time.time() # current time in milliseconds
        if seed >= 0:
            np.random.seed(seed)
        else:
            np.random.seed(int(t) % 2**32)
        np.random.shuffle(classes) # random shuffle class 
        print(classes)
        
        self.num_per_cls_dict = dict()
        class_len = []
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            
            self.num_per_cls_dict[the_class] = len(idx) 
            if the_img_num != 0:
                selec_idx = idx[:int(len(idx)*self.percent)] 
            else:
                selec_idx = idx[:the_img_num]
            class_len.append(len(selec_idx))
            new_data.extend([self.data[x] for x in selec_idx])
            new_targets.extend([the_class, ] * len(selec_idx))
        #new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        print(class_len)
        return classes
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

class IMBALANCECIFAR_beta(iCIFAR100):

    def __init__(self, root, train=True, transform=None, download_flag=False, percent= 0.1, 
    imb_type='exp', imb_factor=0.01,seed=-1,tasks=None,validation=False,rand_split=False,client_idx=0):
        self.imb_type = imb_type
        self.imb_factor = imb_factor
        self.percent = percent
        self.client_idx = client_idx
        super(IMBALANCECIFAR_beta, self).__init__(root, train, transform, download_flag,tasks=tasks,seed=seed,validation=validation,rand_split=rand_split)
        # np.random.seed(rand_number)

    def load_dataset(self, t, train=True, label_counts=None, seed=-1, cutoff=False, cutoff_ratio = 0):
        
        if train:
            self.data, self.targets = self.archive[t]
            self.cls_num = len(np.unique(self.targets))
            if self.client_idx >= 0 and self.train:
                img_num_list = (np.array(self.get_img_num_per_cls(self.cls_num, self.imb_type, self.imb_factor, cutoff_ratio)) * self.percent).astype(int)
                if cutoff :
                    cutoff_index = int(cutoff_ratio * self.cls_num)
                    img_num_list = np.append(img_num_list,np.array([0]*cutoff_index))
                print(img_num_list)
                classes = self.gen_imbalanced_data(img_num_list,seed=seed) 
                for c,i in zip(classes,img_num_list):
                    label_counts[c] = i
                return label_counts


        else:
            self.data    = np.concatenate([self.archive[s][0] for s in range(t+1)], axis=0)
            self.targets = np.concatenate([self.archive[s][1] for s in range(t+1)], axis=0)
        self.t = t

        #print(np.unique(self.targets))

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor,cutoff_ratio=0):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        new_cls_num = cls_num - int(cls_num * cutoff_ratio)
        if imb_type == 'exp':
            for cls_idx in range(new_cls_num):
                num = img_max * (imb_factor**(cls_idx / (new_cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls,seed=-1):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        t = 1000 * time.time() # current time in milliseconds
        if seed >= 0:
            np.random.seed(seed)
        else:
            np.random.seed(int(t) % 2**32)
        np.random.shuffle(classes) 
        print(classes)
        
        self.num_per_cls_dict = dict()
        class_len = []
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            
            self.num_per_cls_dict[the_class] = the_img_num 
            selec_idx = idx[:the_img_num]
            class_len.append(len(selec_idx))
            new_data.extend([self.data[x] for x in selec_idx])
            new_targets.extend([the_class, ] * len(selec_idx))
        #new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        print(class_len)
        return classes
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list



class iDOMAIN_NET(iIMAGENET_R):
    base_folder = 'domainnet'
    im_size=224
    nch=3
    def load(self):
        self.data, self.targets = [], []
        images_path = os.path.join(self.root, self.base_folder)
        data_dict = get_data_deep(images_path)
        y = 0
        for key in data_dict.keys():
            num_y = len(data_dict[key])
            self.data.extend([data_dict[key][i] for i in np.arange(0,num_y)])
            self.targets.extend([y for i in np.arange(0,num_y)])
            y += 1
        n_data = len(self.targets)
        index_sample = [i for i in range(n_data)]
        import random
        random.seed(0)
        random.shuffle(index_sample)
        if self.train or self.validation:
            index_sample = index_sample[:int(0.8*n_data)]
        else:
            index_sample = index_sample[int(0.8*n_data):]

        self.data = [self.data[i] for i in index_sample]
        self.targets = [self.targets[i] for i in index_sample]


class IMBALANCEDNET(iDOMAIN_NET):

    def __init__(self, root, train=True, transform=None, download_flag=False, percent= 0.1, 
    imb_type='exp', imb_factor=0.01,seed=-1,tasks=None,validation=False,rand_split=False,client_idx=0):
        self.imb_type = imb_type
        self.imb_factor = imb_factor
        self.percent = percent
        self.client_idx = client_idx
        super(IMBALANCEDNET, self).__init__(root, train, transform, download_flag,tasks=tasks,seed=seed,validation=validation,rand_split=rand_split)
        # np.random.seed(rand_number)

    def load_dataset(self, t, train=True, label_counts=None, seed=-1, cutoff=False, cutoff_ratio = 0):
        
        if train:
            self.data, self.targets = self.archive[t]
            self.cls_num = len(np.unique(self.targets))
            if self.client_idx >= 0 and self.train:
                img_num_list = (np.array(self.get_img_num_per_cls(self.cls_num, self.imb_type, self.imb_factor, cutoff_ratio)) * self.percent).astype(int)
                if cutoff :
                    cutoff_index = int(cutoff_ratio * self.cls_num)
                    img_num_list = np.append(img_num_list,np.array([0]*cutoff_index))
                #print(img_num_list)
                classes = self.gen_imbalanced_data(img_num_list,seed=seed) 
                for c,i in zip(classes,img_num_list):
                    label_counts[c] = i
                return label_counts


        else:
            self.data    = np.concatenate([self.archive[s][0] for s in range(t+1)], axis=0)
            self.targets = np.concatenate([self.archive[s][1] for s in range(t+1)], axis=0)
        self.t = t

        #print(np.unique(self.targets))

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor,cutoff_ratio=0):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        new_cls_num = cls_num - int(cls_num * cutoff_ratio)
        if imb_type == 'exp':
            for cls_idx in range(new_cls_num):
                num = img_max * (imb_factor**(cls_idx / (new_cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls,seed=-1):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        t = 1000 * time.time() # current time in milliseconds
        if seed >= 0:
            np.random.seed(seed)
        else:
            np.random.seed(int(t) % 2**32)
        np.random.shuffle(classes) # random shuffle class 
        print(classes)
        
        self.num_per_cls_dict = dict()
        class_len = []
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            
            self.num_per_cls_dict[the_class] = len(idx) 
            if the_img_num != 0:
                selec_idx = idx[:int(len(idx)*self.percent)] 
            else:
                selec_idx = idx[:the_img_num]
            class_len.append(len(selec_idx))
            new_data.extend([self.data[x] for x in selec_idx])
            new_targets.extend([the_class, ] * len(selec_idx))
        
        self.data = new_data
        self.targets = new_targets
        print(class_len)
        return classes
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

def get_data(root_images):

    import glob
    files = glob.glob(root_images+'/*/*.jpg')
    data = {}
    for path in files:
        y = os.path.basename(os.path.dirname(path))
        if y in data:
            data[y].append(path)
        else:
            data[y] = [path]
    return data

def get_data_deep(root_images):

    import glob
    files = glob.glob(root_images+'/*/*/*.jpg')
    data = {}
    for path in files:
        y = os.path.basename(os.path.dirname(path))
        if y in data:
            data[y].append(path)
        else:
            data[y] = [path]
    return data

def jpg_image_to_array(image_path):
    """
    Loads JPEG image into 3D Numpy array of shape 
    (width, height, channels)
    """
    with Image.open(image_path) as image:      
        image = image.convert('RGB')
        im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((image.size[1], image.size[0], 3))                                   
    return im_arr