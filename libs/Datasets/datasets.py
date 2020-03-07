import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import preprocessing as Preprocessing
import math
import os
import struct
import gzip
import errno
import numpy as np


class CUSTOM:
    def __init__(self, is_gpu, args):
        self.valdir = args.val_data
        self.traindir = args.train_data

        self.normalize = self.__get_normalize(args.workers)
        self.train_transforms, self.val_transforms = self.__get_transforms(args.patch_size)

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

    def __get_normalize(self, workers):
        preprocess_params = Preprocessing()
        preprocess_params.get_mean_and_std(self.traindir, workers)
        normalize = transforms.Normalize(preprocess_params.mean, preprocess_params.std)
        return normalize

    def __get_transforms(self, patch_size):
        train_transforms = transforms.Compose([
            transforms.Scale(size=(patch_size, patch_size)),
            transforms.RandomCrop(patch_size, int(math.ceil(patch_size * 0.1))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize,
            ])

        val_transforms = transforms.Compose([
            transforms.Scale(size=(patch_size, patch_size)),
            transforms.ToTensor(),
            self.normalize,
            ])

        return train_transforms, val_transforms

    def get_dataset(self):
        trainset = datasets.ImageFolder(self.traindir, self.train_transforms)
        valset = datasets.ImageFolder(self.valdir, self.val_transforms)

        return trainset, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader


class IMAGENET:
    def __init__(self, is_gpu, args):
        self.valdir = args.val_data
        self.traindir = args.train_data

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.train_transforms, self.val_transforms = self.__get_transforms()

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

    def __get_transforms(self):
        train_transforms = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize,
            ])

        val_transforms = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.normalize,
            ])

        return train_transforms, val_transforms

    def get_dataset(self):
        trainset = datasets.ImageFolder(self.traindir, self.train_transforms)
        valset = datasets.ImageFolder(self.valdir, self.val_transforms)

        return trainset, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader


class CIFAR10:
    def __init__(self, is_gpu, args):
        self.normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                              std=[0.2023, 0.1994, 0.2010])

        self.train_transforms, self.val_transforms = self.__get_transforms(args.patch_size)

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)
        self.val_loader.dataset.class_to_idx = {'airplane': 0,
                                                'automobile': 1,
                                                'bird': 2,
                                                'cat': 3,
                                                'deer': 4,
                                                'dog': 5,
                                                'frog': 6,
                                                'horse': 7,
                                                'ship': 8,
                                                'truck': 9}

    def __get_transforms(self, patch_size):
        train_transforms = transforms.Compose([
            transforms.RandomCrop(patch_size, int(math.ceil(patch_size * 0.1))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize,
            ])

        val_transforms = transforms.Compose([
            transforms.Scale(patch_size),
            transforms.ToTensor(),
            self.normalize,
            ])

        return train_transforms, val_transforms

    def get_dataset(self):
        trainset = datasets.CIFAR10('datasets/CIFAR10/train/', train=True, transform=self.train_transforms,
                                    target_transform=None, download=True)
        valset = datasets.CIFAR10('datasets/CIFAR10/test/', train=False, transform=self.val_transforms,
                                  target_transform=None, download=True)

        return trainset, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader


class CIFAR100:
    def __init__(self, is_gpu, args):
        self.normalize = transforms.Normalize(mean=[0.5071, 0.4866, 0.4409],
                                              std=[0.2009, 0.1984, 0.2023])

        self.train_transforms, self.val_transforms = self.__get_transforms(args.patch_size)

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)
        self.val_loader.dataset.class_to_idx = {'apples': 0,
                                                'aquariumfish': 1,
                                                'baby': 2,
                                                'bear': 3,
                                                'beaver': 4,
                                                'bed': 5,
                                                'bee': 6,
                                                'beetle': 7,
                                                'bicycle': 8,
                                                'bottles': 9,
                                                'bowls': 10,
                                                'boy': 11,
                                                'bridge': 12,
                                                'bus': 13,
                                                'butterfly': 14,
                                                'camel': 15,
                                                'cans': 16,
                                                'castle': 17,
                                                'caterpillar': 18,
                                                'cattle': 19,
                                                'chair': 20,
                                                'chimpanzee': 21,
                                                'clock': 22,
                                                'cloud': 23,
                                                'cockroach': 24,
                                                'computerkeyboard': 25,
                                                'couch': 26,
                                                'crab': 27,
                                                'crocodile': 28,
                                                'cups': 29,
                                                'dinosaur': 30,
                                                'dolphin': 31,
                                                'elephant': 32,
                                                'flatfish': 33,
                                                'forest': 34,
                                                'fox': 35,
                                                'girl': 36,
                                                'hamster': 37,
                                                'house': 38,
                                                'kangaroo': 39,
                                                'lamp': 40,
                                                'lawnmower': 41,
                                                'leopard': 42,
                                                'lion': 43,
                                                'lizard': 44,
                                                'lobster': 45,
                                                'man': 46,
                                                'maple': 47,
                                                'motorcycle': 48,
                                                'mountain': 49,
                                                'mouse': 50,
                                                'mushrooms': 51,
                                                'oak': 52,
                                                'oranges': 53,
                                                'orchids': 54,
                                                'otter': 55,
                                                'palm': 56,
                                                'pears': 57,
                                                'pickuptruck': 58,
                                                'pine': 59,
                                                'plain': 60,
                                                'plates': 61,
                                                'poppies': 62,
                                                'porcupine': 63,
                                                'possum': 64,
                                                'rabbit': 65,
                                                'raccoon': 66,
                                                'ray': 67,
                                                'road': 68,
                                                'rocket': 69,
                                                'roses': 70,
                                                'sea': 71,
                                                'seal': 72,
                                                'shark': 73,
                                                'shrew': 74,
                                                'skunk': 75,
                                                'skyscraper': 76,
                                                'snail': 77,
                                                'snake': 78,
                                                'spider': 79,
                                                'squirrel': 80,
                                                'streetcar': 81,
                                                'sunflowers': 82,
                                                'sweetpeppers': 83,
                                                'table': 84,
                                                'tank': 85,
                                                'telephone': 86,
                                                'television': 87,
                                                'tiger': 88,
                                                'tractor': 89,
                                                'train': 90,
                                                'trout': 91,
                                                'tulips': 92,
                                                'turtle': 93,
                                                'wardrobe': 94,
                                                'whale': 95,
                                                'willow': 96,
                                                'wolf': 97,
                                                'woman': 98,
                                                'worm': 99}

    def __get_transforms(self, patch_size):
        train_transforms = transforms.Compose([
            transforms.RandomCrop(patch_size, int(math.ceil(patch_size * 0.1))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize,
            ])

        val_transforms = transforms.Compose([
            transforms.Scale(patch_size),
            transforms.ToTensor(),
            self.normalize,
            ])

        return train_transforms, val_transforms

    def get_dataset(self):
        trainset = datasets.CIFAR100('datasets/CIFAR100/train/', train=True, transform=self.train_transforms,
                                     target_transform=None, download=True)
        valset = datasets.CIFAR100('datasets/CIFAR100/test/', train=False, transform=self.val_transforms,
                                   target_transform=None, download=True)

        return trainset, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader


class MNIST:
    def __init__(self, is_gpu, patch_size, train_batch_size, val_batch_size, workers, train_dir, val_dir):
        self.normalize = transforms.Normalize(mean=[0.1299, 0.1299, 0.1299],
                                              std=[0.3002, 0.3002, 0.3002])

        self.traindir = train_dir
        self.valdir = val_dir

        self.train_transforms, self.val_transforms = self.__get_transforms(patch_size)

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(train_batch_size, val_batch_size, workers, is_gpu)
        self.val_loader.dataset.class_to_idx = {'0': 0,
                                                '1': 1,
                                                '2': 2,
                                                '3': 3,
                                                '4': 4,
                                                '5': 5,
                                                '6': 6,
                                                '7': 7,
                                                '8': 8,
                                                '9': 9}

    def __get_transforms(self, patch_size):
        train_transforms = transforms.Compose([
            transforms.Scale(patch_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            self.normalize,
            ])

        val_transforms = transforms.Compose([
            transforms.Scale(patch_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            self.normalize,
            ])

        return train_transforms, val_transforms

    def get_dataset(self):
        trainset = datasets.MNIST(self.traindir, train=True, transform=self.train_transforms,
                                  target_transform=None, download=True)
        valset = datasets.MNIST(self.valdir, train=False, transform=self.val_transforms,
                                target_transform=None, download=True)
        return trainset, valset

    def get_dataset_loader(self, train_batch_size, val_batch_size,workers, is_gpu):
        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=train_batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=val_batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader


class FashionMNIST:
    def __init__(self, is_gpu, args):
        self.__path = os.path.expanduser('datasets/FashionMNIST')
        self.__download()

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

        self.val_loader.dataset.class_to_idx = {'T-shirt/top': 0,
                                                'Trouser': 1,
                                                'Pullover': 2,
                                                'Dress': 3,
                                                'Coat': 4,
                                                'Sandal': 5,
                                                'Shirt': 6,
                                                'Sneaker': 7,
                                                'Bag': 8,
                                                'Ankle boot': 9}

    def __check_exists(self):
        return os.path.exists(os.path.join(self.__path, 'train-images-idx3-ubyte.gz')) and \
               os.path.exists(os.path.join(self.__path, 'train-labels-idx1-ubyte.gz')) and \
               os.path.exists(os.path.join(self.__path, 't10k-images-idx3-ubyte.gz')) and \
               os.path.exists(os.path.join(self.__path, 't10k-labels-idx1-ubyte.gz'))

    def __download(self):
        from six.moves import urllib

        if self.__check_exists():
            return

        print("Downloading FashionMNIST dataset")
        urls = [
            'https://cdn.rawgit.com/zalandoresearch/fashion-mnist/ed8e4f3b/data/fashion/train-images-idx3-ubyte.gz',
            'https://cdn.rawgit.com/zalandoresearch/fashion-mnist/ed8e4f3b/data/fashion/train-labels-idx1-ubyte.gz',
            'https://cdn.rawgit.com/zalandoresearch/fashion-mnist/ed8e4f3b/data/fashion/t10k-images-idx3-ubyte.gz',
            'https://cdn.rawgit.com/zalandoresearch/fashion-mnist/ed8e4f3b/data/fashion/t10k-labels-idx1-ubyte.gz',
        ]
        try:
            os.makedirs(self.__path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.__path, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())

        print('Done!')

    def __get_fashion_mnist(self, path, kind='train'):
        labels_path = os.path.join(path,
                                   '%s-labels-idx1-ubyte.gz'
                                   % kind)
        images_path = os.path.join(path,
                                   '%s-images-idx3-ubyte.gz'
                                   % kind)

        with gzip.open(labels_path, 'rb') as lbpath:
            struct.unpack('>II', lbpath.read(8))
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8)

        with gzip.open(images_path, 'rb') as imgpath:
            struct.unpack(">IIII", imgpath.read(16))
            images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 784)

        return images, labels

    def get_dataset(self):
        x_train, y_train = self.__get_fashion_mnist(self.__path, kind='train')
        x_val, y_val = self.__get_fashion_mnist(self.__path, kind='t10k')

        x_train = torch.from_numpy(x_train).float() / 255
        y_train = torch.from_numpy(y_train).int()
        x_val = torch.from_numpy(x_val).float() / 255
        y_val = torch.from_numpy(y_val).int()

        x_train.resize_(x_train.size(0), 1, 28, 28)
        x_val.resize_(x_val.size(0), 1, 28, 28)
        x_train = x_train.repeat(1, 3, 1, 1)
        x_val = x_val.repeat(1, 3, 1, 1)

        trainset = torch.utils.data.TensorDataset(x_train, y_train)
        valset = torch.utils.data.TensorDataset(x_val, y_val)

        return trainset, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True,
                                                   num_workers=workers, pin_memory=is_gpu, sampler=None)
        test_loader = torch.utils.data.DataLoader(self.valset, batch_size=batch_size, shuffle=False,
                                                  num_workers=workers, pin_memory=is_gpu, sampler=None)

        return train_loader, test_loader
