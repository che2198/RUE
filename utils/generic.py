import pickle
import os
import sys
import logging
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

import models
from . import data


class AverageMeter():
    def __init__(self):
        self.cnt = 0
        self.sum = 0
        self.mean = 0

    def update(self, val, cnt):
        self.cnt += cnt
        self.sum += val * cnt
        self.mean = self.sum / self.cnt

    def average(self):
        return self.mean
    
    def total(self):
        return self.sum


def add_log(log, key, value):
    if key not in log.keys():
        log[key] = []
    log[key].append(value)


def get_transforms(dataset, train=True, is_tensor=True):
    if train:
        if dataset == 'cifar10' or dataset == 'cifar100':
            comp1 = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4), ]
        else:
            raise NotImplementedError
    else:
        comp1 = []

    if is_tensor:
        comp2 = [
            torchvision.transforms.Normalize((255*0.5, 255*0.5, 255*0.5), (255., 255., 255.))]
    else:
        comp2 = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.))]

    trans = transforms.Compose( [*comp1, *comp2] )

    if is_tensor: trans = data.ElementWiseTransform(trans)

    return trans


def get_dataset(dataset, train=True):
    if dataset == 'cifar10' or dataset == 'cifar100':
        if train:
            compose = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.))]
        else:
            compose = [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.))]
    else:
        raise NotImplementedError('dataset {} is not supported'.format(dataset))

    transform = transforms.Compose(compose)

    if dataset == 'cifar10':
        target_set = data.datasetCIFAR10(root='./data', train=train, transform=transform)
    elif dataset == 'cifar100':
        target_set = data.datasetCIFAR100(root='./data', train=train, transform=transform)

    return target_set


def get_indexed_loader(dataset, batch_size, train=True):
    target_set = get_dataset(dataset, train=train)

    if train:
        target_set = data.IndexedDataset(x=target_set.data, y=target_set.targets, transform=target_set.transform)
    else:
        target_set = data.Dataset(x=target_set.data, y=target_set.targets, transform=target_set.transform)

    if train:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=False, drop_last=False)

    return loader


def get_poisoned_loader(dataset, batch_size, train=True, noise_path=None, noise_rate=1.0):
    target_set = get_dataset(dataset, train=train)

    if noise_path is not None:
        with open(noise_path, 'rb') as f:
            raw_noise = pickle.load(f)

        assert isinstance(raw_noise, np.ndarray)
        assert raw_noise.dtype == np.int8

        raw_noise = raw_noise.astype(np.int32)

        noise = np.zeros_like(raw_noise)
        indices = np.random.permutation(len(noise))[:int(len(noise)*noise_rate)]
        noise[indices] += raw_noise[indices]

        ''' restore noise (NCWH) for raw images (NHWC) '''
        noise = np.transpose(noise, [0,2,3,1])

        ''' add noise to images (uint8, 0~255) '''
        imgs = target_set.data.astype(np.int32) + noise
        imgs = imgs.clip(0,255).astype(np.uint8)
        target_set.data = imgs

    target_set = data.Dataset(x=target_set.data, y=target_set.targets, transform=target_set.transform)

    if train:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=False, drop_last=False)

    return loader


def get_arch(arch, dataset):
    if dataset == 'cifar10':
        in_dims, out_dims = 3, 10
    elif dataset == 'cifar100':
        in_dims, out_dims = 3, 100
    else:
        raise NotImplementedError('dataset {} is not supported'.format(dataset))

    if arch == 'resnet18':
        return models.ResNet18(in_dims, out_dims)
    elif arch == 'vgg16-bn':
        return models.VGG16_BN(in_dims, out_dims)
    elif arch == 'inception-v3':
        return models.inception_v3(pretrained=True, resize_input=True)
    else:
        raise NotImplementedError('architecture {} is not supported'.format(arch))


def get_optim(optim, params, lr=0.1, weight_decay=1e-4, momentum=0.9):
    if optim == 'sgd':
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif optim == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    raise NotImplementedError('optimizer {} is not supported'.format(optim))


def generic_init(args):
    if os.path.exists(args.save_dir) == False:
        os.makedirs(args.save_dir)

    fmt = '%(asctime)s %(name)s:%(levelname)s:  %(message)s'
    formatter = logging.Formatter(
        fmt, datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(
        '{}/{}_log.txt'.format(args.save_dir, args.save_name), mode='w')
    fh.setFormatter(formatter)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=fmt, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.addHandler(fh)

    logger.info('Arguments')
    for arg in vars(args):
        logger.info('    {:<22}        {}'.format(arg+':', getattr(args,arg)) )
    logger.info('')

    return logger


def evaluate(model, criterion, loader, cpu):
    acc = AverageMeter()
    loss = AverageMeter()

    model.eval()
    for x, y in loader:
        if not cpu: x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            _y = model(x)
            ac = (_y.argmax(dim=1) == y).sum().item() / len(x)
            lo = criterion(_y,y).item()
        acc.update(ac, len(x))
        loss.update(lo, len(x))

    return acc.average(), loss.average()
