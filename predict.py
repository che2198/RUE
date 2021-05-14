import pickle
import argparse
import numpy as np
import torch
import torchvision

import utils


def get_args():
    parser = argparse.ArgumentParser()
    utils.add_shared_args(parser)

    parser.add_argument('--noise-path', type=str, default=None,
                        help='set the path to the train images noises')
    parser.add_argument('--resume-path', type=str, default=None,
                        help='set where to resume the model')

    return parser.parse_args()


def get_train_loader(dataset='cifar10', batch_size=128, noise_path=None):
    compose = [
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(32, 4),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.))]

    transform = torchvision.transforms.Compose( compose )
    if dataset == 'cifar10':
        target_set = torchvision.datasets.CIFAR10(
            root='./data', train=True, transform=transform, download=True)
    elif dataset == 'cifar100':
        target_set = torchvision.datasets.CIFAR100(
            root='./data', train=True, transform=transform, download=True)
    else:
        raise NotImplementedError

    if noise_path is not None:
        with open(noise_path, 'rb') as f:
            raw_noise = pickle.load(f)

        raw_noise = raw_noise.astype(np.int32)
        noise = raw_noise
        noise = np.transpose(noise, [0,2,3,1])

        imgs = target_set.data.astype(np.int32) + noise
        imgs = imgs.clip(0,255).astype(np.uint8)
        target_set.data = imgs

    loader = torch.utils.data.DataLoader(
        target_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)

    return loader


def get_prediction(model, loader):
    model.eval()
    prediction = []
    with torch.no_grad():
        prediction = [model(x.cuda()).softmax(dim=1) for x, y in loader]
    return torch.cat(prediction).cpu().numpy()


def main(args, logger):
    ''' resume model '''
    model = utils.get_arch(args.arch, args.dataset)
    if args.resume_path is not None:
        state_dict = torch.load(args.resume_path)
        model.load_state_dict( state_dict['model_state_dict'] )
        del state_dict
    model.cuda()

    ''' get train loader '''
    loader = get_train_loader(args.dataset, args.batch_size, noise_path=args.noise_path)

    pred = get_prediction(model, loader)
    print(pred.shape)
    print(pred.dtype)

    with open('{}/{}-pred.pkl'.format(args.save_dir, args.save_name), 'wb') as f:
        pickle.dump(pred, f)

    return


if __name__ == '__main__':
    args = get_args()
    logger = utils.generic_init(args)

    logger.info('EXP: get prediction')
    try:
        main(args, logger)
    except Exception as e:
        logger.exception('Unexpected exception! %s', e)

