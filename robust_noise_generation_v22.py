import os
import pickle
import argparse
import numpy as np
import torch
import torchvision

import utils


class RobustPGDAttackerV1():
    def __init__(self, samp_num, radius, steps, step_size, random_start, ascending=True):
        self.samp_num = samp_num

        self.radius = radius / 255.
        self.steps = steps
        self.step_size = step_size / 255.
        self.random_start = random_start

        self.ascending = ascending

    def perturb(self, model, criterion, sampler):
        ''' initialize noise '''
        x, y = next(sampler)
        delta = torch.zeros_like(x.data)

        if self.steps==0 or self.radius==0:
            return delta

        if self.random_start:
            delta.uniform_(-self.radius, self.radius)

        ''' temporarily shutdown autograd of model to improve pgd efficiency '''
        model.eval()
        for pp in model.parameters():
            pp.requires_grad = False

        delta.requires_grad_()
        for step in range(self.steps):
            delta.grad = None

            for i in range(self.samp_num):
                # x, y = next(sampler)
                # adv_x = (x + delta).clamp(-0.5, 0.5)
                adv_x = sampler.trans(x + delta * 255)

                adv_x.requires_grad_()
                _y = model(adv_x)
                lo = criterion(_y, y)
                lo.backward()


            # no need to average since one only calculate sign(grad)
            # grad /= self.samp_num

            with torch.no_grad():
                grad = delta.grad.data
                if not self.ascending: grad.mul_(-1)
                delta.add_(torch.sign(grad), alpha=self.step_size)
                delta.clamp_(-self.radius, self.radius)

        ''' reopen autograd of model after pgd '''
        for pp in model.parameters():
            pp.requires_grad = True

        return delta.data


class RobustMinimaxPGDDefenderV1():
    def __init__(self, samp_num,
        radius, steps, step_size, random_start,
        atk_radius, atk_steps, atk_step_size, atk_random_start):

        self.samp_num = samp_num

        self.radius = radius / 255.
        self.steps = steps
        self.step_size = step_size / 255.
        self.random_start = random_start

        self.atk_radius = atk_radius / 255.
        self.atk_steps = atk_steps
        self.atk_step_size = atk_step_size / 255.
        self.atk_random_start = atk_random_start

    def perturb(self, model, criterion, sampler):
        ''' initialize noise '''
        x, y = next(sampler)
        delta = torch.zeros_like(x.data)

        if self.steps==0 or self.radius==0:
            return delta

        if self.random_start:
            delta.uniform_(-self.radius, self.radius)

        ''' temporarily disable autograd of model to improve pgd efficiency '''
        model.eval()
        for pp in model.parameters():
            pp.requires_grad = False

        delta.requires_grad_()
        for step in range(self.steps):
            delta.grad = None

            for i in range(self.samp_num):
                # x, y = next(sampler)
                # def_x = (x + delta).clamp(-0.5, 0.5)
                def_x = sampler.trans(x + delta * 255)
                adv_x = self._get_adv_(model, criterion, def_x.data, y)

                adv_x.requires_grad_()
                _y = model(adv_x)
                lo = criterion(_y, y)

                gd = torch.autograd.grad(lo, [adv_x])[0]

                upd_lo = (def_x * gd).sum()
                upd_lo.backward()

            # no need to average since one only calculate sign(grad)
            # grad /= self.samp_num

            with torch.no_grad():
                grad = delta.grad.data
                grad.mul_(-1)
                delta.add_(torch.sign(grad), alpha=self.step_size)
                delta.clamp_(-self.radius, self.radius)

        ''' re-enable autograd of model after pgd '''
        for pp in model.parameters():
            pp.requires_grad = True

        return delta

    def _get_adv_(self, model, criterion, x, y):
        adv_x = x.clone()
        if self.atk_random_start:
            adv_x += 2 * (torch.rand_like(x) - 0.5) * self.atk_radius
            self._clip_(adv_x, x, radius=self.atk_radius)

        for step in range(self.atk_steps):
            adv_x.requires_grad_()
            _y = model(adv_x)
            loss = criterion(_y, y)

            ''' gradient ascent '''
            grad = torch.autograd.grad(loss, [adv_x])[0]

            with torch.no_grad():
                adv_x.add_(torch.sign(grad), alpha=self.atk_step_size)
                self._clip_(adv_x, x, radius=self.atk_radius)

        return adv_x.data

    def _clip_(self, adv_x, x, radius):
        adv_x -= x
        adv_x.clamp_(-radius, radius)
        adv_x += x
        adv_x.clamp_(-0.5, 0.5)


class Sampler():
    def __init__(self, dataset, indices=None, cpu=True):
        self.dataset = dataset
        self.indices = indices
        self.cpu     = cpu
        compose = [
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32, 4),
            torchvision.transforms.Normalize((255*0.5, 255*0.5, 255*0.5), (255., 255., 255.))]
        self.trans = torchvision.transforms.Compose(compose)

    def __next__(self):
        x, y = [], []

        for i in self.indices:
            xx, yy = self.dataset.x[i], self.dataset.y[i]
            xx = torch.tensor(xx, dtype=torch.float32).permute(2,0,1)
            x.append(xx.reshape(1, *xx.shape))
            y.append(yy)

        x = torch.cat(x)
        y = torch.tensor(y)

        if self.cpu: return x, y
        return x.cuda(), y.cuda()

    def set_indices(self, indices):
        self.indices = indices

    def cuda(self, cpu=False):
        self.cpu = cpu


def get_args():
    parser = argparse.ArgumentParser()
    utils.add_shared_args(parser)

    parser.add_argument('--defender', type=str, default='pgd-v1',
                        choices=['pgd-v1', 'minimax-pgd-v1'],
                        help='set the method for noise generation')

    parser.add_argument('--samp-num', type=int, default=1,
                        help='set the number of samples for calculating expectations')

    parser.add_argument('--atk-pgd-radius', type=float, default=0,
                        help='set the adv perturbation radius in minimax-pgd')
    parser.add_argument('--atk-pgd-steps', type=int, default=0,
                        help='set the number of adv iteration steps in minimax-pgd')
    parser.add_argument('--atk-pgd-step-size', type=float, default=0,
                        help='set the adv step size in minimax-pgd')
    parser.add_argument('--atk-pgd-random-start', action='store_true',
                        help='if select, randomly choose starting points each time performing adv pgd in minimax-pgd')

    parser.add_argument('--resume-path', type=str, default=None,
                        help='set where to resume the model')

    return parser.parse_args()


def main(args, logger):
    model = utils.get_arch(args.arch, args.dataset)
    criterion = torch.nn.CrossEntropyLoss()

    state_dict = torch.load(args.resume_path, map_location=torch.device('cpu'))
    model.load_state_dict( state_dict['model_state_dict'] )
    del state_dict

    if args.defender == 'pgd-v1':
        defender = RobustPGDAttackerV1(
            samp_num     = args.samp_num,
            radius       = args.pgd_radius,
            steps        = args.pgd_steps,
            step_size    = args.pgd_step_size,
            random_start = args.pgd_random_start,
            ascending    = False,
        )

    elif args.defender == 'minimax-pgd-v1':
        defender = RobustMinimaxPGDDefenderV1(
            samp_num         = args.samp_num,
            radius           = args.pgd_radius,
            steps            = args.pgd_steps,
            step_size        = args.pgd_step_size,
            random_start     = args.pgd_random_start,
            atk_radius       = args.atk_pgd_radius,
            atk_steps        = args.atk_pgd_steps,
            atk_step_size    = args.atk_pgd_step_size,
            atk_random_start = args.atk_pgd_random_start,
        )

    trainset = utils.get_dataset(args.dataset, train=True)
    trainset = utils.Dataset(x=trainset.data, y=trainset.targets, transform=trainset.transform)
    sampler = Sampler(trainset)

    if not args.cpu:
        model.cuda()
        criterion.cuda()
        sampler.cuda()

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        def_noise = np.zeros([50000, 3, 32, 32], dtype=np.float32)
    else:
        raise NotImplementedError

    for i in range(0, len(trainset), args.batch_size):
        start, stop = i, min(i+args.batch_size, len(trainset))
        indices = range(start, stop)
        sampler.set_indices( indices )

        delta = defender.perturb(model, criterion, sampler)
        def_noise[indices] = delta.cpu().numpy()

    def_noise = (def_noise * 255).round()
    assert (def_noise.max()<=127 and def_noise.min()>=-128)
    def_noise = def_noise.astype(np.int8)

    with open(os.path.join(args.save_dir, '{}-def-noise.pkl'.format(args.save_name)), 'wb') as f:
        pickle.dump(def_noise, f)


if __name__ == '__main__':
    args = get_args()
    logger = utils.generic_init(args)
    logger.info('EXP: robust noise generation')

    try:
        main(args, logger)
    except Exception as e:
        logger.exception('Unexpected exception! %s', e)
