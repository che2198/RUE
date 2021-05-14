import os
import pickle
import argparse
import numpy as np
import torch

import utils
import attacks


def get_args():
    parser = argparse.ArgumentParser()
    utils.add_shared_args(parser)

    parser.add_argument('--type', type=str, default='pgd',
                        choices=['pgd', 'minimax-pgd'],
                        help='set the method for noise generation')

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


def regenerate_def_noise(def_noise, model, criterion, loader, defender, cpu):
    for x, y, ii in loader:
        if not cpu: x, y = x.cuda(), y.cuda()
        def_x = defender.perturb(model, criterion, x, y)
        def_noise[ii] = (def_x - x).cpu().numpy()


def main(args, logger):
    model = utils.get_arch(args.arch, args.dataset)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = utils.get_indexed_loader(
        args.dataset, batch_size=args.batch_size, train=True)

    state_dict = torch.load(args.resume_path, map_location=torch.device('cpu'))
    model.load_state_dict( state_dict['model_state_dict'] )
    del state_dict

    if args.type == 'pgd':
        defender = attacks.PGDAttacker(
            radius = args.pgd_radius,
            steps = args.pgd_steps,
            step_size = args.pgd_step_size,
            random_start = args.pgd_random_start,
            norm_type = args.pgd_norm_type,
            ascending = False,
        )
    elif args.type == 'minimax-pgd':
        defender = attacks.MinimaxPGDDefender(
            radius           = args.pgd_radius,
            steps            = args.pgd_steps,
            step_size        = args.pgd_step_size,
            random_start     = args.pgd_random_start,
            atk_radius       = args.atk_pgd_radius,
            atk_steps        = args.atk_pgd_steps,
            atk_step_size    = args.atk_pgd_step_size,
            atk_random_start = args.atk_pgd_random_start,
        )

    if not args.cpu:
        model.cuda()
        criterion = criterion.cuda()

    log = dict()

    def_noise = np.zeros_like(train_loader.loader.dataset.x).transpose([0,3,1,2])
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        def_noise = np.zeros([50000, 3, 32, 32], dtype=np.float32)

    regenerate_def_noise(
        def_noise, model, criterion, train_loader, defender, args.cpu)

    def_noise = (def_noise * 255).round()
    assert (def_noise.max()<=127 and def_noise.min()>=-128)
    def_noise = def_noise.astype(np.int8)

    with open(os.path.join(args.save_dir, '{}-def-noise.pkl'.format(args.save_name)), 'wb') as f:
        pickle.dump(def_noise, f)

if __name__ == '__main__':
    args = get_args()
    logger = utils.generic_init(args)

    try:
        main(args, logger)
    except Exception as e:
        logger.exception('Unexpected exception! %s', e)
