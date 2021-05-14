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

    parser.add_argument('--noise-path', type=str, default=None,
                        help='set the path to the train images noises')
    parser.add_argument('--resume-path', type=str, default=None,
                        help='set where to resume the model')

    return parser.parse_args()


def evaluate(model, criterion, loader, attacker=None, cpu=False):
    acc = utils.AverageMeter()
    loss = utils.AverageMeter()

    for x, y in loader:
        if not cpu:
            x, y = x.cuda(), y.cuda()

        if attacker is not None:
            adv_x = attacker.perturb(model, criterion, x, y)
        else:
            adv_x = x

        with torch.no_grad():
            model.eval()
            _y = model(adv_x)
            ac = (_y.argmax(dim=1) == y).sum().item() / len(y)
            lo = criterion(_y, y)

        acc.update(ac, len(y))
        loss.update(lo.item(), len(y))

    return acc.average(), loss.average()


def main(args, logger):
    model = utils.get_arch(args.arch, args.dataset)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = utils.get_poisoned_loader(
        args.dataset, batch_size=args.batch_size, train=True,
        noise_path=args.noise_path, noise_rate=1)
    test_loader = utils.get_poisoned_loader(
        args.dataset, batch_size=args.batch_size, train=False)

    state_dict = torch.load(args.resume_path, map_location=torch.device('cpu'))
    model.load_state_dict( state_dict['model_state_dict'] )
    del state_dict

    attacker = attacks.PGDAttacker(
        radius       = args.pgd_radius,
        steps        = args.pgd_steps,
        step_size    = args.pgd_step_size,
        random_start = args.pgd_random_start,
        norm_type    = args.pgd_norm_type,
        ascending    = True,
    )

    if not args.cpu:
        model.cuda()
        criterion = criterion.cuda()

    log = dict()

    nat_train_acc, nat_train_loss = evaluate(
        model, criterion, train_loader, cpu=args.cpu)
    log['nat_train_acc'] = nat_train_acc
    log['nat_train_loss'] = nat_train_loss
    logger.info('nat_train_acc {:.2%} \t nat_train_loss {:.3e}'
                .format( nat_train_acc, nat_train_loss ))

    adv_train_acc, adv_train_loss = evaluate(
        model, criterion, train_loader, attacker=attacker, cpu=args.cpu)
    log['adv_train_acc'] = adv_train_acc
    log['adv_train_loss'] = adv_train_loss
    logger.info('adv_train_acc {:.2%} \t adv_train_loss {:.3e}'
                .format( adv_train_acc, adv_train_loss ))

    nat_test_acc, nat_test_loss = evaluate(
        model, criterion, test_loader, cpu=args.cpu)
    log['nat_test_acc'] = nat_test_acc
    log['nat_test_loss'] = nat_test_loss
    logger.info('nat_test_acc {:.2%} \t nat_test_loss {:.3e}'
                .format( nat_test_acc, nat_test_loss ))

    adv_test_acc, adv_test_loss = evaluate(
        model, criterion, test_loader, attacker=attacker, cpu=args.cpu)
    log['adv_test_acc'] = adv_test_acc
    log['adv_test_loss'] = adv_test_loss
    logger.info('adv_test_acc {:.2%} \t adv_test_loss {:.3e}'
                .format( adv_test_acc, adv_test_loss ))


if __name__ == '__main__':
    args = get_args()
    logger = utils.generic_init(args)

    try:
        main(args, logger)
    except Exception as e:
        logger.exception('Unexpected exception! %s', e)
