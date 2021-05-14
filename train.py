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

    parser.add_argument('--perturb-freq', type=int, default=1,
                        help='set the perturbation frequency')
    parser.add_argument('--report-freq', type=int, default=500,
                        help='set the report frequency')
    parser.add_argument('--save-freq', type=int, default=5000,
                        help='set the checkpoint saving frequency')

    return parser.parse_args()


def save_checkpoint(save_dir, save_name, model, optim, log):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optim.state_dict(),
        }, os.path.join(save_dir, '{}-model.pkl'.format(save_name)))
    with open(os.path.join(save_dir, '{}-log.pkl'.format(save_name)), 'wb') as f:
        pickle.dump(log, f)


def main(args, logger):
    ''' init model / optim / dataloader / loss func '''
    model = utils.get_arch(args.arch, args.dataset)
    optim = utils.get_optim(
        args.optim, model.parameters(),
        lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    train_loader = utils.get_poisoned_loader(
        args.dataset, batch_size=args.batch_size, train=True,
        noise_path=args.noise_path, noise_rate=1)
    test_loader = utils.get_poisoned_loader(
        args.dataset, batch_size=args.batch_size, train=False)
    criterion = torch.nn.CrossEntropyLoss()

    attacker = attacks.PGDAttacker(
        radius = args.pgd_radius,
        steps = args.pgd_steps,
        step_size = args.pgd_step_size,
        random_start = args.pgd_random_start,
        norm_type = args.pgd_norm_type,
        ascending = True,
    )

    if args.resume_path is not None:
        state_dict = torch.load(args.resume_path, map_location=torch.device('cpu'))
        model.load_state_dict( state_dict['model_state_dict'] )
        del state_dict

    if not args.cpu:
        model.cuda()
        criterion = criterion.cuda()

    log = dict()

    for step in range(args.train_steps):
        lr = args.lr * (args.lr_decay_rate ** (step // args.lr_decay_freq))
        for group in optim.param_groups:
            group['lr'] = lr

        x, y = next(train_loader)
        if not args.cpu:
            x, y = x.cuda(), y.cuda()

        if (step+1) % args.perturb_freq == 0:
            adv_x = attacker.perturb(model, criterion, x, y)
        else:
            adv_x = x

        model.train()
        _y = model(adv_x)
        adv_acc = (_y.argmax(dim=1) == y).sum().item() / len(x)
        adv_loss = criterion(_y, y)
        optim.zero_grad()
        adv_loss.backward()
        optim.step()

        utils.add_log(log, 'adv_acc', adv_acc)
        utils.add_log(log, 'adv_loss', adv_loss.item())

        if (step+1) % args.save_freq == 0:
            save_checkpoint(
                args.save_dir, '{}-ckpt-{}'.format(args.save_name, step+1),
                model, optim, log)

        if (step+1) % args.report_freq == 0:
            test_acc, test_loss = utils.evaluate(model, criterion, test_loader, args.cpu)
            utils.add_log(log, 'test_acc', test_acc)
            utils.add_log(log, 'test_loss', test_loss)

            logger.info('step [{}/{}]:'.format(step+1, args.train_steps))
            logger.info('adv_acc {:.2%} \t adv_loss {:.3e}'
                        .format( adv_acc, adv_loss.item() ))
            logger.info('test_acc  {:.2%} \t test_loss  {:.3e}'
                        .format( test_acc, test_loss ))
            logger.info('')

    save_checkpoint(args.save_dir, '{}-fin'.format(args.save_name, step), model, optim, log)

    return


if __name__ == '__main__':
    args = get_args()
    logger = utils.generic_init(args)

    try:
        main(args, logger)
    except Exception as e:
        logger.exception('Unexpected exception! %s', e)
