import os
import torch
import argparse
import torchvision
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

import pocket
from pocket.data import HICODet

from models_scg import SpatiallyConditionedGraph as SCG
from utils import custom_collate, CustomisedDLE, DataFactory

def main(rank, args):

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )

    trainset = DataFactory(
        name=args.dataset, partition=args.partitions[0],
        data_root=args.data_root,
        detection_root=args.train_detection_dir,
        flip=True
    )

    valset = DataFactory(
        name=args.dataset, partition=args.partitions[1],
        data_root=args.data_root,
        detection_root=args.val_detection_dir
    )

    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True,
        sampler=DistributedSampler(
            trainset, 
            num_replicas=args.world_size, 
            rank=rank)
    )

    val_loader = DataLoader(
        dataset=valset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True,
        sampler=DistributedSampler(
            valset, 
            num_replicas=args.world_size, 
            rank=rank)
    )

    # Fix random seed for model synchronisation
    torch.manual_seed(args.random_seed)

    if args.dataset == 'hicodet':
        object_to_target = train_loader.dataset.dataset.object_to_verb
        human_idx = 49
        num_classes = 117
        freq_path = "./misc/hicodet_verb_objwise_freq.pkl"
    elif args.dataset == 'vcoco':
        object_to_target = train_loader.dataset.dataset.object_to_action
        human_idx = 1
        num_classes = 24
        freq_path = "./misc/vcoco_verb_objwise_freq.pkl"
    elif args.dataset == 'hoicoco':
        object_to_target = train_loader.dataset.dataset.object_to_verb
        human_idx = 1
        num_classes = 21
        freq_path = "./misc/hoicoco_verb_objwise_freq.pkl"
    net = SCG(
        object_to_target, human_idx, num_classes=num_classes,
        num_iterations=args.num_iter, postprocess=False,
        max_human=args.max_human, max_object=args.max_object,
        box_score_thresh=args.box_score_thresh,
        distributed=True, freq_path=freq_path,
        K=args.K, buffer_size=args.buffer_size,
        args=args
    )

    if os.path.exists(args.checkpoint_path):
        print("=> Rank {}: continue from saved checkpoint".format(
            rank), args.checkpoint_path)
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optim_state_dict = checkpoint['optim_state_dict']
        sched_state_dict = checkpoint['scheduler_state_dict']
        epoch = checkpoint['epoch']
        iteration = checkpoint['iteration']
    else:
        print("=> Rank {}: start from a randomly initialised model".format(rank))
        optim_state_dict = None
        sched_state_dict = None
        epoch = 0; iteration = 0

    engine = CustomisedDLE(
        net,
        train_loader,
        val_loader,
        num_classes=num_classes,
        print_interval=args.print_interval,
        cache_dir=args.cache_dir
    )
    # Seperate backbone parameters from the rest
    param_group_1 = []
    param_group_2 = []
    for k, v in engine.fetch_state_key('net').named_parameters():
        if v.requires_grad:
            if k.startswith('module.backbone'):
                param_group_1.append(v)
            elif k.startswith('module.interaction_head'):
                param_group_2.append(v)
            else:
                raise KeyError(f"Unknown parameter name {k}")
    # Fine-tune backbone with lower learning rate
    optim = torch.optim.AdamW([
        {'params': param_group_1, 'lr': args.learning_rate * args.lr_decay},
        {'params': param_group_2}
        ], lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    lambda1 = lambda epoch: 1. if epoch < args.milestones[0] else args.lr_decay
    lambda2 = lambda epoch: 1. if epoch < args.milestones[0] else args.lr_decay
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=[lambda1, lambda2]
    )

    if os.path.exists(args.checkpoint_path):
        # optim.load_state_dict(optim_state_dict) currently do not load opt parameters
        lr_scheduler.load_state_dict(sched_state_dict)
    # Override optimiser and learning rate scheduler
    engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler)
    engine.update_state_key(epoch=epoch, iteration=iteration)

    engine(args.num_epochs)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--world-size', required=True, type=int,
                        help="Number of subprocesses/GPUs to use")
    parser.add_argument('--dataset', default='hicodet', type=str)
    parser.add_argument('--partitions', nargs='+', default=['train2015', 'test2015'], type=str)
    parser.add_argument('--data-root', default='hicodet', type=str)
    parser.add_argument('--train-detection-dir', default='hicodet/detections/train2015', type=str)
    parser.add_argument('--val-detection-dir', default='hicodet/detections/test2015', type=str)
    parser.add_argument('--num-iter', default=2, type=int,
                        help="Number of iterations to run message passing")
    parser.add_argument('--num-epochs', default=30, type=int)
    parser.add_argument('--random-seed', default=1, type=int)
    parser.add_argument('--learning-rate', default=0.0001, type=float)
    # parser.add_argument('--learning-rate', default=0.000025, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--batch-size', default=12, type=int,
                        help="Batch size for each subprocess")
    parser.add_argument('--lr-decay', default=0.2, type=float,
                        help="The multiplier by which the learning rate is reduced")
    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--max-human', default=15, type=int)
    parser.add_argument('--max-object', default=15, type=int)
    parser.add_argument('--milestones', nargs='+', default=[6, 10], type=int,
                        help="The epoch number when learning rate is reduced")
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--print-interval', default=200, type=int)
    parser.add_argument('--checkpoint-path', default='', type=str)
    parser.add_argument('--cache-dir', type=str, default='./checkpoints')
    
    # buffer
    parser.add_argument('--K', default=2, type=int)
    parser.add_argument('--buffer-size', default=4, type=int)
    parser.add_argument('--balance-classifier-weight', default=0.5, type=float)
    parser.add_argument('--consistency-weight', default=1.0, type=float)
    parser.add_argument('--start-balance-epoch', default=1, type=int)
    parser.add_argument('--start-consistency-epoch', default=3, type=int)
    parser.add_argument('--age-weight', default=0.5, type=float)
    args = parser.parse_args()
    # Automatic learning rate adjustment
    args.learning_rate = args.learning_rate * args.batch_size * args.world_size / 32
    print(args)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "1234"

    mp.spawn(main, nprocs=args.world_size, args=(args,))
