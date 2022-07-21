"""
Test a model and compute detection mAP

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import torch
import argparse
import torchvision
from torch.utils.data import DataLoader

import pocket

from hicodet.hicodet import HICODet
from models_scg import SpatiallyConditionedGraph as SCG
from utils import DataFactory, custom_collate, test

def write_ap(aps, file):
    with open(file, "w+") as f:
        for i in range(600):
            f.write(str(aps[i].item())+"\n")

def main(args):
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False

    num_anno = torch.tensor(HICODet(None, anno_file=os.path.join(
        args.data_root, 'instances_train2015.json')).anno_interaction)
    rare = torch.nonzero(num_anno < 10).squeeze(1)
    non_rare = torch.nonzero(num_anno >= 10).squeeze(1)

    dataloader = DataLoader(
        dataset=DataFactory(
            name='hicodet', partition=args.partition,
            data_root=args.data_root,
            detection_root=args.detection_dir,
        ), collate_fn=custom_collate, batch_size=1,
        num_workers=args.num_workers, pin_memory=True
    )

    net = SCG(
        dataloader.dataset.dataset.object_to_verb, 49,
        num_iterations=args.num_iter,
        max_human=args.max_human,
        max_object=args.max_object,
        box_score_thresh=args.box_score_thresh,
        freq_path = "./misc/hicodet_verb_objwise_freq.pkl",
        K=args.K, buffer_size=args.buffer_size,
        args=args
    )
    epoch = 0
    if os.path.exists(args.model_path):
        print("Loading model from ", args.model_path)
        checkpoint = torch.load(args.model_path, map_location="cpu")
        net.load_state_dict(checkpoint['model_state_dict'], strict=False) 
        epoch = checkpoint["epoch"]
    elif len(args.model_path):
        print("\nWARNING: The given model path does not exist. "
            "Proceed to use a randomly initialised model.\n")

    net.cuda()
    timer = pocket.utils.HandyTimer(maxlen=1)
    
    name_list = ["ensemble", "base", "balance"]
    with timer:
        test_aps = test(net, dataloader, args.detection_dir, args.model_path)
    for i, test_ap in enumerate(test_aps):
        model_file = args.model_path.split(".")[0]
        detection_file=args.detection_dir.split("/")[-1]
        write_ap(test_ap, f"results_lam/baseline_hicodet_aps_{name_list[i]}_{args.lam}.txt")
        print("{} Model at epoch: {} | time elapsed: {:.2f}s\n"
            "Full: {:.4f}, rare: {:.4f}, non-rare: {:.4f}".format(name_list[i],
            epoch, timer[0], test_ap.mean(),
            test_ap[rare].mean(), test_ap[non_rare].mean()
        ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an interaction head")
    parser.add_argument('--data-root', default='hicodet', type=str)
    parser.add_argument('--detection-dir', default='hicodet/detections/test2015',
                        type=str, help="Directory where detection files are stored")
    parser.add_argument('--partition', default='test2015', type=str)
    parser.add_argument('--num-iter', default=2, type=int,
                        help="Number of iterations to run message passing")
    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--max-human', default=15, type=int)
    parser.add_argument('--max-object', default=15, type=int)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--model-path', default='', type=str)

    parser.add_argument('--K', default=2, type=int)
    parser.add_argument('--buffer-size', default=64, type=int)
    parser.add_argument('--balance-classifier-weight', default=1.0, type=float)
    parser.add_argument('--consistency-weight', default=1.0, type=float)
    parser.add_argument('--start_balance_epoch', default=100, type=int)
    parser.add_argument('--age-weight', default=0.5, type=float)
    parser.add_argument('--lam', default=0.5, type=float)
    
    args = parser.parse_args()
    print(args)

    main(args)
