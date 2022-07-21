# ODM
Implementation of our Objectwise-Debiasing Memory (ODM) for Human Object Interaction Detection in our ECCV-2022 paper: "Chairs can be stood on: Overcoming Object Bias in Human-Object Interaction Detection"

## Abstract
Detecting Human-Object Interaction (HOI) in images is an important step towards high-level visual comprehension. Existing work often shed light on improving either human and object detection, or interaction recognition. However, due to the limitation of datasets, these methods tend to fit well on frequent interactions conditioned on the detected objects, yet largely ignoring the rare ones, which is referred to as the object bias problem in this paper. In this work, we for the first time, uncover the problem from two aspects: unbalanced interaction distribution and biased model learning. To overcome the object bias problem, we propose a novel plug-and-play Object-wise Debiasing Memory (ODM) method for re-balancing the distribution of interactions under detected objects. Equipped with carefully designed read and write strategies, the proposed ODM allows rare interaction instances to be more frequently sampled for training, thereby alleviating the object bias induced by the unbalanced interaction distribution. We apply this method to three advanced baselines and conduct experiments on the HICO-DET and HOI-COCO datasets. To quantitatively study the object bias problem, we advocate a new protocol for evaluating model performance. As demonstrated in the experimental results, our method brings consistent and significant improvements over baselines, especially on rare interactions under each object. In addition, when evaluating under the conventional standard setting, our method achieves new state-of-the-art on the two benchmarks.

## Usage
For the adopted baseline, please follow the instructions in the respective folder.

## Acknowledgement
This repo is implemented based on the following work:

[SCG](https://github.com/fredzzhang/spatially-conditioned-graphs) 
[QPIC](https://github.com/hitachi-rd-cv/qpic)
[HOID](https://github.com/scwangdyd/zero_shot_hoi)

Thanks them for their great work!