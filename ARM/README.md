# Attention-driven Robotic Manipulation (ARM)

Codebase of Q-attention, coarse-to-fine Q-attention, and other variants. Code from the following papers:

- [Q-attention: Enabling Efficient Learning for Vision-based Robotic Manipulation](https://arxiv.org/abs/2105.14829) (ARM system)
- [Coarse-to-Fine Q-attention: Efficient Learning for Visual Robotic Manipulation via Discretisation](https://arxiv.org/abs/2106.12534) (C2F-ARM system)
- [Coarse-to-Fine Q-attention with Learned Path Ranking](https://arxiv.org/abs/2204.01571) (C2F-ARM+LPR system)
- [Coarse-to-Fine Q-attention with Tree Expansion](https://arxiv.org/abs/2204.12471)

![task grid image missing](readme_files/variants.png)


## Installation

ARM is trained using the **YARR framework**. Head to the [YARR github](https://github.com/stepjam/YARR) page and follow 
installation instructions.

ARM is evaluated on **RLBench** 1.1.0. Head to the [RLBench github](https://github.com/stepjam/RLBench) page and follow 
installation instructions. 

Now install project requirements:
```bash
pip install -r requirements.txt
```

## Running experiments

Be sure to have RLBench demos saved on your machine before proceeding. To generate demos for a task, go to the 
tools directory in RLBench _(rlbench/tools)_, and run:
```bash
python dataset_generator.py --save_path=/mnt/my/save/dir --tasks=take_lid_off_saucepan --image_size=128,128 \
--renderer=opengl --episodes_per_task=100 --variations=1 --processes=1
```


Experiments are launched via [Hydra](https://hydra.cc/). To start training **C2F-ARM** on the 
**take_lid_off_saucepan** task with the default parameters on **gpu 0**:
```bash
python launch.py method=C2FARM rlbench.task=take_lid_off_saucepan rlbench.demo_path=/mnt/my/save/dir framework.gpu=0
```

To launch **C2F-ARM+LPR**:
```bash
python launch.py method=LPR rlbench.task=take_lid_off_saucepan rlbench.demo_path=/mnt/my/save/dir framework.gpu=0
```

To launch **C2F-ARM+QTE**:
```bash
python launch.py method=QTE rlbench.task=take_lid_off_saucepan rlbench.demo_path=/mnt/my/save/dir framework.gpu=0
```
