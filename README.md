# SACFormer
This is a Pytorch implementation of SACFormer.

## Environment Setup

We mainly utilized [Mujoco](https://github.com/openai/mujoco-py) for experiments.

```
git clone https://github.com/XTTTZ/SACFormer.git
cd SACFormer
conda env create -f environment.yaml
```
## Example Usage

```
python main.py --env-name Hopper-v2 --alpha 0.2 --goal 4 --cuda --K 6
```
| Environment **(`--env-name`)**| Temperature **(`--alpha`)**| Sequence length **(`--K`)** | 
| ---------------| -------------| -------------|
| HalfCheetah-v2| 0.2| 6|
| Hopper-v2| 0.2| 6|
| Walker2d-v2| 0.2| 6|
| Ant-v2| 0.2| 6|
| Humanoid-v2| 0.05| 10|

## Acknowledgements

We stole some code form [pytorch-soft-actor-critic](https://github.com/pranz24/pytorch-soft-actor-critic) and [decision-transformer](https://github.com/kzl/decision-transformer).

We would like to express our gratitude to the authors for generously sharing their code.




