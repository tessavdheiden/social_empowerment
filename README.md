# Improving MADDPG with empowerment
PyTorch Implementation of MADDPG from [*Multi-Agent Actor-Critic for Mixed
Cooperative-Competitive Environments*](https://arxiv.org/abs/1706.02275) (Lowe et. al. 2017)
MADDPG implemented by [shariqiqbal2810](https://github.com/shariqiqbal2810/maddpg-pytorch)

## Requirements

* [OpenAI baselines](https://github.com/openai/baselines), commit hash: 98257ef8c9bd23a24a330731ae54ed086d9ce4a7
* My [fork](https://github.com/shariqiqbal2810/multiagent-particle-envs) of Multi-agent Particle Environments
* [PyTorch](http://pytorch.org/), version: 0.3.0.post4
* [OpenAI Gym](https://github.com/openai/gym), version: 0.9.4
* [Tensorboard](https://github.com/tensorflow/tensorboard), version: 0.4.0rc3 and [Tensorboard-Pytorch](https://github.com/lanpa/tensorboard-pytorch), version: 1.0 (for logging)

The versions are just what I used and not necessarily strict requirements.

## How to Run

All training code is contained within `main.py`. To view options simply run:

```
python main.py --help
```

If you want to checkout the training loss on tensorboard, activate the VE and use:

```
tensorboard --logdir models/model_name
```

## Results

### Cooperative Navigation

| Agent     | Average dist. | # collisions |
| :---:     | :---:         | :---: |
| MADDPG    | 1.767         | 0.209 |
| DDPG      | 1.858         | 0.375 |
| MADDPG+E  | 1.980         | 0.020 |

### Cooperative Communication

| Agent     | Taget reach % | Average distance |
| :---:     | :---:         | :---: |
| MADDPG    | 84.0         | 0.133 |
| DDPG      | 32.0         | 0.456 |
| MADDPG+E  | 82.8         | 0.212 |

