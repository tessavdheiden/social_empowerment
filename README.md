# Multi-Agent Reinforcement Learning for Cooperative Coordination
This repo contains an extension of the [*MADDPG*](https://github.com/shariqiqbal2810/maddpg-pytorch) algorithm 
and simulator which is a combination of [particle-env](https://github.com/openai/multiagent-particle-envs) and
[OpenAI Gym Car](https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py).

The MADDPG agents can have complex rules, which make them unable to cooperate with novel partners. 
My solution is to extend it with Empowerment, a information theoretic notion, giving agents the ability to be in control.

## Requirements

* [OpenAI baselines](https://github.com/openai/baselines), commit hash: 98257ef8c9bd23a24a330731ae54ed086d9ce4a7
* [OpenAi multi-agent-particle-envs](https://github.com/shariqiqbal2810/multiagent-particle-envs) of Multi-agent Particle Environments
* [PyTorch](http://pytorch.org/), version: 0.3.0.post4
* [OpenAI Gym](https://github.com/openai/gym), version: 0.9.4
* [Tensorboard](https://github.com/tensorflow/tensorboard), version: 0.4.0rc3 and [Tensorboard-Pytorch](https://github.com/lanpa/tensorboard-pytorch), version: 1.0 (for logging)

## How to Run

All training code is contained within `main.py`. To view options simply run:

```
python main.py --help
```

If you want to checkout the training loss on tensorboard, activate the VE and use:

```
tensorboard --logdir models/model_name
```

## Simulation Videos

### Cooperative Communication
MADDPG             | EMADDPG
:-------------------------:|:-------------------------:
<img src="https://user-images.githubusercontent.com/24938569/89042658-91c58080-d347-11ea-8acc-92f1ef9a7b15.gif" width="400" />|<img src="https://user-images.githubusercontent.com/24938569/89042716-abff5e80-d347-11ea-9ff1-fed829d10d57.gif" width="400" />

### Cooperative Driving

<img src="https://user-images.githubusercontent.com/24938569/89043030-2d56f100-d348-11ea-8066-4b2584595439.gif" width="400" />

### Cooperative Coordination

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

