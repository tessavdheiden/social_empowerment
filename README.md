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
DDPG              | MADDPG             | EMADDPG
:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://user-images.githubusercontent.com/24938569/89288011-3bad5180-d655-11ea-8c5d-d3c895510985.gif" width="200" />|<img src="https://user-images.githubusercontent.com/24938569/89042716-abff5e80-d347-11ea-9ff1-fed829d10d57.gif" width="200" />|<img src="https://user-images.githubusercontent.com/24938569/89042658-91c58080-d347-11ea-8acc-92f1ef9a7b15.gif" width="200" />

### Cooperative Coordination
MADDPG             | EMADDPG
:-------------------------:|:-------------------------:
<img src="https://user-images.githubusercontent.com/24938569/89157902-94092400-d56d-11ea-985e-ec243e9daa49.gif" width="200" />|<img src="https://user-images.githubusercontent.com/24938569/89157957-a84d2100-d56d-11ea-93ec-a3dd27494d24.gif" width="200" />


### Cooperative Driving

EMADDPG
:-------------------------:
<img src="https://user-images.githubusercontent.com/24938569/89291181-5c2bda80-d65a-11ea-8166-75539b6ae62c.gif" width="200" />

Visual inputs:

Red Agent          | Green Agent
:-------------------------:|:-------------------------:
<img src="https://user-images.githubusercontent.com/24938569/89044659-8aec3d00-d34a-11ea-8617-69b0b3a1776c.gif" width="40" /> | <img src="https://user-images.githubusercontent.com/24938569/89044549-61331600-d34a-11ea-9c16-d02461694e40.gif" width="40" />

### Cooperative Coordination

| Agent     | Average dist. | Collisions % |
| :---:     | :---:         | :---: |
| MADDPG    | 1.767         | 20.9 |
| EMADDPG   | 0.180         | 2.01 |

The average distance of a landmark (lower is better) and number of collisions between agents.

### Cooperative Communication

| Agent     | Taget reach % | Average distance | Obstacle hits % |
| :---:     | :---:         | :---: |           :---: |
| MADDPG    | 84.0          | 2.233 |           53.5 |
| EMADDPG   | 98.8          | 0.012 |           1.90 |

The target is reached if it has <.1 from the target landmark (higher is better).