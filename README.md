# Multi-Agent Reinforcement Learning for Cooperative Coordination
This repo contains an extension of the [*MADDPG*](https://github.com/shariqiqbal2810/maddpg-pytorch) algorithm 
and simulator which is a combination of [particle-env](https://github.com/openai/multiagent-particle-envs) and
[OpenAI Gym Car](https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py).

The MADDPG agents can have complex rules, which make them unable to cooperate with novel partners. 
My solution is to extend it with Empowerment, a information theoretic notion, giving agents the ability to be in control.

## Requirements

```
pip install -e .
```

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
The moving agent needs to go to a landmark with a particular color. However, it is blind and another agent sends messages
that help to navigate. Since there are more landmarks than communication channels, the speaking
agent cannot simply output a symbol corresponding to a particular color. If the listening agent is not receptive to
the messages, the speaker will output random signals. This in turn forces the listener to ignore them. With *empowerment* 
agents remain reactive to one another. 
 
DDPG              | MADDPG             | EMADDPG
:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://user-images.githubusercontent.com/24938569/89288011-3bad5180-d655-11ea-8c5d-d3c895510985.gif" width="200" />|<img src="https://user-images.githubusercontent.com/24938569/89042716-abff5e80-d347-11ea-9ff1-fed829d10d57.gif" width="200" />|<img src="https://user-images.githubusercontent.com/24938569/89042658-91c58080-d347-11ea-8acc-92f1ef9a7b15.gif" width="200" />

```
python simple_speaker_listener3 maddpg+ve --recurrent --variational_transfer_empowerment
```

### Cooperative Coordination
In this simple task agents need to cover all landmarks. MADDPG algorithm is trained by self-play, causing them to agree upon
a rule. For example, agent 1 goes to the red, agent 2 goes to the green and agent 3 to the blue landmark. 
At test time, agent 1 is paired with agent 2 and 3 from a different run and so the former rule does not necessarily results
in the most efficient landmark selection. In contrast, EMADDPG uses *empowerment* that results in picking a landmark closest
to each agent.  

MADDPG             | EMADDPG
:-------------------------:|:-------------------------:
<img src="https://user-images.githubusercontent.com/24938569/89157902-94092400-d56d-11ea-985e-ec243e9daa49.gif" width="200" />|<img src="https://user-images.githubusercontent.com/24938569/89157957-a84d2100-d56d-11ea-93ec-a3dd27494d24.gif" width="200" />

```
python main.py maddpg+ve --recurrent --variational_joint_empowerment
```

### Cooperative Driving
Cars need to stay on the road and need to avoid collisions. Agents only obtain a small top view image and their own states, 
such as orientation and velocity.

Visual inputs:

Red Agent          | Green Agent
:-------------------------:|:-------------------------:
<img src="https://user-images.githubusercontent.com/24938569/89423810-404a3680-d737-11ea-90a4-eb770f49d2cd.gif" width="40" /> | <img src="https://user-images.githubusercontent.com/24938569/89423739-2e689380-d737-11ea-91c0-3bbc108085e3.gif" width="40" />

|                          |DDPG                | MADDPG             
:------------------------- |:-------------------------:|:-------------------------:
Overtaking |<img src="https://user-images.githubusercontent.com/24938569/89404721-bdb37e00-d71a-11ea-93f8-b64af5de5bec.gif" width="400" />|<img src="https://user-images.githubusercontent.com/24938569/89404866-f6535780-d71a-11ea-8d25-00172886a9d8.gif" width="400" />
Obstacle avoidance |<img src="https://user-images.githubusercontent.com/24938569/90010281-e3eb9780-dc9f-11ea-974a-5c5bb2d03e0d.gif" width="400" />|<img src="https://user-images.githubusercontent.com/24938569/90010260-ddf5b680-dc9f-11ea-85b7-66e447a5cae2.gif" width="400" />
Junctions |<img src="https://user-images.githubusercontent.com/24938569/91639279-add74300-ea15-11ea-99cc-5b29628e6f1b.gif" width="400" />|<img src="https://user-images.githubusercontent.com/24938569/91639258-91d3a180-ea15-11ea-885e-50008a9b3c60.gif" width="400" />


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