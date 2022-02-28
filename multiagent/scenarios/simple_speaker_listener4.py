import numpy as np
import random
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

colors = np.array([[0.65, 0.15, 0.15], [0.15, 0.65, 0.15], [0.15, 0.15, 0.65],
                   [0.15, 0.65, 0.65], [0.65, 0.15, 0.65], [0.65, 0.65, 0.15],
                   [0.95, 0.05, 0.05], [0.05, 0.95, 0.05], [0.15, 0.15, 0.65],
                   [0.15, 0.65, 0.65], [0.65, 0.15, 0.65], [0.65, 0.65, 0.15]
                   ])

# A lot more landmarks and obstacles

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 5
        num_landmarks = 12
        num_obstacles = 12
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.size = 0.075
        # speaker
        world.agents[0].movable = False
        # listener
        world.agents[1].silent = True
        world.agents[1].collide = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.04
        # add obstacles
        world.obstacles = [Landmark() for i in range(num_obstacles)]
        for i, landmark in enumerate(world.obstacles):
            landmark.name = 'obstacle %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.04
            landmark.color = np.array([0.15, 0.15, 0.15])
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # assign goals to agents
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
        # want listener to go to the goal landmark
        world.agents[0].goal_a = world.agents[1]
        world.agents[0].goal_b = np.random.choice(world.landmarks)
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = colors[i]

        # special colors for goals
        world.agents[0].goal_a.color = world.agents[0].goal_b.color + np.array([0.45, 0.45, 0.45])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, landmark in enumerate(world.obstacles):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.silent:
            message = -1
        else:
            message = np.argmax(agent.state.c)
        collisions = 0
        if agent.collide:
            for obs in world.obstacles:
                if self.is_collision(obs, agent):
                    collisions += 1
        return (-self.reward(agent, world), np.sum(np.square(world.agents[0].goal_a.state.p_pos - world.agents[0].goal_b.state.p_pos)), message, collisions)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # squared distance from listener to landmark
        a = world.agents[0]
        rew = -np.sum(np.square(a.goal_a.state.p_pos - a.goal_b.state.p_pos))
        if agent.collide:
            for obs in world.obstacles:
                if self.is_collision(obs, agent):
                    rew -= 1
        return rew

    def observation(self, agent, world):
        # goal color
        goal_pos = [np.zeros(world.dim_p), np.zeros(world.dim_p)]
        if agent.goal_b is not None:
            goal_pos[0] = agent.goal_a.state.p_pos
            goal_pos[1] = agent.goal_b.state.p_pos

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        random.shuffle(entity_pos)

        # communication of all other agents
        comm = []
        for other in world.agents:
            if other is agent or (other.state.c is None): continue
            comm.append(other.state.c)

        # speaker
        if not agent.movable:
            obs_pos = []
            for entity in world.obstacles:
                obs_pos.append(entity.state.p_pos)
            random.shuffle(obs_pos)
            return np.concatenate([goal_pos[0]] + [goal_pos[1]] + obs_pos)
        # listener
        if agent.silent:
            return np.concatenate([agent.state.p_vel] + entity_pos + comm)
