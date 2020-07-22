import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # add agents
        world.agents = [Agent() for i in range(1)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.size = .2
        # add landmarks
        num_land = 3
        world.landmarks = [Landmark() for i in range(num_land)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = .2

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for landmarks
        world.landmarks[0].color = np.array([0.75, 0.25, 0.25])
        world.landmarks[1].color = np.array([0.25, 0.75, 0.25])
        world.landmarks[2].color = np.array([0.25, 0.25, 0.75])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        world.agents[0].goal = world.landmarks[0]
        world.agents[0].goals = np.zeros(len(world.landmarks))
        # random properties for agents
        world.agents[0].color = world.agents[0].goal.color

    def reward(self, agent, world):
        rew = - len(world.landmarks) + sum(world.agents[0].goals)
        rew -= np.sum(np.square(agent.state.p_pos - agent.goal.state.p_pos))
        if self.outside_boundary(agent):
            rew -= 1.
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        return np.concatenate([agent.state.p_vel] + entity_pos + [agent.goal.color])

    def outside_boundary(self, agent):
        if agent.state.p_pos[0] > 1 or agent.state.p_pos[0] < -1 or agent.state.p_pos[1] > 1 or agent.state.p_pos[1] < -1:
            return True
        else:
            return False

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def done(self, agent, world):
        if np.all(agent.goals == 1):
            return True

        # define new target if target reached
        if self.is_collision(agent, agent.goal):
            idx = int(''.join(x for x in agent.goal.name if x.isdigit()))
            agent.goals[idx] = 1
            agent.goal = world.landmarks[idx+1] if idx < len(world.landmarks) - 1 else world.landmarks[idx]

        return False

    def benchmark_data(self, agent, world):
        return (self.reward(agent, world), )
