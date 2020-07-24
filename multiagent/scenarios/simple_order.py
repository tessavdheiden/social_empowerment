import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

theta = np.radians(5)
c, s = np.cos(theta), np.sin(theta)
R = np.array(((c, -s), (s, c)))

vecmod = np.vectorize(lambda x, y : (x + y / 2) % y - y / 2)

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # add agents
        world.agents = [Agent() for i in range(3)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.size = .1
        # add landmarks
        num_land = 3
        world.landmarks = [Landmark() for i in range(num_land)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = .1

        # make initial conditions
        self.reset_world(world)
        self.dims = (2, 2)
        return world

    def reset_world(self, world):
        # random properties for landmarks
        world.landmarks[0].color = np.array([0.75, 0.25, 0.25])
        world.landmarks[1].color = np.array([0.25, 0.75, 0.25])
        world.landmarks[2].color = np.array([0.25, 0.25, 0.75])
        # set random initial states
        for i, agent in enumerate(world.agents):
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.color = world.landmarks[i].color
        for i, landmark in enumerate(world.landmarks):
            theta = np.random.uniform(0, np.pi)
            landmark.state.p_pos = .5 * np.array([np.cos(theta), np.sin(theta)])
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        rew = 0.
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
        for a in world.agents:
            if self.outside_boundary(a):
                rew -= 1.
        return rew

    def observation(self, agent, world):
        def rotate_entity(entity):
            entity.state.p_pos = np.dot(R, entity.state.p_pos) + np.random.uniform(-.01, .01, world.dim_p)
            if self.dims:
                entity.state.p_pos = vecmod(entity.state.p_pos, self.dims)

        for landmark in world.landmarks:
            rotate_entity(landmark)

        # get positions of all entities in this agent's reference frame
        other_pos = []
        for other in world.agents:
            if other == agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)

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
        pass

    def benchmark_data(self, agent, world):
        return (self.reward(agent, world), )
