import numpy as np
from multiagent.core import World, Agent, Landmark, AgentState, Action, Surface
from multiagent.scenario import BaseScenario
from multiagent.scenarios.car_dynamics import Car
from multiagent.scenarios.cars_racing import CarRacing, FrictionDetector
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
import Box2D


colors = np.array([[0.65, 0.15, 0.15], [0.15, 0.65, 0.15], [0.15, 0.15, 0.65],
                   [0.15, 0.65, 0.65], [0.65, 0.15, 0.65], [0.65, 0.65, 0.15]])

import matplotlib.pyplot as plt
class DynamicAgent(Agent):
    def __init__(self):
        super(DynamicAgent, self).__init__()
        self.body = None

class RoadWorld(World):
    def __init__(self):
        super(RoadWorld, self).__init__()
        self.track = None
        self.physics = None


class Scenario(BaseScenario):
    def make_world(self):
        world = RoadWorld()
        # set any world properties first
        world.dim_p = 2 # x, y, orientation, speed
        world.collaborative = True
        road = CarRacing()
        world.road = road

        # add agents
        self.num_agents = 1
        world.agents = [DynamicAgent() for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.body = Car()
            agent.size = 0.01

        # add landmarks
        num_land = 20
        world.landmarks = [Landmark() for i in range(num_land)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.01

        world.surfaces = [Surface() for i in range(1)]
        for i, s in enumerate(world.surfaces):
            s.name = 'surface %d' % i
            s.collide = False
            s.movable = False

        # make initial conditions
        self.arc_lengths = None
        self.reset_world(world)
        return world

    def normalize_position(self, track, pos):
        x, y = pos
        track_width = abs(max(track[:, 0]) - min(track[:, 0]))
        track_height = abs(max(track[:, 1]) - min(track[:, 1]))
        x_new = (x - min(track[:, 0])) / track_width - .5
        y_new = (y - min(track[:, 1])) / track_height - .5
        return np.array([x_new, y_new])

    def reset_world(self, world):
        world.road.reset()

        track = np.array(world.road.track)[:, 2:4]
        track = np.array([self.normalize_position(track, location) for location in track])

        # random properties for agents
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.95, 0.95, 0.95])
            idx = np.minimum(int(len(track) / len(world.landmarks) * i), len(track) - 1)
            landmark.state.p_pos = track[idx]
            landmark.state.p_vel = np.zeros(world.dim_p)

        self.arc_lengths = np.linalg.norm(track - np.roll(track, -1, axis=0), axis=1)

        for i, agent in enumerate(world.agents):
            agent.color = colors[i]
            agent.action_callback = None
            agent.state.p_pos = world.landmarks[2].state.p_pos + np.random.uniform(-.1,+.1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)

        for i, surface in enumerate(world.surfaces):
            surface.color = np.array([0.75, 0.15, 0.15])
            surface.state.p_pos = np.zeros(world.dim_p)
            surface.state.p_vel = np.zeros(world.dim_p)
            surface.v = track


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def lat_dist(self, agent, world):
        def dist(p1, p2, p3):
            return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

        dists = np.linalg.norm(agent.state.p_pos - world.surfaces[0].v, axis=1)
        i = np.argmin(dists, axis=0)
        p3 = agent.state.p_pos
        if (i < len(dists) - 1) & (i > 0):
            p1 = world.surfaces[0].v[i+1]
            p2 = world.surfaces[0].v[i-1]
        elif i == 0:
            p1 = world.surfaces[0].v[1]
            p2 = world.surfaces[0].v[-1]
        else:
            p1 =world.surfaces[0].v[0]
            p2 = world.surfaces[0].v[-2]

        return dist(p1, p2, p3)

    def backwards(self, agent, world):
        def angle(vector_1, vector_2):
            unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
            unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
            dot_product = np.dot(unit_vector_1, unit_vector_2)
            return np.arccos(dot_product)

        dists = np.linalg.norm(agent.state.p_pos - world.surfaces[0].v, axis=1)
        i = np.argmin(dists, axis=0)

        if (i < len(dists) - 1) & (i > 0):
            p1 = world.surfaces[0].v[i + 1]
            p2 = world.surfaces[0].v[i - 1]
        elif i == 0:
            p1 = world.surfaces[0].v[1]
            p2 = world.surfaces[0].v[-1]
        else:
            p1 = world.surfaces[0].v[0]
            p2 = world.surfaces[0].v[-2]
        angle = angle(agent.state.p_vel, p2 - p1)

        return angle < np.pi / 2

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0.

        # move from start to end
        # dists = np.linalg.norm(agent.state.p_pos - world.surfaces[0].v, axis=1)
        # i = np.argmin(dists, axis=0)
        # rew -= sum(self.arc_lengths[i:])

        if np.all(np.abs(agent.state.p_vel) < .2):
            rew -= 1.

        # stay on road
        if self.lat_dist(agent, world) > .01:
            rew -= 1.

        # moving backwards
        if self.backwards(agent, world):
            rew -= 1.
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)

    def benchmark_data(self, agent, world):
        return (self.reward(agent, world), )