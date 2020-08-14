import numpy as np
from multiagent.core import Surface
from multiagent.dynamic_agent import DynamicAgent
from multiagent.road_world import RoadWorld, ROAD_COLOR, TRACK_RAD
from multiagent.scenario import BaseScenario
from multiagent.scenarios.car_dynamics import Car, HULL_POLY1, HULL_POLY2, HULL_POLY3, HULL_POLY4, SIZE
import scipy.ndimage


colors = np.array([[0.65, 0.15, 0.15], [0.15, 0.65, 0.15], [0.15, 0.15, 0.65],
                   [0.15, 0.65, 0.65], [0.65, 0.15, 0.65], [0.65, 0.65, 0.15]])

SCALE = TRACK_RAD * 2 / 2


class Scenario(BaseScenario):
    def make_world(self):
        world = RoadWorld()
        # set any world properties first
        world.dim_p = 2 # x, y, orientation, speed
        world.collaborative = True

        # add agents
        num_agents = 2
        n_frames = 4
        world.set_agents([DynamicAgent() for i in range(num_agents)])
        for i, agent in enumerate(world.agents):
            agent.name = 'dynamic agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.color = colors[i]
            agent.body = Car(world.box2d)
            agent.scale = SCALE

            agent.shape = [[(x * SIZE / SCALE, y * SIZE / SCALE) for x, y in HULL_POLY1],
                           [(x * SIZE / SCALE, y * SIZE / SCALE) for x, y in HULL_POLY2],
                           [(x * SIZE / SCALE, y * SIZE / SCALE) for x, y in HULL_POLY3],
                           [(x * SIZE / SCALE, y * SIZE / SCALE) for x, y in HULL_POLY4]]
            agent.size = SIZE
            self.stacks = [[self.rgb2gray(world.get_views()[i])] * n_frames for i in range(len(world.agents))]

        world.agents[0].max_speed = .1
        world.agents[1].max_speed = .15

        world.surfaces = [Surface() for i in range(1)]
        for i, s in enumerate(world.surfaces):
            s.name = 'surface %d' % i
            s.collide = False
            s.movable = False

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.reset()
        coord = np.array(world.track)[:, 2:4]
        norm_coord = np.array([(c[0] / SCALE, c[1] / SCALE) for c in coord])

        # all agents start somewhere
        start_i = np.random.choice(len(coord))
        dist = 5
        for i, agent in enumerate(world.agents):
            idx = (start_i - i * dist) % len(coord)
            agent.state.p_pos = norm_coord[idx]
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.angle = world.track[idx][1]
            agent.body.make(*world.track[idx][1:4]) # TODO: set x, y

        # pure for visualizing the track
        for i, surface in enumerate(world.surfaces):
            surface.color = np.array([color for (_, color) in world.road_poly])
            surface.state.p_pos = np.zeros(world.dim_p)
            surface.state.p_vel = np.zeros(world.dim_p)
            surface.poly = np.array([[(c_i[0] / SCALE, c_i[1] / SCALE) for c_i in coordinates] for (coordinates, _) in world.road_poly])
            surface.v = np.mean(surface.poly[:, 0:2], axis=1)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        rew = 0.
        for view in world.top_views:
            for road_color, road_patch in zip(ROAD_COLOR, view.transpose(2, 1, 0)):
                rew -= abs(road_color - road_patch / 255.)
        c, w, h = view.shape
        rew /= (c*w*h)
        rew = np.sum(rew)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    @staticmethod
    def rgb2gray(rgb, norm=True):
        # rgb image -> gray [0, 1]
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray

    def observation(self, agent, world):
        for i, other in enumerate(world.agents):
            if other == agent:
                view = world.get_views()[i]
                view = self.rgb2gray(view)
                self.stacks[i].pop(0)
                self.stacks[i].append(view)
                return np.expand_dims(np.array(self.stacks[i]), axis=0)

    def done(self, agent, world):
        pass

    def benchmark_data(self, agent, world):
        return (self.reward(agent, world), )

