import numpy as np
from multiagent.core import Surface
from multiagent.dynamic_agent import DynamicAgent
from multiagent.road_world import RoadWorld, STATE_H, STATE_W
from multiagent.scenarios.road_creator import ROAD_COLOR, TRACK_RAD
from multiagent.scenario import BaseScenario
from multiagent.scenarios.car_dynamics import Car, HULL_POLY1, HULL_POLY2, HULL_POLY3, HULL_POLY4, SIZE
import scipy.ndimage


colors = np.array([[0.65, 0.15, 0.15], [0.15, 0.65, 0.15], [0.15, 0.15, 0.65],
                   [0.15, 0.65, 0.65], [0.65, 0.15, 0.65], [0.65, 0.65, 0.15],
                   ROAD_COLOR])

SCALE = TRACK_RAD * 2 / 2


class Scenario(BaseScenario):
    def make_world(self):
        world = RoadWorld()
        # set any world properties first
        world.dim_p = 2 # x, y, orientation, speed
        world.collaborative = True

        # add agents
        num_agents = 3
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

        world.agents[0].max_speed = .05
        world.agents[1].max_speed = .1
        world.agents[2].max_speed = .15


        world.surfaces = [Surface() for i in range(2)]
        for i, s in enumerate(world.surfaces):
            s.name = 'surface %d' % i
            s.collide = False
            s.movable = False

        self.mask = create_circular_mask(STATE_H, STATE_W)
        self.mask = (self.mask - np.min(self.mask))/(np.max(self.mask)-np.min(self.mask))

        # make initial conditions
        world.reset()
        self.reset_world(world)
        return world

    def reset_world(self, world):
        coord = np.array(world.track)[:, 2:4]
        norm_coord = np.array([(c[0] / SCALE, c[1] / SCALE) for c in coord])

        # all agents start somewhere
        start_i = np.random.choice(len(coord))
        dist = 5
        orient_clockwise = np.random.choice(2)
        delta_angle = orient_clockwise * np.pi
        for i, agent in enumerate(world.agents):
            idx = (start_i - i * dist) % len(coord) if orient_clockwise == 0. else (start_i + i * dist) % len(coord)
            agent.state.p_pos = norm_coord[idx]
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.angle = world.track[idx][1] + delta_angle
            agent.body.make(*world.track[idx][1:4]) # TODO: set x, y

        # pure for visualizing the track
        for i, surface in enumerate(world.surfaces):
            surface.color = np.array([color for poly, color, id, lane in world.road_poly if lane == i])
            surface.state.p_pos = np.zeros(world.dim_p)
            surface.state.p_vel = np.zeros(world.dim_p)
            surface.poly = np.array([[(c_i[0] / SCALE, c_i[1] / SCALE) for c_i in poly] for poly, color, id, lane in world.road_poly if lane == i])

    def is_off_road(self, view):
        center = STATE_H // 2, STATE_W // 2
        area_under_car = np.array(view[center[0] - 1: center[0] + 1,
                                        center[1] - 1: center[1] + 1])

        return np.any(area_under_car > 100)  # outside road color is white

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        rew = 0.

        for view in world.top_views:
            h, w, c = view.shape
            pixels = view.transpose(2, 1, 0) #* np.repeat(self.mask.reshape(1, h, w), 3, axis=0)
            rew -= np.sum(abs(np.array(ROAD_COLOR).reshape(3, -1) - pixels.reshape(3, -1) / 255.) / (c*h*w))

        # if agent.collide:
        #     for a in world.agents:
        #         if self.is_collision(a, agent):
        #             rew -= 1
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
        views = world.get_views()
        for i, other in enumerate(world.agents):
            if other == agent:
                view = views[i]
                view = self.rgb2gray(view)
                self.stacks[i].pop(0)
                self.stacks[i].append(view)
                return np.expand_dims(np.array(self.stacks[i]), axis=0)

    def done(self, agent, world):
        pass

    def benchmark_data(self, agent, world):
        collisions = 0
        off_road = 0

        views = world.get_views()
        if agent.collide:
            for i, a in enumerate(world.agents):
                if a != agent:
                    if self.is_collision(a, agent):
                        collisions += 1
                else:
                    if self.is_off_road(views[i]):
                        off_road += 1

        return (self.reward(agent, world), collisions, off_road)


def create_circular_mask(h, w):
    center = (int(w / 2), int(h / 2))

    Y, X = np.ogrid[:h, :w]
    return np.sqrt(center[0] ** 2 + center[1]**2) - np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)