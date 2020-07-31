import numpy as np
from multiagent.core import Surface
from multiagent.dynamic_agent import DynamicAgent
from multiagent.road_world import RoadWorld
from multiagent.scenario import BaseScenario
from multiagent.scenarios.car_dynamics import Car
from multiagent.scenarios.cars_racing import CarRacing, FrictionDetector
import scipy.ndimage


colors = np.array([[0.65, 0.15, 0.15], [0.15, 0.65, 0.15], [0.15, 0.15, 0.65],
                   [0.15, 0.65, 0.65], [0.65, 0.15, 0.65], [0.65, 0.65, 0.15]])
ROAD_COLOR = [0.4, 0.4, 0.4]


class Scenario(BaseScenario):
    def make_world(self):
        world = RoadWorld()
        # set any world properties first
        world.dim_p = 2 # x, y, orientation, speed
        world.collaborative = True

        # add agents
        num_agents = 2
        world.set_agents([DynamicAgent() for i in range(num_agents)])
        for i, agent in enumerate(world.agents):
            agent.name = 'dynamic agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.color = colors[i]
            agent.body = Car()
            agent.size = 0.025
            agent.max_speed = 1.

        world.surfaces = [Surface() for i in range(1)]
        for i, s in enumerate(world.surfaces):
            s.name = 'surface %d' % i
            s.collide = False
            s.movable = False

        # make initial conditions
        self.reset_world(world)
        return world

    def scale_track(self, pos, min_x, min_y, max_x, max_y):
        x, y = pos
        track_width = abs(max_x - min_x)
        track_height = abs(max_y - min_y)
        scale = max(track_height, track_width) / 1.5

        x_new = (x - min_x) / scale - .75
        y_new = (y - min_y) / scale - .75
        return np.array([x_new, y_new])

    def reset_world(self, world):
        world.reset()
        coord = np.array(world.track)[:, 2:4]
        min_x, min_y, max_x, max_y = min(coord[:, 0]),  min(coord[:, 1]),  max(coord[:, 0]),  max(coord[:, 1])
        norm_coord = np.array([self.scale_track(c, min_x, min_y, max_x, max_y) for c in coord])

        # all agents start somewhere
        start_i = np.random.choice(len(norm_coord), len(world.agents), replace=False)
        for i, agent in enumerate(world.agents):
            agent.state.p_pos = norm_coord[start_i[i]]
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.angle = world.track[start_i[i]][1]
            #agent.body.make(init_angle=world.track[start_i[i]][1], init_x=0, init_y=0, world=world.box2d, color=(0.8,0.0,0.0)) # TODO: set x, y

        # pure for visualizing the track
        for i, surface in enumerate(world.surfaces):
            surface.color = np.array([color for (_, color) in world.road_poly])
            surface.state.p_pos = np.zeros(world.dim_p)
            surface.state.p_vel = np.zeros(world.dim_p)
            surface.poly = np.array([[self.scale_track(c_i, min_x, min_y, max_x, max_y) for c_i in coordinates] for (coordinates, _) in world.road_poly])
            surface.v = np.mean(surface.poly[:, 0:2], axis=1)

    def reward(self, agent, world):
        rew = 0.
        for view in world.top_views:
            for road_color, road_patch in zip(ROAD_COLOR, view.transpose(2, 1, 0)):
                rew -= abs(road_color - road_patch / 255.)
        c, w, h = view.shape
        rew /= (c*w*h)
        rew = np.sum(rew)
        #speed = np.sqrt(np.square(agent.state.p_vel[0]) + np.square(agent.state.p_vel[1]))
        #rew += speed
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
        # get positions of all entities in this agent's reference frame
        other_pos = []
        view = None
        for i, other in enumerate(world.agents):
            if other == agent:
                view = world.get_views()[i]
            else:
                other_pos.append(other.state.p_pos - agent.state.p_pos)

        dir = np.array([np.cos(agent.state.angle), np.sin(agent.state.angle)])
        view = self.rgb2gray(view)
        obs = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [dir] + other_pos)
        return np.hstack((obs, view.reshape(-1)))

    def done(self, agent, world):
        pass

    def benchmark_data(self, agent, world):
        return (self.reward(agent, world), )

