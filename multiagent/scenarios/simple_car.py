import numpy as np
from multiagent.core import World, Agent, Landmark, AgentState, Action, Surface
from multiagent.scenario import BaseScenario
from multiagent.scenarios.car_dynamics import Car
from multiagent.scenarios.cars_racing import CarRacing, FrictionDetector
import scipy.ndimage


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
        self.top_view = None


class Scenario(BaseScenario):
    def make_world(self):
        world = RoadWorld()
        # set any world properties first
        world.dim_p = 2 # x, y, orientation, speed
        world.collaborative = True
        road = CarRacing(seed=2) # TODO: remove seed and reset track in self.reset_world
        world.road = road

        # add agents
        self.num_agents = 1
        world.agents = [DynamicAgent() for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.body = Car()
            agent.size = 0.1

        # add landmarks
        num_land = 6
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
        world.top_view = road.render
        world.road.reset()
        self.reset_world(world)
        return world

    def scale_track(self, track, pos):
        x, y = pos
        track_width = abs(max(track[:, 0]) - min(track[:, 0]))
        track_height = abs(max(track[:, 1]) - min(track[:, 1]))
        scale = max(track_height, track_width) / 1.8
        x_new = (x - min(track[:, 0])) / scale - .9
        y_new = (y - min(track[:, 1])) / scale - .9
        return np.array([x_new, y_new])

    def reset_world(self, world):
        track = np.array(world.road.track)[:, 2:4]
        normalized_track = np.array([self.scale_track(track, location) for location in track])

        # random properties for agents
        for i, landmark in enumerate(world.landmarks):
            landmark.color = colors[i]
            idx = np.minimum(int(len(normalized_track) / len(world.landmarks) * i), len(normalized_track) - 1)
            landmark.state.p_pos = normalized_track[idx]
            landmark.state.p_vel = np.zeros(world.dim_p)

        # want listener to go to the goal landmark
        world.agents[0].goal = world.landmarks[-1]
        world.agents[0].goals = np.zeros(len(world.landmarks))

        #self.arc_lengths = np.linalg.norm(normalized_track - np.roll(normalized_track, -1, axis=0), axis=1)

        for i, agent in enumerate(world.agents):
            agent.color = colors[i]
            agent.action_callback = None
            agent.state.p_pos = world.landmarks[0].state.p_pos + np.random.uniform(-.1,+.1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)

        for i, surface in enumerate(world.surfaces):
            surface.color = np.array([c for (_, c) in world.road.road_poly])
            surface.state.p_pos = np.zeros(world.dim_p)
            surface.state.p_vel = np.zeros(world.dim_p)
            surface.v = track
            surface.poly = np.array([[self.scale_track(track, p) for p in ps] for (ps, _) in world.road.road_poly])

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

    def outside_boundary(self, agent):
        if agent.state.p_pos[0] > 1 or agent.state.p_pos[0] < -1 or agent.state.p_pos[1] > 1 or agent.state.p_pos[1] < -1:
            return True
        else:
            return False

    def reward(self, agent, world):
        rew = -1 if self.outside_boundary(agent) else 0
        # squared distance from listener to landmark
        rew -= np.sum(np.square(agent.state.p_pos - agent.goal.state.p_pos))
        return rew
        # img_rgb = world.top_view(mode="state_pixels")
        # # green penalty
        # if np.mean(img_rgb[:, :, 1]) > 185.0:
        #     rew -= 0.05

    @staticmethod
    def rgb2gray(rgb, norm=True, scale_factor=2):
        # rgb image -> gray [0, 1]
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        if scale_factor > 1:
            gray = gray[::scale_factor, ::scale_factor]

        return gray

    def observation(self, agent, world):
        #return self.rgb2gray(world.top_view(mode="state_pixels")).reshape(-1)

        # goal color
        goal_color = np.zeros(world.dim_color)
        if agent.goal is not None:
            goal_color = np.array([int(''.join(x for x in agent.goal.name if x.isdigit()))])

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        return np.concatenate([agent.state.p_vel] + entity_pos + [goal_color])

    def done(self, agent, world):
        if np.all(agent.goals):
            return True

        # define new target if target reached
        # if self.is_collision(agent, agent.goal):
        #     idx = int(''.join(x for x in agent.goal.name if x.isdigit()))
        #     agent.goals[idx] = True
        #     agent.goal = world.landmarks[idx+1]

        return False # self.lat_dist(agent, world) > .01

    def benchmark_data(self, agent, world):
        return (self.reward(agent, world), )