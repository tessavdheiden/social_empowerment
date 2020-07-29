import numpy as np
from multiagent.core import World, Agent, Landmark, AgentState, Action, Surface
from multiagent.scenario import BaseScenario
from multiagent.scenarios.car_dynamics import Car
from multiagent.scenarios.cars_racing import CarRacing, FrictionDetector
import scipy.ndimage


colors = np.array([[0.65, 0.15, 0.15], [0.15, 0.65, 0.15], [0.15, 0.15, 0.65],
                   [0.15, 0.65, 0.65], [0.65, 0.15, 0.65], [0.65, 0.65, 0.15]])

ROAD_COLOR = [0.4, 0.4, 0.4]
import matplotlib.pyplot as plt
class DynamicAgent(Agent):
    def __init__(self):
        super(DynamicAgent, self).__init__()
        self.body = None


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
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.color = colors[i]
            agent.body = Car()
            agent.size = 0.025

        # add landmarks
        num_land = 0
        world.landmarks = [Landmark() for i in range(num_land)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.05
            landmark.color = colors[i]

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

        # equal distance between landmarks
        for i, landmark in enumerate(world.landmarks):
            idx = np.minimum(int(len(norm_coord) / len(world.landmarks) * i), len(norm_coord) - 1)
            landmark.state.p_pos = norm_coord[idx]
            landmark.state.p_vel = np.zeros(world.dim_p)

        # all agents start somewhere
        for i, agent in enumerate(world.agents):
            agent.state.p_pos = np.random.uniform(-1, 1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)

        # pure for visualizing the track
        for i, surface in enumerate(world.surfaces):
            surface.color = np.array([color for (_, color) in world.road_poly])
            surface.state.p_pos = np.zeros(world.dim_p)
            surface.state.p_vel = np.zeros(world.dim_p)
            surface.poly = np.array([[self.scale_track(c_i, min_x, min_y, max_x, max_y) for c_i in coordinates] for (coordinates, _) in world.road_poly])
            surface.v = np.mean(surface.poly[:, 0:2], axis=1)

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
        rew = 0.
        for view in world.top_views:
            for road_color, road_patch in zip(ROAD_COLOR, view.transpose(2, 1, 0)):
                rew -= abs(road_color - road_patch)

        return rew

    @staticmethod
    def rgb2gray(rgb, norm=True, scale_factor=70):
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

        view = self.rgb2gray(view)
        obs = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos)
        return np.hstack((obs, view.reshape(-1)))


    def done(self, agent, world):
        pass

    def benchmark_data(self, agent, world):
        return (self.reward(agent, world), )


import math
from gym.utils import seeding
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

STATE_W = 16   # less than Atari 160x192
STATE_H = 16
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0             # Track scale
TRACK_RAD = 900/SCALE   # Track is heavily morphed circle with this radius
PLAYFIELD = 2000/SCALE  # Game over boundary
FPS = 50                # Frames per second
ZOOM = 2.7              # Camera zoom
ZOOM_FOLLOW = True      # Set to False for fixed view (don't use zoom)


TRACK_DETAIL_STEP = 21/SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40/SCALE
BORDER = 8/SCALE
BORDER_MIN_COUNT = 4


class RoadWorld(World):
    def __init__(self):
        super(RoadWorld, self).__init__()
        self.track = None
        self.road = None
        self.road_poly = []
        self.contactListener_keepref = FrictionDetector(self)
        self.box2d = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)

        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]))

        self.seed()
        while True:
            success = self._create_track()
            if success:
                break
        self.shared_viewer = False
        self.top_views = []

    def set_agents(self, agents):
        self.agents = agents
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * len(self.agents)

    def _reset_viewers(self):
        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(STATE_W,STATE_H)

    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    def _add_geoms_to_viewer(self):
        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.entities:
                geom = rendering.make_circle(
                    entity.size) if 'surface' not in entity.name else rendering.make_polygon_with_hole(entity.poly)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                elif 'surface' in entity.name:
                    geom.set_color(entity.color)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

    def _create_top_view(self):
        self.top_views = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = 1 if self.shared_viewer else .1
            if self.shared_viewer:
                pos = np.zeros(self.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0] - cam_range, pos[0] + cam_range, pos[1] - cam_range, pos[1] + cam_range)
            # update geometry positions
            for e, entity in enumerate(self.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            self.top_views.append(self.viewers[i].render(return_rgb_array=True))

    def get_views(self):
        if len(self.top_views) == 0:
            self.top_views = np.random.rand(len(self.agents), STATE_W, STATE_H, 3)

        return self.top_views

    # update state of the world
    def step(self):
        self._reset_render()
        self._reset_viewers()
        self._add_geoms_to_viewer()
        self._create_top_view()
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.box2d.DestroyBody(t)
        self.road = []

    def reset(self):
        self._destroy()
        self.road_poly = []

        while True:
            success = self._create_track()
            if success:
                break

    def _create_track(self):
        CHECKPOINTS = 12

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            alpha = 2 * math.pi * c / CHECKPOINTS + self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
            rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)
            if c == 0:
                alpha = 0
                rad = 1.5 * TRACK_RAD
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1.5 * TRACK_RAD
            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi
            while True:  # Find destination from checkpoints
                failed = True
                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break
                if not failed:
                    break
                alpha -= 2 * math.pi
                continue
            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            proj = r1x * dest_dx + r1y * dest_dy  # destination vector projected on rad
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break

        assert i1 != -1
        assert i2 != -1

        track = track[i1:i2 - 1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2])) +
            np.square(first_perp_y * (track[0][3] - track[-1][3])))
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i - neg - 0][1]
                beta2 = track[i - neg - 1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i - neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (x1 - TRACK_WIDTH * math.cos(beta1), y1 - TRACK_WIDTH * math.sin(beta1))
            road1_r = (x1 + TRACK_WIDTH * math.cos(beta1), y1 + TRACK_WIDTH * math.sin(beta1))
            road2_l = (x2 - TRACK_WIDTH * math.cos(beta2), y2 - TRACK_WIDTH * math.sin(beta2))
            road2_r = (x2 + TRACK_WIDTH * math.cos(beta2), y2 + TRACK_WIDTH * math.sin(beta2))
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.box2d.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3)
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            t.road_visited = False
            t.road_friction = 1.0
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (x1 + side * TRACK_WIDTH * math.cos(beta1), y1 + side * TRACK_WIDTH * math.sin(beta1))
                b1_r = (x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
                        y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1))
                b2_l = (x2 + side * TRACK_WIDTH * math.cos(beta2), y2 + side * TRACK_WIDTH * math.sin(beta2))
                b2_r = (x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
                        y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2))
                self.road_poly.append(([b1_l, b1_r, b2_r, b2_l], (1, 1, 1) if i % 2 == 0 else (1, 0, 0)))
        self.track = track
        return True


class FrictionDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        tile.color[0] = ROAD_COLOR[0]
        tile.color[1] = ROAD_COLOR[1]
        tile.color[2] = ROAD_COLOR[2]
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited:
                tile.road_visited = True
                self.env.reward += 1000.0/len(self.env.track)
                self.env.tile_visited_count += 1
        else:
            obj.tiles.remove(tile)