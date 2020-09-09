import math
import numpy as np
from gym.utils import seeding
from multiagent.rendering import Viewer, Transform
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
from multiagent.core import World
from multiagent.scenarios.road_creator import RoadCreator

STATE_W = 16   # less than Atari 160x192
STATE_H = 16


class RoadViewer(Viewer):
    def __init__(self, width, height):
        super(RoadViewer, self).__init__(width, height)
        pass

    def set_bounds_at_angle(self, left, right, bottom, top, angle):
        assert right > left and top > bottom
        # assert angle > 0 and angle < 2*np.pi
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        p = np.array([(right + left) / 2 * scalex, (top + bottom) / 2 * scaley])
        c, s = np.cos(angle), np.sin(angle)
        R = np.array(((c, -s), (s, c)))
        p = np.dot(R, p)
        x_new, y_new = p[0], p[1]
        self.transform = Transform(
            translation=(self.width / 2 - x_new, self.height / 2 - y_new),
            scale=(scalex, scaley),
            rotation=angle)



class RoadWorld(World, RoadCreator):
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

        self.shared_viewer = False
        self.top_views = []
        self.car_physics=False

        self.load_tracks_from = None
        self.num_tracks = 2
        self.num_lanes_changes = 0
        self.num_lanes = 2
        self.max_single_lane = 0
        self.verbose = 0
        self.border_poly = []
        self.road = []

    def set_agents(self, agents):
        self.agents = agents
        if self.shared_viewer:
            self.viewers = [None]
            self.transforms = [None]
        else:
            self.viewers = [None] * len(self.agents)
            self.transforms = [None] * len(self.agents)

    def _reset_viewers(self):
        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = RoadViewer(STATE_W,STATE_H)
                self.transforms[i] = rendering.Transform()

    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    def _add_geoms_to_viewer(self):
        black_list = [] # not render agent in its own viewer
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
                    black_list.append(geom)
                elif 'surface' in entity.name:
                    geom.set_color(entity.color)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for i, viewer in enumerate(self.viewers):
                viewer.geoms = []
                for geom in self.render_geoms:
                    if geom == black_list[i]: continue
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
            angle = -self.agents[i].state.angle
            self.viewers[i].set_bounds_at_angle(pos[0] - cam_range, pos[0] + cam_range, pos[1] - cam_range, pos[1] + cam_range, angle)
            # update geometry positions
            for e, entity in enumerate(self.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            self.top_views.append(self.viewers[i].render(return_rgb_array=True))

    def get_views(self):
        if len(self.top_views) == 0:
            self.top_views = np.random.rand(len(self.agents), STATE_W, STATE_H, 3)

        return self.top_views

    def propagate(self, agent):
        agent.state.angle = agent.state.angle + agent.action.r
        agent.state.p_vel[0] = agent.action.v * np.cos(agent.state.angle + np.pi/2)
        agent.state.p_vel[1] = agent.action.v * np.sin(agent.state.angle + np.pi/2)

    # integrate physical state
    def integrate_state(self, p_force):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            self.propagate(entity)
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt

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
        if self.car_physics:
            for agent in self.agents:
                agent.transform_action_car_input()
                agent.body.step(1.0 / FPS)
            self.box2d.Step(1.0 / FPS, 6 * 30, 2 * 30)
            for agent in self.agents:
                agent.update_state()
        else:
            for agent in self.agents:
                agent.transform_action()
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

        if self.num_lanes == 1:
            while True:
                success = self._create_single_track()
                if success: break

            self.road_poly = [[poly, color, i, 0] for i, (poly, color) in enumerate(self.road_poly)]

        else:
            while True:
                success = self._create_track()
                if success: break

            self.track = np.vstack((self.track[:, 1], self.track[:, 0]))

        self.road_poly = np.array(self.road_poly)


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
        else:
            obj.tiles.remove(tile)