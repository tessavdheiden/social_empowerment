import numpy as np
import math

from multiagent.core import Agent, AgentState, Action


class DynamicState(AgentState):
    def __init__(self):
        super(DynamicState, self).__init__()
        self.angle = None


class DynamicAction(Action):
    def __init__(self):
        super(DynamicAction, self).__init__()
        self.r = None
        self.v = None


class DynamicAgent(Agent):
    def __init__(self):
        super(DynamicAgent, self).__init__()
        self.body = None
        self.actions = {'Left':         [-1.0, 0.0, 0.0],
                        'Right':        [+1.0, 0.0, 0.0],
                        'Brake':        [0.0, 0.0, 0.8],
                        'Accelerate':   [0.0, 1.0, 0.0],
                        'Nothing':      [0.0, 2.0, 0.0]}
        self.state = DynamicState()
        self.action = DynamicAction()
        self.scale = None

    def update_state(self):
        self.state.p_pos[0] = self.body.hull.position[0] / self.scale
        self.state.p_pos[1] = self.body.hull.position[1] / self.scale
        self.state.angle = self.body.hull.angle
        vel = self.body.hull.linearVelocity
        #speed = np.linalg.norm(vel)
        #if speed > 0.5:
            #self.state.angle = math.atan2(vel[0], vel[1])
        self.state.p_vel = np.array([vel[0], vel[1]])

    def transform_action(self):
        u = self.action.u
        if u[0] == 0 and u[1] == 0:     self.action.v = +self.max_speed; self.action.r = 0
        elif u[0] == 0 and u[1] < 0:    self.action.v = +self.max_speed; self.action.r = .1
        elif u[0] == 0 and u[1] > 0:    self.action.v = +self.max_speed; self.action.r = -.1
        elif u[0] < 0 and u[1] == 0:    self.action.v = +2*self.max_speed; self.action.r = .1
        elif u[0] > 0 and u[1] == 0:    self.action.v = +2*self.max_speed; self.action.r = -.1

    def transform_action_car_input(self):
        u = self.action.u
        if u[0] == 0 and u[1] == 0:     action = list(self.actions.values())[0]
        elif u[0] == 0 and u[1] < 0:    action = list(self.actions.values())[1]
        elif u[0] == 0 and u[1] > 0:    action = list(self.actions.values())[2]
        elif u[0] < 0 and u[1] == 0:    action = list(self.actions.values())[3]
        elif u[0] > 0 and u[1] == 0:    action = list(self.actions.values())[4]
        if action is not None:
            self.body.steer(-action[0])
            self.body.gas(action[1])
            self.body.brake(action[2])