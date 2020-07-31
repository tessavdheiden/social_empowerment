import numpy as np
import math

from multiagent.core import Agent, AgentState


class CarState(AgentState):
    def __init__(self):
        super(CarState, self).__init__()
        self.angle = None


class DynamicAgent(Agent):
    def __init__(self):
        super(DynamicAgent, self).__init__()
        self.body = None
        self.actions = {'Left':         [-1.0, 0.0, 0.0],
                        'Right':        [+1.0, 0.0, 0.0],
                        'Brake':        [0.0, 0.0, 0.8],
                        'Accelerate':   [0.0, 1.0, 0.0],
                        'Nothing':      [0.0, 0.0, 0.0]}
        self.state = CarState()

    def update_state(self):
        self.state.p_pos[0] = self.body.hull.position[0]
        self.state.p_pos[1] = self.body.hull.position[1]
        self.state.angle = -self.body.hull.angle
        vel = self.body.hull.linearVelocity
        speed = np.linalg.norm(vel)
        if speed > 0.5:
            self.state.angle = math.atan2(vel[0], vel[1])
        self.state.p_vel = np.array([vel[0], vel[1]])