import random
from rsoccer_gym.Utils.Utils import OrnsteinUhlenbeckAction
from typing import Dict

import gym
import numpy as np
from rsoccer_gym.Entities import Frame, Robot, Ball
from rsoccer_gym.vss.vss_gym_base import VSSBaseEnv
from rsoccer_gym.Utils import KDTree
from rsoccer_gym.vss.env_vss.controller.controller import goal_keeper_controller, set_goal_keeper_coordinates


class VSSEnv(VSSBaseEnv):
    """This environment controls a single robot in a VSS soccer League 3v3 match 


        Description:
        Observation:
            Type: Box(40)
            Normalized Bounds to [-1.25, 1.25]
            Num             Observation normalized
        Actions:
            Type: Box(2, )
            Num     Action
            0       id 0 Blue Left Wheel Speed  (%)
            1       id 0 Blue Right Wheel Speed (%)
        Reward:
            Sum of Rewards:
                Goal
                Ball Potential Gradient
                Move to Ball
                Energy Penalty
        Starting State:
            Randomized Robots and Ball initial Position
        Episode Termination:
            5 minutes match time
    """

    def __init__(self):
        super().__init__(field_type=0, n_robots_blue=3, n_robots_yellow=3,
                         time_step=0.025)

        self.action_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(4, ), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-self.NORM_BOUNDS,
                                                high=self.NORM_BOUNDS,
                                                shape=(40, ), dtype=np.float32)

        # Initialize Class Atributes
        self.previous_ball_potential = None
        self.actions: Dict = None
        self.reward_shaping_total = None
        self.v_wheel_dead_zone = 0.05

        self.ou_actions = []
        for i in range(self.n_robots_blue + self.n_robots_yellow):
            self.ou_actions.append(
                OrnsteinUhlenbeckAction(self.action_space, dt=self.time_step)
            )

        print('Environment initialized')

    def reset(self):
        self.actions = None
        self.reward_shaping_total = None
        self.previous_ball_potential = None
        for ou in self.ou_actions:
            ou.reset()

        return super().reset()

    def step(self, action):
        observation, reward, done, _ = super().step(action)
        return observation, reward, done, self.reward_shaping_total

    def _frame_to_observations(self):
        observation = [self.norm_pos(self.frame.ball.x), self.norm_pos(self.frame.ball.y),
                       self.norm_v(self.frame.ball.v_x), self.norm_v(self.frame.ball.v_y)]

        for i in range(self.n_robots_blue):
            observation.append(self.norm_pos(self.frame.robots_blue[i].x))
            observation.append(self.norm_pos(self.frame.robots_blue[i].y))

            if i == 1 or i == 2:
                observation.append(self.norm_v(self.frame.robots_blue[i].v_x))
                observation.append(self.norm_v(self.frame.robots_blue[i].v_y))
                observation.append(np.deg2rad(self.frame.robots_blue[i].theta))
                observation.append(self.norm_w(self.frame.robots_blue[i].v_theta))

        for i in range(self.n_robots_yellow):
            observation.append(self.norm_pos(self.frame.robots_yellow[i].x))
            observation.append(self.norm_pos(self.frame.robots_yellow[i].y))

        return np.array(observation, dtype=np.float32)

    def _get_commands(self, actions):
        commands = []
        # self.actions = {1: actions, 2: actions}

        goal_keeper_x, goal_keeper_y = set_goal_keeper_coordinates(self.frame.ball)
        goal_keeper_actions = goal_keeper_controller(goal_keeper_x, goal_keeper_y,
                                                    np.deg2rad(self.frame.robots_blue[0].theta),
                                                    self.frame.robots_blue[0].x, self.frame.robots_blue[0].y)
        v_wheel0, v_wheel1 = self._actions_to_v_wheels(goal_keeper_actions, 1)
        commands.append(Robot(yellow=False, id=0, v_wheel0=goal_keeper_actions[0],
                               v_wheel1=goal_keeper_actions[1]))

        v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions, 1)
        commands.append(Robot(yellow=False, id=1, v_wheel0=v_wheel0,
                              v_wheel1=v_wheel1))

        v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions, 2)
        commands.append(Robot(yellow=False, id=2, v_wheel0=v_wheel0,
                              v_wheel1=v_wheel1))
                                  
        for i in range(self.n_robots_yellow):
            actions = self.ou_actions[self.n_robots_blue+i].sample()
            v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions, i)
            commands.append(Robot(yellow=True, id=i, v_wheel0=v_wheel0, v_wheel1=v_wheel1))
            # commands.append(Robot(yellow=True, id=i, v_wheel0=0,
            #                       v_wheel1=0))

        return commands

    def _calculate_reward_and_done(self):
        reward = 0
        goal = False
        if self.reward_shaping_total is None:
            self.reward_shaping_total = {'goal_score': 0, 'move': 0,
                                         'ball_grad': 0, 'energy': 0,
                                         'goals_blue': 0, 'goals_yellow': 0}

        if self.frame.ball.x > (self.field.length / 2):
            self.reward_shaping_total['goal_score'] += 1
            self.reward_shaping_total['goals_blue'] += 1
            return 10, True

        if self.frame.ball.x < -(self.field.length / 2):
            self.reward_shaping_total['goal_score'] -= 1
            self.reward_shaping_total['goals_yellow'] += 1
            return -10, True

        reward = 0

        return reward, goal

    def _get_initial_positions_frame(self):
        """Returns the position of each robot and ball for the initial frame"""
        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2

        def x(): return random.uniform(-field_half_length + 0.1,
                                       field_half_length - 0.1)

        def y(): return random.uniform(-field_half_width + 0.1,
                                       field_half_width - 0.1)

        def theta(): return random.uniform(0, 360)

        pos_frame: Frame = Frame()

        pos_frame.ball = Ball(x=x(), y=y())

        min_dist = 0.1

        places = KDTree()
        places.insert((pos_frame.ball.x, pos_frame.ball.y))
        
        for i in range(self.n_robots_blue):
            pos = (x(), y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(), y())

            places.insert(pos)
            pos_frame.robots_blue[i] = Robot(x=pos[0], y=pos[1], theta=theta())

        for i in range(self.n_robots_yellow):
            pos = (x(), y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(), y())

            places.insert(pos)
            pos_frame.robots_yellow[i] = Robot(x=pos[0], y=pos[1], theta=theta())

        return pos_frame

    def _actions_to_v_wheels(self, actions, index):
        if index == 0:
            left_wheel_speed = actions[0] * self.max_v
            right_wheel_speed = actions[1] * self.max_v
        else:
            left_wheel_speed = actions[(index - 1) * 2] * self.max_v
            right_wheel_speed = actions[((index - 1) * 2) + 1] * self.max_v

        left_wheel_speed, right_wheel_speed = np.clip(
            (left_wheel_speed, right_wheel_speed), -self.max_v, self.max_v
        )

        # Deadzone
        if -self.v_wheel_dead_zone < left_wheel_speed < self.v_wheel_dead_zone:
            left_wheel_speed = 0

        if -self.v_wheel_dead_zone < right_wheel_speed < self.v_wheel_dead_zone:
            right_wheel_speed = 0

        # Convert to rad/s
        if index != 0:
            left_wheel_speed /= self.field.rbt_wheel_radius
            right_wheel_speed /= self.field.rbt_wheel_radius

        return left_wheel_speed, right_wheel_speed
