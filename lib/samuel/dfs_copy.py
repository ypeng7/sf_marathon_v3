# -*- coding: utf-8 -*- 
# @Author: Samuel Chan
# @E-mail: samuelchan1205@gmail.com
# @Date: 2018/12/12 上午10:59
# @Last Modified by: Samuel Chan


from env import *
from a_star import *

import time
import random
import copy
# import numpy as np
from itertools import product


class MyDFS:
    def __init__(self, name, max_bags=10, no_look=3):
        self.NAME = name
        self.player = None
        self.MAPSIZE = 12
        self.POINT = 1.0
        self.DIVISOR = 100.0
        self.RATE = 0.7
        self.RATE_E = 0.95
        self.MAX_SEARCH = 10
        self.MAX_STEPS = 200
        self.MAX_BAGS = max_bags
        self.NO_LOOK = no_look
        self.n_jobs = 0
        self.steps = 0
        self.go_home = False
        self.origin_map = None
        self.real_map = None
        self.help_map = None
        self.home_path = None
        self.my_home = None
        self.pre_position = None
        self.max_parcel = None
        self.delta_list = None

    def create_map(self, p1, p2, walls):
        self.origin_map = list()
        for i in range(self.MAPSIZE):
            row = list()
            for j in range(self.MAPSIZE):
                row.append(-1)
            self.origin_map.append(row)
        # self.origin_map = np.array(self.origin_map)
        self.walls = dict()
        for w in walls:
            # self.origin_map[w["x"], w["y"]] = -2
            self.origin_map[w["x"]][w["y"]] = -2
            self.walls[(w["x"], w["y"])] = 1
        if p1["name"] == self.NAME:
            # self.origin_map[p2["home_x"], p2["home_y"]] = -2
            self.origin_map[p2["home_x"]][p2["home_y"]] = -2
        else:
            # self.origin_map[p1["home_x"], p1["home_y"]] = -2
            self.origin_map[p1["home_x"]][p1["home_y"]] = -2
        self.help_map = copy.deepcopy(self.origin_map)
        self._weight_map(self.help_map)

        for i in range(self.MAPSIZE):
            for j in range(self.MAPSIZE):
                if i == 0 and j == 0:
                    continue
                can_have_job = True
                if (i, j) in self.walls:
                    can_have_job = False
                if can_have_job:
                    for k in range(1, 12):
                        for m in range(k+1):
                            n = k - m
                            self._add_weight(k * self.DIVISOR, self.help_map, [i - m, j - n], self.POINT)
                            self._add_weight(k * self.DIVISOR, self.help_map, [i - m, j + n], self.POINT)
                            self._add_weight(k * self.DIVISOR, self.help_map, [i + m, j - n], self.POINT)
                            self._add_weight(k * self.DIVISOR, self.help_map, [i + m, j + n], self.POINT)

    def _weight_map(self, map):
        for i in range(self.MAPSIZE):
            for j in range(self.MAPSIZE):
                # if map[i, j] != -2:
                #     map[i, j] = 0
                if map[i][j] != -2:
                    map[i][j] = 0

    def _add_weight(self, x, map, position, p):
        if position[0] < 0 or position[0] >= self.MAPSIZE or position[1] < 0 or position[1] >= self.MAPSIZE or \
                map[position[0]][position[1]] == -2:
                # map[position[0], position[1]] == -2:
            return
        # map[position[0], position[1]] += p / x
        map[position[0]][position[1]] += p / x

    def add_jobs(self, map, jobs, position):
        max_parcel = 0
        for j in jobs:
            # map[j["x"], j["y"]] = j["value"]
            map[j["x"]][j["y"]] = j["value"]
            max_parcel = max(j["value"], max_parcel)
        # map[position[0], position[1]] = 0
        map[position[0]][position[1]] = 0
        return max_parcel

    def explore(self, max_parcel, position, help_map, real_map, tmp_trace, tmp_reward, fake_reward, exploration, step):
        if step >= max_parcel or position[0] < 0 or position[0] >= self.MAPSIZE or position[1] < 0 or position[
            1] >= self.MAPSIZE or real_map[position[0]][position[1]] == -2:
            return

        # tmp_reward += pow(self.RATE_E, step) * (real_map[position[0], position[1]] - step) if real_map[position[0],
        #                                                                                      position[1]] > step else 0
        # fake_reward += pow(self.RATE_E, step) * (real_map[position[0], position[1]] - step) if real_map[position[0],
        #                                                                                       position[1]] > step else 0
        tmp_reward += pow(self.RATE_E, step) * (real_map[position[0]][position[1]] - step) if real_map[position[0]][position[1]] > step else 0
        fake_reward += pow(self.RATE_E, step) * (real_map[position[0]][position[1]] - step) if real_map[position[0]][position[1]] > step else 0
        # real = real_map[position[0], position[1]]
        real = real_map[position[0]][position[1]]
        # real_map[position[0], position[1]] = 0
        real_map[position[0]][position[1]] = 0

        add_on = 0
        # add_on += help_map[position[0], position[1]] * pow(self.RATE, step)
        add_on += help_map[position[0]][position[1]] * pow(self.RATE, step)
        fake_reward += add_on

        if fake_reward > exploration["fake_reward"]:
            exploration["fake_reward"] = fake_reward
            exploration["max_reward"] = tmp_reward
            exploration["max_trace"] = tmp_trace

        delta = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        for d in delta:
            self.explore(max_parcel, [position[0] + d[0], position[1] + d[1]], help_map, real_map, tmp_trace, tmp_reward,
                    fake_reward, exploration, step + 1)

        real_map[position[0]][position[1]] = real
        # real_map[position[0], position[1]] = real

    def move(self, position, pre_position):
        if position[0] > pre_position[0]:
            diretion = "D"
        elif position[0] < pre_position[0]:
            diretion = "U"
        elif position[1] > pre_position[1]:
            diretion = "R"
        elif position[1] < pre_position[1]:
            diretion = "L"
        else:
            diretion = "S"
        return diretion

    def load_state(self, state):
        if self.origin_map is None:
            self.create_map(state["player1"], state["player2"], state["walls"])
            if self.NAME == state["player1"]["name"]:
                self.player = "player1"
            else:
                self.player = "player2"
            self.my_home = [state[self.player]["home_x"], state[self.player]["home_y"]]
            self.pre_position = self.my_home

        self.n_jobs = state[self.player]["n_jobs"]
        self.real_map = copy.deepcopy(self.origin_map)
        self.max_parcel = self.add_jobs(self.real_map, state["jobs"], self.pre_position)

    def step(self):
        # Not at home
        if (self.pre_position[0], self.pre_position[1]) != (self.my_home[0], self.my_home[1]):
            a_star = AStar(Array2D(self.origin_map),Point(self.pre_position[0], self.pre_position[1]), Point(self.my_home[0], self.my_home[1]),passTag=-1)
            path_list = a_star.start()
        # At home
        else:
            path_list = list()

        if self.n_jobs == self.MAX_BAGS or (self.MAX_STEPS-self.steps)-(len(path_list)) <= 1 or (
                len(path_list) < 3 and self.n_jobs > 5):
            self.go_home = True

        if self.go_home and (self.pre_position[0], self.pre_position[1]) != (self.my_home[0], self.my_home[1]):
            if not self.home_path:
                self.home_path = list()
                for point in path_list:
                    self.home_path.append([point.x, point.y])
            next_position = self.home_path.pop(0)
            direction = self.move(next_position, self.pre_position)
            self.pre_position = next_position
            if not self.home_path:
                self.go_home = False
                self.home_path = list()

        else:
            exploration = {"max_reward": -1, "max_trace": [], "fake_reward": -1}
            delta = [[0, 1], [0, -1], [1, 0], [-1, 0]]
            max_step = self.max_parcel
            if max_step > self.MAX_SEARCH:
                max_step = self.MAX_SEARCH

            for d in delta:
                self.explore(max_step, [self.pre_position[0]+d[0], self.pre_position[1]+d[1]],self.help_map, self.real_map,[self.pre_position[0]+d[0], self.pre_position[1]+d[1]], 0, 0, exploration, 1)

            direction = self.move(exploration["max_trace"], self.pre_position)
            self.pre_position = exploration["max_trace"]
        self.steps += 1
        return direction

    def take_action(self, state):
        self.load_state(state)
        direction = self.step()
        return direction


# state = {'player1': {'name': 'p1', 'x': 5, 'y': 6, 'home_x': 5, 'home_y': 5, 'n_jobs': 1, 'value': 6.0, 'score': 0},
#          'player2': {'name': 'p2', 'x': 2, 'y': 5, 'home_x': 6, 'home_y': 6, 'n_jobs': 2, 'value': 20.0, 'score': 0},
#          'walls': [{'x': 0, 'y': 8}, {'x': 1, 'y': 1}, {'x': 1, 'y': 10}, {'x': 1, 'y': 11},
#                    {'x': 3, 'y': 11}, {'x': 4, 'y': 0}, {'x': 4, 'y': 10}, {'x': 5, 'y': 0},
#                    {'x': 5, 'y': 1}, {'x': 5, 'y': 4}, {'x': 5, 'y': 7}, {'x': 5, 'y': 9},
#                    {'x': 6, 'y': 5}, {'x': 7, 'y': 0}, {'x': 7, 'y': 1}, {'x': 7, 'y': 8},
#                    {'x': 8, 'y': 3}, {'x': 8, 'y': 4}, {'x': 9, 'y': 1}, {'x': 9, 'y': 3},
#                    {'x': 9, 'y': 9}, {'x': 9, 'y': 10}, {'x': 10, 'y': 3}, {'x': 10, 'y': 6}],
#          'jobs': [{'x': 0, 'y': 2, 'value': 8.0}, {'x': 0, 'y': 4, 'value': 11.0}, {'x': 1, 'y': 2, 'value': 11.0},
#                   {'x': 1, 'y': 4, 'value': 7.0}, {'x': 1, 'y': 7, 'value': 11.0}, {'x': 1, 'y': 8, 'value': 9.0},
#                   {'x': 2, 'y': 7, 'value': 11.0}, {'x': 3, 'y': 6, 'value': 11.0}, {'x': 3, 'y': 8, 'value': 10.0},
#                   {'x': 4, 'y': 9, 'value': 6.0}, {'x': 4, 'y': 11, 'value': 6.0}, {'x': 5, 'y': 3, 'value': 7.0},
#                   {'x': 5, 'y': 8, 'value': 10.0}, {'x': 6, 'y': 8, 'value': 6.0}, {'x': 6, 'y': 9, 'value': 7.0},
#                   {'x': 6, 'y': 10, 'value': 8.0}, {'x': 6, 'y': 11, 'value': 8.0}, {'x': 7, 'y': 6, 'value': 11.0},
#                   {'x': 8, 'y': 1, 'value': 9.0}, {'x': 9, 'y': 5, 'value': 8.0}, {'x': 9, 'y': 11, 'value': 12.0},
#                   {'x': 10, 'y': 5, 'value': 7.0}, {'x': 10, 'y': 9, 'value': 10.0}, {'x': 11, 'y': 3, 'value': 7.0}]}


# if __name__ == '__main__':
#     conf = {
#         'world_size': 12,
#         'capacity': 10,
#         'player1_home': (5, 5),
#         'player2_home': (6, 6),
#         'num_walls': 24,
#         'num_jobs': 24,
#         'value_range': (6, 12),
#         'max_steps': 200
#     }
#     p2_actions = ['S', 'U', 'D', 'L', 'R']
#     env = Env("", "p1", "p2", conf, random.Random(50))
#     env.reset()
#     env.render()
#
#     env_init = env.get_state()
#
#     s = time.time()
#     bot = MyDFS()
#     bot.create_map(env_init["player1"], env_init["player2"], env_init["walls"])
#     bot.real_map = copy.deepcopy(bot.origin_map)
#
#     if bot.NAME == env_init["player1"]["name"]:
#         my_home = [env_init["player1"]["home_x"], env_init["player1"]["home_y"]]
#     else:
#         my_home = [env_init["player2"]["home_x"], env_init["player2"]["home_y"]]
#     pre_position = my_home
#     max_parcel = bot.add_jobs(bot.real_map, env_init["jobs"], pre_position)
#     print("create map time:", time.time()-s)
#
#     for i in range(conf["max_steps"]):
#         s = time.time()
#
#         if (pre_position[0], pre_position[1]) != (my_home[0], my_home[1]):
#             a_star = AStar(Array2D(bot.origin_map),
#                            Point(pre_position[0], pre_position[1]), Point(my_home[0], my_home[1]),
#                            passTag=-1)
#             path_list = a_star.start()
#         else:
#             path_list = list()
#
#         if bot.n_jobs == 10 or (conf["max_steps"]-i)-(len(path_list)) <= 1 or (
#                 len(path_list) < 3 and bot.n_jobs > 5):
#             bot.go_home = True
#
#         if bot.go_home:
#             if not bot.home_path:
#                 bot.home_path = list()
#                 for point in path_list:
#                     bot.home_path.append([point.x, point.y])
#             next_position = bot.home_path.pop(0)
#             direction = bot.move(next_position, pre_position)
#             pre_position = next_position
#             if not bot.home_path:
#                 bot.go_home = False
#                 bot.home_path = list()
#
#         else:
#             exploration = {"max_reward": -1, "max_trace": [], "fake_reward": -1}
#             delta = [[0, 1], [0, -1], [1, 0], [-1, 0]]
#             max_step = max_parcel
#             if max_step > bot.MAX_SEARCH:
#                 max_step = bot.MAX_SEARCH
#
#             for d in delta:
#                 bot.explore(max_step, [pre_position[0]+d[0], pre_position[1]+d[1]], bot.help_map, bot.real_map,
#                             [pre_position[0]+d[0], pre_position[1]+d[1]], 0, 0, exploration, 1)
#
#             direction = bot.move(exploration["max_trace"], pre_position)
#             pre_position = exploration["max_trace"]
#
#         state = env.step(bot.NAME, direction)
#         bot.n_jobs = state["player1"]["n_jobs"]
#         bot.real_map = copy.deepcopy(bot.origin_map)
#         max_parcel = bot.add_jobs(bot.real_map, state["jobs"], pre_position)
#         _ = env.step("p2", random.choice(p2_actions))
#         print("coord:", state["player1"]["x"], state["player1"]["y"])
#         print("bag:", state["player1"]["n_jobs"])
#         print("value:", state["player1"]["value"])
#         print("score:", state["player1"]["score"])
#         print("step:", i)
#         print("step time:", time.time()-s)
#         env.render()
#
#     print("final score:", state["player1"]["score"])


def battle():
    from lib.py.bot import Bot
    conf = {
        'world_size': 12,
        'capacity': 10,
        'player1_home': (5, 5),
        'player2_home': (6, 6),
        'num_walls': 24,
        'num_jobs': 24,
        'value_range': (6, 12),
        'max_steps': 200
    }
    env = Env("", "p1", "p2", conf, random.Random(20))
    env.reset()
    env.render()

    env_init = env.get_state()

    # py bot
    py_bot = Bot("p2")
    py_bot.load_state(env_init)

    # sam bot
    sam_bot = MyDFS("p1")
    sam_bot.load_state(env_init)

    for i in range(conf["max_steps"]):

        # sam bot step
        direction = sam_bot.step()

        p1_state = env.step(sam_bot.NAME, direction)
        sam_bot.load_state(p1_state)

        # py bot step
        try:
            # state = env.step("p1", p1_action.pop())
            aa = py_bot.bot_action()
            p2_state = env.step("p2", aa)
            # print("Action:", aa)
            # print("TO GO:", py_bot.to_go_actions)
        except:
            feasible_actions = py_bot.where_to_go(py_bot.us["home_x"], py_bot.us["home_y"])
            # print("Feasible:", feasible_actions)
            aa = random.choice(feasible_actions)
            # print("Action:", aa)
            p2_state = env.step("p2", aa)
        py_bot.load_state(p2_state)

        print("step:", i)

        print("p1")
        print("bag:", p1_state["player1"]["n_jobs"])
        print("value:", p1_state["player1"]["value"])
        print("score:", p1_state["player1"]["score"])

        print("p2")
        print("bag:", p2_state["player2"]["n_jobs"])
        print("value:", p2_state["player2"]["value"])
        print("score:", p2_state["player2"]["score"])
        print("stop steps:", py_bot.stop_flag)

        env.render()

    print("final:")
    print("p1 score:", p1_state["player1"]["score"])
    print("p2 score:", p2_state["player2"]["score"])


def self_battle():
    conf = {
        'world_size': 12,
        'capacity': 10,
        'player1_home': (5, 5),
        'player2_home': (6, 6),
        'num_walls': 24,
        'num_jobs': 24,
        'value_range': (6, 12),
        'max_steps': 200
    }
    env = Env("", "p1", "p2", conf, random.Random(random.randint(1, 1000)))
    env.reset()
    env.render()

    env_init = env.get_state()

    # sam bot1
    sam_bot1 = MyDFS("p1", max_bags=10)

    # sam bot2
    sam_bot2 = MyDFS("p2", max_bags=10)
    p2_state = env_init

    p1_max_time = 0
    p2_max_time = 0

    for i in range(conf["max_steps"]):

#---------------
        # from lib.RL.agent import Agent
        # agt = Agent(10, 12, None, 'p2', 200)
        # _p2_state = p2_state['jobs'], p2_state['walls'], p2_state['player2'], p2_state['player1'], 12
        # _, moves = agt.get_movement(None, False, rule_state=_p2_state)
        # move = moves[0]
# ---------------

        # sam bot1 step
        s = time.time()
        direction = sam_bot1.take_action(p2_state)
        p1_time = time.time() - s
        p1_max_time = max(p1_time, p1_max_time)
        p1_state = env.step(sam_bot1.NAME, direction)
        # sam_bot1.load_state(p1_state)

        # sam bot2 step
        s = time.time()
        direction = sam_bot2.take_action(p1_state)
        p2_time = time.time() - s
        p2_max_time = max(p2_time, p2_max_time)
        p2_state = env.step(sam_bot2.NAME, direction)
        # sam_bot2.load_state(p2_state)

        print("step:", i)

        print("p1")
        print("bag:", p1_state["player1"]["n_jobs"])
        print("value:", p1_state["player1"]["value"])
        print("score:", p1_state["player1"]["score"])
        print("time:", p1_time)

        print("p2")
        print("bag:", p2_state["player2"]["n_jobs"])
        print("value:", p2_state["player2"]["value"])
        print("score:", p2_state["player2"]["score"])
        print("time:", p2_time)

        env.render()

    print("final:")
    print("p1 score:", p1_state["player1"]["score"])
    print("p2 score:", p2_state["player2"]["score"])

    # sam_bot1.reset()
    # sam_bot2.reset()

    return p1_state["player1"]["score"], p2_state["player2"]["score"], p1_max_time, p2_max_time


def sr_battle():
    conf = {
        'world_size': 12,
        'capacity': 10,
        'player1_home': (5, 5),
        'player2_home': (6, 6),
        'num_walls': 24,
        'num_jobs': 24,
        'value_range': (6, 12),
        'max_steps': 200
    }
    env = Env("", "p1", "p2", conf, random.Random(random.randint(1, 1000)))
    env.reset()
    env.render()

    env_init = env.get_state()

    # sam bot
    sam_bot1 = MyDFS("p1", max_bags=10)
    p2_state = env_init

    # sr bot
    from lib.RL.agent import Agent
    agt = Agent(10, 12, None, 'p2', 200)

    p1_max_time = 0
    p2_max_time = 0

    for i in range(conf["max_steps"]):

#---------------
        # from lib.RL.agent import Agent
        # agt = Agent(10, 12, None, 'p2', 200)
        # _p2_state = p2_state['jobs'], p2_state['walls'], p2_state['player2'], p2_state['player1'], 12
        # _, moves = agt.get_movement(None, False, rule_state=_p2_state)
        # move = moves[0]
# ---------------

        # sam bot1 step
        s = time.time()
        direction = sam_bot1.take_action(p2_state)
        p1_time = time.time() - s
        p1_max_time = max(p1_time, p1_max_time)
        p1_state = env.step(sam_bot1.NAME, direction)
        # sam_bot1.load_state(p1_state)

        # sr bot2 step
        s = time.time()
        _p1_state = p1_state['jobs'], p1_state['walls'], p1_state['player2'], p1_state['player1'], 12
        _, moves = agt.get_movement(None, False, rule_state=_p1_state)
        move = moves[0]
        p2_time = time.time() - s
        p2_max_time = max(p2_time, p2_max_time)
        p2_state = env.step("p2", move)
        # sam_bot2.load_state(p2_state)

        print("step:", i)

        print("p1")
        print("bag:", p1_state["player1"]["n_jobs"])
        print("value:", p1_state["player1"]["value"])
        print("score:", p1_state["player1"]["score"])
        print("time:", p1_time)

        print("p2")
        print("bag:", p2_state["player2"]["n_jobs"])
        print("value:", p2_state["player2"]["value"])
        print("score:", p2_state["player2"]["score"])
        print("time:", p2_time)

        env.render()

    print("final:")
    print("p1 score:", p1_state["player1"]["score"])
    print("p2 score:", p2_state["player2"]["score"])

    # sam_bot1.reset()
    # sam_bot2.reset()

    return p1_state["player1"]["score"], p2_state["player2"]["score"], p1_max_time, p2_max_time


# battle()

# p1_wins = 0
# p2_wins = 0
# p1_mean = 0.0
# p2_mean = 0.0
# p1_max_time = 0
# p2_max_time = 0
# rounds = 50
# for i in range(rounds):
#     print("round", i+1)
#     p1_score, p2_score, p1_time, p2_time = self_battle()
#     p1_mean += p1_score
#     p2_mean += p2_score
#     if p1_score > p2_score:
#         p1_wins += 1
#     else:
#         p2_wins += 1
#     p1_max_time = max(p1_time, p1_max_time)
#     p2_max_time = max(p2_time, p2_max_time)
#     print()

# print("p1 vs p2: {}:{}".format(p1_wins, p2_wins),)
# print("p1 mean score:", p1_mean/rounds)
# print("p2 mean score:", p2_mean/rounds)
# print("p1 max time:", p1_max_time)
# print("p2 max time:", p2_max_time)


