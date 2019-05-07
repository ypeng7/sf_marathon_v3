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
import numpy as np


class MyDFS:
    def __init__(self):
        self.NAME = "p1"
        self.MAPSIZE = 12
        self.POINT = 1.0
        self.DIVISOR = 100.0
        self.RATE = 0.7
        self.RATE_E = 0.95
        self.MAX_SEARCH = 9
        self.n_jobs = 0
        self.go_home = False
        self.origin_map = None
        self.real_map = None
        self.help_map = None
        self.home_path = None

    def create_map(self, p1, p2, walls):
        self.origin_map = list()
        for i in range(self.MAPSIZE):
            row = list()
            for j in range(self.MAPSIZE):
                row.append(-1)
            self.origin_map.append(row)
        self.origin_map = np.array(self.origin_map)
        self.walls = dict()
        for w in walls:
            self.origin_map[w["x"], w["y"]] = -2
            self.walls[(w["x"], w["y"])] = 1
        if p1["name"] == self.NAME:
            self.origin_map[p2["home_x"], p2["home_y"]] = -2
        else:
            self.origin_map[p1["home_x"], p1["home_y"]] = -2
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
                if map[i, j] != -2:
                    map[i, j] = 0

    def _add_weight(self, x, map, position, p):
        if position[0] < 0 or position[0] >= self.MAPSIZE or position[1] < 0 or position[1] >= self.MAPSIZE or \
                map[position[0], position[1]] == -2:
            return
        map[position[0], position[1]] += p / x

    def add_jobs(self, map, jobs, position):
        max_parcel = 0
        for j in jobs:
            map[j["x"], j["y"]] = j["value"]
            max_parcel = max(j["value"], max_parcel)
        map[position[0], position[1]] = 0
        return max_parcel

    def explore(self, max_parcel, position, help_map, real_map, tmp_trace, tmp_reward, fake_reward, exploration, step):
        if step >= max_parcel or position[0] < 0 or position[0] >= self.MAPSIZE or position[1] < 0 or position[
            1] >= self.MAPSIZE or real_map[position[0]][position[1]] == -2:
            return

        tmp_reward += pow(self.RATE_E, step) * (real_map[position[0], position[1]] - step) if real_map[position[0],
                                                                                             position[1]] > step else 0
        fake_reward += pow(self.RATE_E, step) * (real_map[position[0], position[1]] - step) if real_map[position[0],
                                                                                              position[1]] > step else 0
        real = real_map[position[0], position[1]]
        real_map[position[0], position[1]] = 0

        add_on = 0
        add_on += help_map[position[0], position[1]] * pow(self.RATE, step)
        fake_reward += add_on

        if fake_reward > exploration["fake_reward"]:
            exploration["fake_reward"] = fake_reward
            exploration["max_reward"] = tmp_reward
            exploration["max_trace"] = tmp_trace

        delta = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        for d in delta:
            self.explore(max_parcel, [position[0] + d[0], position[1] + d[1]], help_map, real_map, tmp_trace, tmp_reward,
                    fake_reward, exploration, step + 1)

        real_map[position[0], position[1]] = real

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

    # def step(self):


state = {'player1': {'name': 'p1', 'x': 5, 'y': 6, 'home_x': 5, 'home_y': 5, 'n_jobs': 1, 'value': 6.0, 'score': 0},
         'player2': {'name': 'p2', 'x': 2, 'y': 5, 'home_x': 6, 'home_y': 6, 'n_jobs': 2, 'value': 20.0, 'score': 0},
         'walls': [{'x': 0, 'y': 8}, {'x': 1, 'y': 1}, {'x': 1, 'y': 10}, {'x': 1, 'y': 11},
                   {'x': 3, 'y': 11}, {'x': 4, 'y': 0}, {'x': 4, 'y': 10}, {'x': 5, 'y': 0},
                   {'x': 5, 'y': 1}, {'x': 5, 'y': 4}, {'x': 5, 'y': 7}, {'x': 5, 'y': 9},
                   {'x': 6, 'y': 5}, {'x': 7, 'y': 0}, {'x': 7, 'y': 1}, {'x': 7, 'y': 8},
                   {'x': 8, 'y': 3}, {'x': 8, 'y': 4}, {'x': 9, 'y': 1}, {'x': 9, 'y': 3},
                   {'x': 9, 'y': 9}, {'x': 9, 'y': 10}, {'x': 10, 'y': 3}, {'x': 10, 'y': 6}],
         'jobs': [{'x': 0, 'y': 2, 'value': 8.0}, {'x': 0, 'y': 4, 'value': 11.0}, {'x': 1, 'y': 2, 'value': 11.0},
                  {'x': 1, 'y': 4, 'value': 7.0}, {'x': 1, 'y': 7, 'value': 11.0}, {'x': 1, 'y': 8, 'value': 9.0},
                  {'x': 2, 'y': 7, 'value': 11.0}, {'x': 3, 'y': 6, 'value': 11.0}, {'x': 3, 'y': 8, 'value': 10.0},
                  {'x': 4, 'y': 9, 'value': 6.0}, {'x': 4, 'y': 11, 'value': 6.0}, {'x': 5, 'y': 3, 'value': 7.0},
                  {'x': 5, 'y': 8, 'value': 10.0}, {'x': 6, 'y': 8, 'value': 6.0}, {'x': 6, 'y': 9, 'value': 7.0},
                  {'x': 6, 'y': 10, 'value': 8.0}, {'x': 6, 'y': 11, 'value': 8.0}, {'x': 7, 'y': 6, 'value': 11.0},
                  {'x': 8, 'y': 1, 'value': 9.0}, {'x': 9, 'y': 5, 'value': 8.0}, {'x': 9, 'y': 11, 'value': 12.0},
                  {'x': 10, 'y': 5, 'value': 7.0}, {'x': 10, 'y': 9, 'value': 10.0}, {'x': 11, 'y': 3, 'value': 7.0}]}


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
    env = Env("", "p1", "p2", conf, random.Random(50))
    env.reset()
    env.render()

    env_init = env.get_state()

    py_bot = Bot("p2")
    py_bot.load_state(env_init)

    # sam bot
    bot = MyDFS()
    bot.create_map(env_init["player1"], env_init["player2"], env_init["walls"])
    bot.real_map = copy.deepcopy(bot.origin_map)

    if bot.NAME == env_init["player1"]["name"]:
        my_home = [env_init["player1"]["home_x"], env_init["player1"]["home_y"]]
    else:
        my_home = [env_init["player2"]["home_x"], env_init["player2"]["home_y"]]
    pre_position = my_home
    max_parcel = bot.add_jobs(bot.real_map, env_init["jobs"], pre_position)

    for i in range(conf["max_steps"]):

        # sam bot step
        if (pre_position[0], pre_position[1]) != (my_home[0], my_home[1]):
            a_star = AStar(Array2D(bot.origin_map),
                           Point(pre_position[0], pre_position[1]), Point(my_home[0], my_home[1]),
                           passTag=-1)
            path_list = a_star.start()
        else:
            path_list = list()

        if bot.n_jobs == 10 or (conf["max_steps"]-i)-(len(path_list)) <= 1 or (
                len(path_list) < 3 and bot.n_jobs > 5):
            bot.go_home = True

        if bot.go_home:
            if not bot.home_path:
                bot.home_path = list()
                for point in path_list:
                    bot.home_path.append([point.x, point.y])
            next_position = bot.home_path.pop(0)
            direction = bot.move(next_position, pre_position)
            pre_position = next_position
            if not bot.home_path:
                bot.go_home = False
                bot.home_path = list()

        else:
            exploration = {"max_reward": -1, "max_trace": [], "fake_reward": -1}
            delta = [[0, 1], [0, -1], [1, 0], [-1, 0]]
            max_step = max_parcel
            if max_step > bot.MAX_SEARCH:
                max_step = bot.MAX_SEARCH

            for d in delta:
                bot.explore(max_step, [pre_position[0]+d[0], pre_position[1]+d[1]], bot.help_map, bot.real_map,
                            [pre_position[0]+d[0], pre_position[1]+d[1]], 0, 0, exploration, 1)

            direction = bot.move(exploration["max_trace"], pre_position)
            pre_position = exploration["max_trace"]

        p1_state = env.step(bot.NAME, direction)
        bot.n_jobs = p1_state["player1"]["n_jobs"]
        bot.real_map = copy.deepcopy(bot.origin_map)
        max_parcel = bot.add_jobs(bot.real_map, p1_state["jobs"], pre_position)

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


battle()
