# -*- coding: utf-8 -*-
import copy
import random
import time
from env import Player, Env
from a_star import Array2D, AStar, Point


GAME_CONF = {
    'world_size': 12,
    'capacity': 10,
    'player1_home': (5, 5),
    'player2_home': (6, 6),
    'num_walls': 24,
    'num_jobs': 24,
    'value_range': (6, 12),
    'max_steps': 200
    }

class Killer(object):
    def __init__(self, team_name):
        self.team_name = team_name
        self.us = None
        self.enemy = None
        self.data = None
        # Step and Job count
        self.step_count = 0

        # A-star map
        self.map2d = None
        self.path_dict = {}
        # Enemy path
        self.enemy_map2d = None
        self.enemy_dict = {}

    def update_state(self, data):
        self.data = data
        # package based DFS
        self.best_path = None
        self.max_reward = 0
        self.step_count += 1
        # Distinguish Data Flow
        if self.us is None or self.enemy is None:
            if data["player1"]["name"] == self.team_name:
                self.us = "player1"
                self.enemy = "player2"
            else:
                self.us = "player2"
                self.enemy = "player1"
        # Initialize shortest path map
        if self.map2d is None:
            self._init_shortest_path(data)
        if self.enemy_map2d is None:
            self._init_enemy_path(data)

        self.jobs_ranking()
        self.enemy_dectect()
        # self.nearest_10_jobs = self.pick_up_ranking(self.nearest_10_jobs)

    def _init_shortest_path(self, data):
        map2d = Array2D(12, 12)
        # init obstacles
        for w in data["walls"]:
            map2d[w["y"]][w["x"]] = 1
        # init enemy home
        map2d[data[self.enemy]["home_y"]][data[self.enemy]["home_x"]] = 1
        # Generate all location
        for x in range(12):
            for y in range(12):
                for m in range(12):
                    for n in range(12):
                        # Same Point No path
                        if m == x and n == y:
                            continue
                        # One of walls No path
                        elif map2d[m][n] == 1 or map2d[x][y] == 1:
                            continue
                        else:
                            astar = AStar(map2d, Point(x, y), Point(m, n))
                            try:
                                path_list = [(p.y, p.x) for p in astar.start()]
                            except:
                                continue
                            self.path_dict[(y, x, n, m)] = path_list
        self.map2d = map2d

    def _init_enemy_path(self, data):
        map2d = Array2D(12, 12)
        # init obstacles
        for w in data["walls"]:
            map2d[w["y"]][w["x"]] = 1
        # init us home
        map2d[data[self.us]["home_y"]][data[self.us]["home_x"]] = 1
        # Generate all location
        for x in range(12):
            for y in range(12):
                for m in range(12):
                    for n in range(12):
                        # Same Point No path
                        if m == x and n == y:
                            continue
                        # One of walls No path
                        elif map2d[m][n] == 1 or map2d[x][y] == 1:
                            continue
                        else:
                            astar = AStar(map2d, Point(x, y), Point(m, n))
                            try:
                                path_list = [(p.y, p.x) for p in astar.start()]
                            except:
                                continue
                            self.enemy_dict[(y, x, n, m)] = path_list
        self.enemy_map2d = map2d

    def global_find_cluster(self):
        # TODO:
        pass

    def enemy_dectect(self):
        self.stop_and_go_home = False
        if self.data[self.enemy]["n_jobs"] == 10:
            enemy_to_home = len(self.enemy_dict[(self.data[self.enemy]["x"], self.data[self.enemy]["y"], self.data[self.enemy]["home_x"], self.data[self.enemy]["home_y"])])
            try:
                if enemy_to_home > len(self.path_dict[(self.data[self.us]["x"], self.data[self.us]["y"], self.data[self.us]["home_x"], self.data[self.us]["home_y"])]):
                    if self.step_count > 25 or self.data[self.us]["n_jobs"] > 3:
                        self.stop_and_go_home = True
            except:
                self.stop_and_go_home = False

    def jobs_ranking(self):
        # Nearest 10 jobs
        me_to_jobs = {}
        self.nearest_10_jobs = []
        for _, j in enumerate(self.data["jobs"]):
            if self.data[self.us]["x"] == j["x"] and self.data[self.us]["y"] == j["y"]:
                continue
            try:
                distance = len(self.path_dict[(self.data[self.us]["x"], self.data[self.us]["y"], j["x"], j["y"])])
                enemy_dist = len(self.enemy_dict[(self.data[self.enemy]["x"], self.data[self.enemy]["y"], j["x"], j["y"])])
                if distance > enemy_dist:
                    continue
            except:
                continue
            me_to_jobs[(j["x"], j["y"], j["value"])] = distance
        me_to_jobs_len = [v for _, v in me_to_jobs.items()]
        me_to_jobs_len = list(set(me_to_jobs_len))
        for l in sorted(me_to_jobs_len):
            if len(self.nearest_10_jobs) == 10:
                break
            possible_jobs = [{"x": k[0], "y": k[1], "value": k[2]} for k, v in me_to_jobs.items() if v == l]
            self.nearest_10_jobs += possible_jobs

    def pick_up_ranking(self, jobs):
        choice = []
        ret = {}
        for to_pick in jobs:
            ori_x, ori_y = to_pick["x"], to_pick["y"]
            ret[(ori_x, ori_y)] = 0
            for j in self.data["jobs"]:
                if j["x"] == ori_x and j["y"] == ori_y:
                    continue
                if len(self.path_dict[(ori_x, ori_y, j["x"], j["y"])]) <= 7:
                    ret[(ori_x, ori_y)] += 1
        score = sorted([v for _, v in ret.items()])
        for s in list(set(score)):
            choice = [k for k, v in ret.items() if v == s]
            if len(choice) > 1:
                tmp = copy.deepcopy(jobs)
                i = 1
                try:
                    less_jobs = tmp.remove(choice[-i])
                except:
                    tmp.pop()
                    less_jobs = tmp
                choice = self.pick_up_ranking(less_jobs)
            else:
                return choice

    def find_path(self, x_start, y_start, x_end, y_end):
        """Set up to_go_actions

        Arguments:
            x_start {[type]} -- [description]
            y_start {[type]} -- [description]
            x_end {[type]} -- [description]
            y_end {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        # TODO: Find the cluster

        # Find the nearest job and avoid enemy
        if x_start == x_end and y_start == y_end:
            try:
                return self.find_path(x_start, y_start, self.nearest_10_jobs[0]["x"], self.nearest_10_jobs[0]["y"]).pop(0)
            except:
                return self.find_path(x_start, y_start, self.nearest_10_jobs[0]["x"], self.nearest_10_jobs[0]["y"])

        to_go_actions = []
        points_list = self.path_dict[(x_start, y_start, x_end, y_end)]
        for i, (xi, yi) in enumerate(points_list):
            # The first two actions set
            # if len(to_go_actions) == 2:
            #     break
            if i == 0:
                if xi == x_start:
                    if yi > y_start:
                        to_go_actions.append("R")
                    elif yi < y_start:
                        to_go_actions.append("L")
                elif yi == y_start:
                    if xi > x_start:
                        to_go_actions.append("D")
                    elif xi < x_start:
                        to_go_actions.append("U")
            else:
                # Previous location
                x_pre, y_pre = points_list[i-1]
                if xi == x_pre:
                    if yi > y_pre:
                        to_go_actions.append("R")
                    elif yi < y_pre:
                        to_go_actions.append("L")
                elif yi == y_pre:
                    if xi > x_pre:
                        to_go_actions.append("D")
                    elif xi < x_pre:
                        to_go_actions.append("U")
        return to_go_actions

    def bot_action(self):
        """Main Function

        Returns:
            [type] -- [description]
        """
        if self.stop_and_go_home:
            if self.data[self.us]["n_jobs"] > 2:
                actions = self.find_path(self.data[self.us]["x"], self.data[self.us]["y"], self.data[self.us]["home_x"], self.data[self.us]["home_y"])
                if isinstance(actions, list):
                    return actions.pop(0)
                else:
                    return actions

        if len(self.find_path(self.data[self.us]["x"], self.data[self.us]["y"], self.data[self.us]["home_x"], self.data[self.us]["home_y"])) <= 3:
            if self.data[self.us]["n_jobs"] > 1:
                return self.find_path(self.data[self.us]["x"], self.data[self.us]["y"], self.data[self.us]["home_x"], self.data[self.us]["home_y"]).pop(0)

        if len(self.nearest_10_jobs) == 0:
            self.nearest_10_jobs = {"x": self.data[self.enemy]["x"], "y": self.data[self.enemy]["y"], "value": 0.0}

        if self.data[self.us]["n_jobs"] == 10:
            actions = self.find_path(self.data[self.us]["x"], self.data[self.us]["y"], self.data[self.us]["home_x"], self.data[self.us]["home_y"])
            if isinstance(actions, list):
                return actions.pop(0)
            else:
                return actions

        if self.step_count + len(self.find_path(self.data[self.us]["x"], self.data[self.us]["y"], self.data[self.us]["home_x"], self.data[self.us]["home_y"])) >= 199:
            actions = self.find_path(self.data[self.us]["x"], self.data[self.us]["y"], self.data[self.us]["home_x"], self.data[self.us]["home_y"])
            if isinstance(actions, list):
                return actions.pop(0)
            else:
                return actions

        if self.data[self.us]["n_jobs"] > 5:
            if self.data[self.us]["x"] != self.data[self.us]["home_x"] and  self.data[self.us]["y"] != self.data[self.us]["home_y"]:
                return_path = self.find_path(self.data[self.us]["x"], self.data[self.us]["y"], self.data[self.us]["home_x"], self.data[self.us]["home_y"])
            next_job_path = self.find_path(self.data[self.us]["x"], self.data[self.us]["y"], self.nearest_10_jobs[0]["x"], self.nearest_10_jobs[0]["y"])
            try:
                if (self.data[self.us]["value"] + self.nearest_10_jobs[0]["value"]) / len(return_path) + len(next_job_path) * 2 > 2.5:
                    return next_job_path.pop(0)
                else:
                    try:
                        return return_path.pop(0)
                    except:
                        return next_job_path.pop(0)
            except:
                return next_job_path.pop(0)

        actions = self.find_path(self.data[self.us]["x"], self.data[self.us]["y"], self.nearest_10_jobs[0]["x"], self.nearest_10_jobs[0]["y"])
        if len(actions) != 0:
            return actions.pop(0)
        else:
            return


def main():
    bot = Killer("p1")
    # bot2 = DFS("p2")
    env = Env("", "p1", "p2", GAME_CONF, random.Random(20))
    state = env.reset()
    # bot2.load_state(state)
    env.render()
    while True:
        # p1 action
        bot.update_state(state)
        action = bot.bot_action()
        state = env.step("p1", action)

        env.render()
        print("Action:", action)
        print("N_JOBS:", bot.data[bot.us]["n_jobs"])
        print("Step:", bot.step_count)
        print("Values:", bot.data[bot.us]["value"])
        print("Scores:", bot.data[bot.us]["score"])

        # Player 2
        _ = env.step("p2", "S")
        # state2 = env.step("p2", bot2.bot_action())
        # bot2.load_state(state2)
        if env.done:
            print("**********P1 Final Score:", bot.data[bot.us]["score"])
            # print("**********P2 Final Score:", bot2.us["score"])
            break
        time.sleep(0.3)


if __name__ == "__main__":
    main()
