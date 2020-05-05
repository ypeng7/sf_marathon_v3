# -*- coding: utf-8 -*-
# @author: Yue Peng
# @email: yuepaang@gmail.com
# @create by: 2018-12-13 17:52:25
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
        pass

    def enemy_dectect(self):
        self.stop_and_go_home = False
        # TODO: 8 best
        if self.data[self.enemy]["n_jobs"] == 8:
            enemy_to_home = len(self.enemy_dict[(self.data[self.enemy]["x"], self.data[self.enemy]["y"], self.data[self.enemy]["home_x"], self.data[self.enemy]["home_y"])])
            try:
                if enemy_to_home > len(self.path_dict[(self.data[self.us]["x"], self.data[self.us]["y"], self.data[self.us]["home_x"], self.data[self.us]["home_y"])]):
                    if self.step_count > 25 or self.data[self.us]["n_jobs"] > 2:
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
            return self.find_path(x_start, y_start, self.nearest_10_jobs[0]["x"], self.nearest_10_jobs[0]["y"]).pop(0)
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
        # find the enemy
        if self.stop_and_go_home:
            return self.find_path(self.data[self.us]["x"], self.data[self.us]["y"], self.data[self.us]["home_x"], self.data[self.us]["home_y"]).pop(0)
        # GoGo home
        if self.data[self.us]["n_jobs"] > 4 and len(self.path_dict[(self.data[self.us]["x"], self.data[self.us]["y"], self.data[self.us]["home_x"], self.data[self.us]["home_y"])]) <= 3:
            return self.find_path(self.data[self.us]["x"], self.data[self.us]["y"], self.data[self.us]["home_x"], self.data[self.us]["home_y"]).pop(0)
        # 6 can go
        if 10 - self.data[self.us]["n_jobs"] > 6:
            self.best_route(6)
        else:
            self.best_route(10-self.data[self.us]["n_jobs"])

        # GOHOME
        if self.best_path is None or len(self.best_path) == 0:
            return self.find_path(self.data[self.us]["x"], self.data[self.us]["y"], self.data[self.us]["home_x"], self.data[self.us]["home_y"]).pop(0)

        if len(self.find_path(self.data[self.us]["x"], self.data[self.us]["y"], self.data[self.us]["home_x"], self.data[self.us]["home_y"])) + self.step_count >= 199:
            # print("BEST", self.best_path)
            # print("HOME", self.find_path(self.data[self.us]["x"], self.data[self.us]["y"], self.data[self.us]["home_x"], self.data[self.us]["home_y"]))
            # print(len(self.best_path))
            # print(len(self.find_path(self.data[self.us]["x"], self.data[self.us]["y"], self.data[self.us]["home_x"], self.data[self.us]["home_y"])))
            #FIXME:
            if len(self.best_path) > len(self.find_path(self.data[self.us]["x"], self.data[self.us]["y"], self.data[self.us]["home_x"], self.data[self.us]["home_y"])): 
                try:
                    return self.find_path(self.data[self.us]["x"], self.data[self.us]["y"], self.data[self.us]["home_x"], self.data[self.us]["home_y"]).pop(0)
                except:
                    return self.find_path(self.data[self.us]["x"], self.data[self.us]["y"], self.data[self.us]["home_x"], self.data[self.us]["home_y"])
            else:
                first_point = self.best_path.pop(0)
                return self.find_path(self.data[self.us]["x"], self.data[self.us]["y"], first_point[0], first_point[1]).pop(0)

        first_point = self.best_path.pop(0)
        return self.find_path(self.data[self.us]["x"], self.data[self.us]["y"], first_point[0], first_point[1]).pop(0)

    def best_route(self, capacity):
        tasks = []
        self.best_path = None
        self.max_reward = 0
        dummy_job = {
            "x": self.data[self.us]['x'],
            "y": self.data[self.us]['y'],
            "value": 0.}
        tasks.append(dummy_job)
        # for _ in self.data["jobs"]:
        #     self.dfs(self.data["jobs"], 0, capacity, tasks, 0)
        for _ in self.nearest_10_jobs[:7]:
            self.dfs(self.nearest_10_jobs[:7], 0, capacity, tasks, 0)

    def dfs(self, jobs, index, capacity, tasks, value):
        current_job = tasks[-1]
        value += current_job["value"]
        self.evaluate(current_job, tasks, value)
        # print("RRRRRRRRR", self.max_reward, index)
        if len(tasks) > capacity or index >= len(jobs):
            return
        # GOHOME
        if self.best_path is not None and self.step_count + len(self.best_path) >= 199:
            return
        for i in range(index, len(jobs)):
            self.swap(jobs, index, i)
            tasks.append(jobs[index])
            self.dfs(jobs, index+1, capacity, tasks, value)
            tasks.pop()
            self.swap(jobs, index, i)

    def evaluate(self, job, tasks, value):
        # If at home, no shortest path
        if self.data[self.us]["x"] != self.data[self.us]["home_x"] or self.data[self.us]["y"] != self.data[self.us]["home_y"]:
            return_path = self.shortest_path((self.data[self.us]["x"], self.data[self.us]["y"], self.data[self.us]["home_x"], self.data[self.us]["home_y"]))
        else:
            return_path = []

        current_path = self.get_action_path(tasks)
        steps = len(current_path) + len(return_path)
        # calculate rewards
        reward = 0
        if steps > 0:
            reward = value / steps
        # update
        if reward > self.max_reward:
            self.max_reward = reward
            # final_path = current_path + return_path
            self.best_path = current_path + return_path

    @staticmethod
    def swap(array, left, right):
        tmp = array[left]
        array[left] = array[right]
        array[right] = tmp

    def shortest_path(self, tup):
        return self.path_dict[tup]

    def get_action_path(self, tasks):
        # print(tasks)
        ret = []
        # tasks = list(set(tasks))
        for i in range(len(tasks)-1):
            ret += self.shortest_path((tasks[i]["x"], tasks[i]["y"], tasks[i+1]["x"], tasks[i+1]["y"]))
        return ret


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
