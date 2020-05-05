# -*- coding: utf-8 -*-
# @author: Yue Peng
# @email: yuepaang@gmail.com
# @create by: 2018-12-12 19:33:14
import random
import time
# from collections import OrderedDict
from env import Player, Env
from a_star import Array2D, AStar, Point
# from bfs import Finder


CONF = {
    'world_size': 12,
    'capacity': 10,
    'player1_home': (5, 5),
    'player2_home': (6, 6),
    'num_walls': 24,
    'num_jobs': 24,
    'value_range': (6, 12),
    'max_steps': 200
    }


# class Node(object):
#     def __init__(self, job=None):
#         self.job = job
#         self.childs = []

#     def add_child(self, node):
#         self.childs.append(node)


# def init_tree(targets, home_x, home_y):
#     """生成多叉树
    
#     Arguments:
#         targets {[type]} -- [description]
#         home_x {[type]} -- [description]
#         home_y {[type]} -- [description]
    
#     Returns:
#         [type] -- [description]
#     """

#     root = Node((5, 6, 0.))
#     for target in targets:
#         root.add_child(Node(target))

#     for n in root.childs:
#         for target in targets:
#             if target == n.job:
#                 continue
#             n.add_child(Node(target))
#         n.add_child(Node((home_x, home_y, 0.)))

#     for n in root.childs:
#         for nn in n.childs:
#             for target in targets:
#                 if target == nn.job or target == n.job:
#                     continue
#                 nn.add_child(Node(target))
#             nn.add_child(Node((home_x, home_y, 0.)))

#     for n in root.childs:
#         for nn in n.childs:
#             for nnn in nn.childs:
#                 for target in targets:
#                     if target == nn.job or target == n.job or target == nnn.job:
#                         continue
#                     nnn.add_child(Node(target))
#                 nnn.add_child(Node((home_x, home_y, 0.)))
#     return root


# result = 0
# path = []
# max_sub = 0

# def calculate_sum(subroot):
#     global result
#     global path
#     global max_sub
#     if len(subroot.childs) == 0:
#         return subroot.job[2]
#     # Child node
#     sub_path_sum = [0] * len(subroot.childs)
#     for i, n in enumerate(subroot.childs):
#         sub_path_sum[i] = calculate_sum(n)
#     # path_sum = [s + subroot.job[2] for s in sub_path_sum]
#     if max(sub_path_sum) > max_sub:
#         # path.append((subroot.job[0], subroot.job[1]))
#         path.append((subroot.childs[sub_path_sum.index(max(sub_path_sum))].job[0], subroot.childs[sub_path_sum.index(max(sub_path_sum))].job[1]))
#         max_sub = max(sub_path_sum)
#     # Write down best score
#     result = max(sum(sub_path_sum) + subroot.job[2], result)
#     return max(0, max(sub_path_sum) + subroot.job[2])


# def dfs(node, target, d=10, visited=None, res=None):
#     if d == 0:
#         return
#     if visited is None:
#         visited = set()
#     if res is None:
#         res = []
#     res.append(target)
#     visited.add(target)
#     for u in graph[target]:
#         if u in visited:
#             continue
#         visited.add(u)
#         dfs(graph, u, d-1, visited, res)
#     return res


# def init_graph(targets):
#     pass


# def dfs(G, s):
#     yielded = set()

#     def recurse(G, s, d, S=None):
#         if s not in yielded:
#             yield s
#             yielded.add(s)
#         if d == 0:
#             return
#         if S is None:
#             S = set()
#         S.add(s)

#         for u in G[s]:
#             if u in S:
#                 continue
#             for v in recurse(G, s, d-1, S):
#                 yield v
    
#     n = len(G)
#     for d in range(n):
#         if len(yielded) == n:
#             break
#         for u in recurse(G, s, d):
#             yield u


class DFS(object):
    def __init__(self, team_name):
        self.team_name = team_name
        self.us = None
        self.enemy = None
        self.targets = None
        self.obstacles = None
        # Step and Job count
        self.step_count = 0
        self.job_count = 0
        # Non-pickup count
        self.non_pickup = 0
        # Stop Count
        self.stop_flag = 0
        # history cache
        self.value_cache = None
        self.x_cache = []
        self.y_cache = []
        # Memory
        self.memory_actions = []
        # A-star map
        self.map2d = None
        # self.path_dict = OrderedDict()
        self.path_dict = {}
        self.to_go_actions = []

        # Finder
        self.home = None
        self.best_path = None
        self.max_reward = 0

    def _init_map(self):
        """让你八秒
        """
        map2d = Array2D(12, 12)
        # init obstacles
        for xi, yi in self.obstacles:
            map2d[yi][xi] = 1
        # init enemy home
        map2d[self.enemy["home_y"]][self.enemy["home_x"]] = 1
        # Generate all location
        for x in range(12):
            for y in range(12):
                for m in range(12):
                    for n in range(12):
                        if m == x and n == y:
                            continue
                        elif map2d[m][n] == 1 or map2d[x][y] == 1:
                            continue
                        else:
                            astar = AStar(map2d, Point(x, y), Point(m, n))
                            path_list = [(p.y, p.x) for p in astar.start()]
                            self.path_dict[(y, x, n, m)] = path_list

    def _which_player(self, data):
        if data["player1"]["name"] == self.team_name:
            # print("I am Player 1!")
            self.us = data["player1"]
            self.enemy = data["player2"]
        elif data["player2"]["name"] == self.team_name:
            # print("I am Player 2!")
            self.us = data["player2"]
            self.enemy = data["player1"]

    def _home_or_not(self):
        """When reaching home, job count clear to zero
        """
        if self.us["x"] == self.us["home_x"] and self.us["y"] == self.us["home_y"]:
            self.job_count = 0

    def _job_change(self):
        if self.us["value"] != self.value_cache:
            self.job_count += 1
            self.non_pickup = 0
        else:
            self.non_pickup += 1

    def stop_or_not(self):
        if self.step_count > 1:
            if self.us["x"] == self.x_cache[-1] and self.us["y"] == self.y_cache[-1]:
                self.stop_flag += 1
            else:
                self.stop_flag = 0

    def update_state(self, data):
        self.best_path = None
        self.max_reward = 0

        self.step_count += 1
        # load us and enemy
        self._which_player(data)
        if self.home is None:
            self.home = {
                "x": self.us["home_x"],
                "y": self.us["home_y"]
            }
        # Stop?
        self.stop_or_not()
        # Home?
        self._home_or_not()
        # Pick up?
        self._job_change()

        # update cache
        self.x_cache.append(self.us["x"])
        self.y_cache.append(self.us["y"])
        self.value_cache = self.us["value"]

        self.obstacles = [(d["x"], d["y"]) for d in data["walls"]]
        self.targets = [(d["x"], d["y"], d["value"]) for d in data["jobs"]]

        if len(self.path_dict) == 0:
            self._init_map() # 让你八秒
            print(len(self.path_dict))
        # print(list(self.path_dict.keys())[:10])

    def where_to_go(self, x, y):
        """Find out the direction bot can go
        """
        feasible_actions = ["U", "D", "R", "L"]
        ori_x = x
        ori_y = y
        surround_x = [ori_x - 1, ori_x + 1]
        surround_y = [ori_y - 1, ori_y + 1]
        # detect if up, down, left, right can reach
        for (m, n) in self.obstacles:
            if m in surround_x and n == ori_y:
                if m < ori_x and "U" in feasible_actions:
                    feasible_actions.remove("U")
                elif m > ori_x and "D" in feasible_actions:
                    feasible_actions.remove("D")
            elif m == ori_x and n in surround_y:
                if n < ori_y and "L" in feasible_actions:
                    feasible_actions.remove("L")
                elif n > ori_y and "R" in feasible_actions:
                    feasible_actions.remove("R")
        
        #Enemy home detection
        if self.enemy["home_x"] in surround_x and self.enemy["home_y"] == ori_y:
            if self.enemy["home_x"] < ori_x and "U" in feasible_actions:
                feasible_actions.remove("U")
            elif self.enemy["home_x"] > ori_x and "D" in feasible_actions:
                feasible_actions.remove("D")
        elif self.enemy["home_x"] == ori_x and self.enemy["home_y"] in surround_y:
            if self.enemy["home_y"] < ori_y and "L" in feasible_actions:
                feasible_actions.remove("L")
            elif self.enemy["home_y"] > ori_y and "R" in feasible_actions:
                feasible_actions.remove("R")

        # Edge detection
        if ori_x == 0 and "U" in feasible_actions:
            feasible_actions.remove("U")
        if ori_x == 11 and "D" in feasible_actions:
            feasible_actions.remove("D")
        if ori_y == 0 and "L" in feasible_actions:
            feasible_actions.remove("L")
        if ori_y == 11 and "R" in feasible_actions:
            feasible_actions.remove("R")
        if len(feasible_actions) == 0:
            return "S"
        return feasible_actions

    def find_path(self, x_start, y_start, x_end, y_end):
        """Set up to_go_actions
        
        Arguments:
            x_start {[type]} -- [description]
            y_start {[type]} -- [description]
            x_end {[type]} -- [description]
            y_end {[type]} -- [description]
        """
        to_go_actions = []
        # FIXME:
        points_list = self.path_dict[(x_start, y_start, x_end, y_end)]
        for i, (xi, yi) in enumerate(points_list):
            if len(to_go_actions) == 1:
                break
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

    # TODO:
    # def dfs_search(self):
    #     Finder.home = {
    #         "x": self.us["home_x"],
    #         "y": self.us["home_y"]
    #     }
    #     current_location = {
    #         "x": self.us["x"],
    #         "y": self.us["y"]
    #     }
    #     jobs = [0] * len(self.targets)
    #     for i, (xi, yi, vi) in self.targets:
    #         jobs[i] = {
    #             "x": xi,
    #             "y": yi,
    #             "v": vi
    #         }
    #     Finder.best_route(current_location, jobs, 6)
    #     Finder.get_action_path

    def bot_action(self):
        if 10 - self.job_count > 3:
            self.best_route(3)
        else:
            self.best_route(10-self.job_count)
        if self.best_path is None:
            return self.find_path(self.us["x"], self.us["y"], self.us["home_x"], self.us["home_y"]).pop(0)
        first_point = self.best_path.pop(0)
        return self.find_path(self.us["x"], self.us["y"], first_point[0], first_point[1]).pop(0)

    def best_route(self, capacity):
        tasks = []
        self.best_path = None
        self.max_reward = 0
        # FIXME:
        dummy_job = (self.us['x'], self.us['y'], 0.)
        tasks.append(dummy_job)
        for _ in self.targets:
            # tasks.append(job)
            self.dfs(self.targets, 0, capacity, tasks, 0)
    
    def dfs(self, jobs, index, capacity, tasks, value):
        current_job = tasks[-1]
        value += current_job[-1]
        self.evaluate(current_job, tasks, value)
        # print("RRRRRRRRR", self.max_reward, index)
        if len(tasks) > capacity or index >= len(jobs):
            return
        for i in range(index, len(jobs)):
            self.swap(jobs, index, i)
            tasks.append(jobs[index])
            self.dfs(jobs, index+1, capacity, tasks, value)
            tasks.pop()
            self.swap(jobs, index, i)
    
    def evaluate(self, job, tasks, value):
        # If at home, no shortest path
        if self.us["x"] != self.us["home_x"] or self.us["y"] != self.us["home_y"]:
            return_path = self.shortest_path((self.us["x"], self.us["y"], self.us["home_x"], self.us["home_y"]))
        else:
            return_path = []

        current_path = self.get_action_path(tasks)
        # FIXME:
        steps = len(current_path) + len(return_path)
        # calculate rewards
        reward = 0
        if steps > 0:
            reward = value / steps
        # update
        if reward > self.max_reward:
            self.max_reward = reward
            # final_path = current_path + return_path
            self.best_path = current_path[:2]

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
            ret += self.shortest_path((tasks[i][0], tasks[i][1], tasks[i+1][0], tasks[i+1][1]))
        return ret


def main():
    bot = DFS("p1")
    # bot2 = DFS("p2")
    env = Env("", "p1", "p2", CONF, random.Random(20))
    state = env.reset()
    # bot2.load_state(state)
    env.render()
    while True:
        # p1 action
        bot.load_state(state)
        action = bot.bot_action()
        state = env.step("p1", action)

        env.render()
        print("Stop:", bot.stop_flag)
        print("Action:", action)
        print("N_JOBS:", bot.job_count)
        print("Step:", bot.step_count)
        print("Values:", bot.us["value"])
        print("Scores:", bot.us["score"])
        
        # Player 2
        _ = env.step("p2", "S")
        # state2 = env.step("p2", bot2.bot_action())
        # bot2.load_state(state2)
        if env.done:
            print("**********P1 Final Score:", bot.us["score"])
            # print("**********P2 Final Score:", bot2.us["score"])
            break
        time.sleep(0.3)


# def a_star():
#     test_state = {'player1': {'name': 'p1', 'x': 5, 'y': 5, 'home_x': 5, 'home_y': 5, 'n_jobs': 0, 'value': 0, 'score': 0}, 'player2': {'name': 'p2', 'x': 6, 'y': 6, 'home_x': 6, 'home_y': 6, 'n_jobs': 0, 'value': 0, 'score': 0}, 'walls': [{'x': 0, 'y': 8}, {'x': 1, 'y': 1}, {'x': 1, 'y': 10}, {'x': 1, 'y': 11}, {'x': 3, 'y': 11}, {'x': 4, 'y': 0}, {'x': 4, 'y': 10}, {'x': 5, 'y': 0}, {'x': 5, 'y': 1}, {'x': 5, 'y': 4}, {'x': 5, 'y': 7}, {'x': 5, 'y': 9}, {'x': 6, 'y': 5}, {'x': 7, 'y': 0}, {'x': 7, 'y': 1}, {'x': 7, 'y': 8}, {'x': 8, 'y': 3}, {'x': 8, 'y': 4}, {'x': 9, 'y': 1}, {'x': 9, 'y': 3}, {'x': 9, 'y': 9}, {'x': 9, 'y': 10}, {'x': 10, 'y': 3}, {'x': 10, 'y': 6}], 'jobs': [{'x': 0, 'y': 2, 'value': 8.0}, {'x': 0, 'y': 4, 'value': 11.0}, {'x': 1, 'y': 4, 'value': 7.0}, {'x': 1, 'y': 7, 'value': 11.0}, {'x': 1, 'y': 8, 'value': 9.0}, {'x': 2, 'y': 7, 'value': 11.0}, {'x': 3, 'y': 5, 'value': 10.0}, {'x': 3, 'y': 6, 'value': 11.0}, {'x': 4, 'y': 5, 'value': 10.0}, {'x': 4, 'y': 9, 'value': 6.0}, {'x': 5, 'y': 3, 'value': 7.0}, {'x': 5, 'y': 6, 'value': 6.0}, {'x': 5, 'y': 8, 'value': 10.0}, {'x': 6, 'y': 8, 'value': 6.0}, {'x': 6, 'y': 9, 'value': 7.0}, {'x': 6, 'y': 10, 'value': 8.0}, {'x': 6, 'y': 11, 'value': 8.0}, {'x': 7, 'y': 6, 'value': 11.0}, {'x': 8, 'y': 1, 'value': 9.0}, {'x': 9, 'y': 5, 'value': 8.0}, {'x': 9, 'y': 11, 'value': 12.0}, {'x': 10, 'y': 5, 'value': 7.0}, {'x': 10, 'y': 9, 'value': 10.0}, {'x': 11, 'y': 3, 'value': 7.0}]}
#     bot = DFS("p1")
#     bot.load_state(test_state)
#     map2d = Array2D(12, 12)
#     # init obstacles
#     for xi, yi in bot.obstacles:
#         map2d[yi][xi] = 1
#     # init enemy home
#     map2d[bot.enemy["home_y"]][bot.enemy["home_x"]] = 1
#     # Generate all location
#     path_dict = OrderedDict()
#     for x in range(12):
#         for y in range(12):
#             for m in range(12):
#                 for n in range(12):
#                     if m == x and n == y:
#                         continue
#                     elif map2d[m][n] == 1 or map2d[x][y] == 1:
#                         continue
#                     else:
#                         astar = AStar(map2d, Point(x, y), Point(m, n))
#                         # Env location swap x&y
#                         path_list = [(p.y, p.x) for p in astar.start()]
#                         path_dict[(x, y, m, n)] = path_list
#     print(len(path_dict))
#     print(path_dict[(0,0,11,11)])

if __name__ == "__main__":
    main()
    # start = time.time()
    # # a_star()
    # bot = DFS("p1")
    # test_state = {'player1': {'name': 'p1', 'x': 5, 'y': 5, 'home_x': 5, 'home_y': 5, 'n_jobs': 0, 'value': 0, 'score': 0}, 'player2': {'name': 'p2', 'x': 6, 'y': 6, 'home_x': 6, 'home_y': 6, 'n_jobs': 0, 'value': 0, 'score': 0}, 'walls': [{'x': 0, 'y': 8}, {'x': 1, 'y': 1}, {'x': 1, 'y': 10}, {'x': 1, 'y': 11}, {'x': 3, 'y': 11}, {'x': 4, 'y': 0}, {'x': 4, 'y': 10}, {'x': 5, 'y': 0}, {'x': 5, 'y': 1}, {'x': 5, 'y': 4}, {'x': 5, 'y': 7}, {'x': 5, 'y': 9}, {'x': 6, 'y': 5}, {'x': 7, 'y': 0}, {'x': 7, 'y': 1}, {'x': 7, 'y': 8}, {'x': 8, 'y': 3}, {'x': 8, 'y': 4}, {'x': 9, 'y': 1}, {'x': 9, 'y': 3}, {'x': 9, 'y': 9}, {'x': 9, 'y': 10}, {'x': 10, 'y': 3}, {'x': 10, 'y': 6}], 'jobs': [{'x': 0, 'y': 2, 'value': 8.0}, {'x': 0, 'y': 4, 'value': 11.0}, {'x': 1, 'y': 4, 'value': 7.0}, {'x': 1, 'y': 7, 'value': 11.0}, {'x': 1, 'y': 8, 'value': 9.0}, {'x': 2, 'y': 7, 'value': 11.0}, {'x': 3, 'y': 5, 'value': 10.0}, {'x': 3, 'y': 6, 'value': 11.0}, {'x': 4, 'y': 5, 'value': 10.0}, {'x': 4, 'y': 9, 'value': 6.0}, {'x': 5, 'y': 3, 'value': 7.0}, {'x': 5, 'y': 6, 'value': 6.0}, {'x': 5, 'y': 8, 'value': 10.0}, {'x': 6, 'y': 8, 'value': 6.0}, {'x': 6, 'y': 9, 'value': 7.0}, {'x': 6, 'y': 10, 'value': 8.0}, {'x': 6, 'y': 11, 'value': 8.0}, {'x': 7, 'y': 6, 'value': 11.0}, {'x': 8, 'y': 1, 'value': 9.0}, {'x': 9, 'y': 5, 'value': 8.0}, {'x': 9, 'y': 11, 'value': 12.0}, {'x': 10, 'y': 5, 'value': 7.0}, {'x': 10, 'y': 9, 'value': 10.0}, {'x': 11, 'y': 3, 'value': 7.0}]}
    # bot.load_state(test_state)
    # print(bot.path_dict[(0,0,11,11)])
    # print(len(bot.path_dict))
    # end = time.time()
    # print("Time:", end - start)
