# -*- coding: utf-8 -*-
# @author: Yue Peng
# @email: yuepaang@gmail.com
# @create by: 2018-12-10 09:59:57
import random
import json
import time
from env import Player, Env
from a_star import Array2D, AStar, Point

# fake_data = {'player1': {'name': 'p1', 'x': 10, 'y': 6, 'home_x': 5, 'home_y': 5, 'n_jobs': 1, 'value': 10.0, 'score': 0}, 'player2': {'name': 'p2', 'x': 2, 'y': 5, 'home_x': 6, 'home_y': 6, 'n_jobs': 0, 'value': 0, 'score': 0}, 'walls': [{'x': 2, 'y': 10}, {'x': 8, 'y': 1}, {'x': 9, 'y': 11}], 'jobs': [{'x': 2, 'y': 4, 'value': 8.0}, {'x': 3, 'y': 9, 'value': 10.0}, {'x': 5, 'y': 0, 'value': 8.0}, {'x': 5, 'y': 4, 'value': 10.0}]}


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


class Bot(object):
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
        # TODO:
        self.non_pickup = 0
        # Stop Count
        self.stop_flag = 0
        # history cache
        self.value_cache = None
        self.x_cache = []
        self.y_cache = []
        self.enemy_threshold = 12 * 24 / 4
        # Memory
        self.memory_actions = []
        # A-star map
        self.map2d = None
        self.to_go_actions = []

        self.find_enemy_actions = []

    def find_enemy(self):
        self.find_path(self.us["x"], self.us["y"], self.enemy["x"], self.enemy["y"])
        self.find_enemy_actions = self.to_go_actions
        self.to_go_actions = []

    def global_scan(self):
        """Score for First direction 
        
        Returns:
            str -- action
        """
        ori_x, ori_y = self.us["x"], self.us["y"]
        direction_scores = {"U": 0, "L": 0, "D": 0, "R": 0}
        feasible_actions = self.where_to_go(self.us["x"], self.us["y"])
        direction_scores = {a:0 for a in feasible_actions}
        # Count total jobs for feasible actions:
        for k in direction_scores:
            for (xi, yi, vi) in self.targets:
                if k == "U":
                    if xi < ori_x:
                        direction_scores[k] += vi
                elif k == "D":
                    if xi > ori_x:
                        direction_scores[k] += vi
                elif k == "R":
                    if yi > ori_y:
                        direction_scores[k] += vi
                elif k == "L":
                    if yi > ori_y:
                        direction_scores[k] += vi

        # # most jobs
        # for (xi, yi, vi) in self.targets:
        #     # x-axis
        #     if xi > ori_x:
        #         direction_scores["D"] += vi
        #     else:
        #         direction_scores["U"] += vi
        #     # y-axis
        #     if yi > ori_y:
        #         direction_scores["R"] += vi
        #     else:
        #         direction_scores["L"] += vi

        # # Avoid the enemy
        # if self.enemy["x"] > ori_x:
        #     direction_scores["D"] -= self.enemy_threshold
        # else:
        #     direction_scores["U"] -= self.enemy_threshold
        # if self.enemy["y"] > ori_y:
        #     direction_scores["R"] -= self.enemy_threshold
        # else:
        #     direction_scores["L"] -= self.enemy_threshold
        max_score = max([v for _, v in direction_scores.items()])
        return [k for k, v in direction_scores.items() if v == max_score][0]

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
                # print("cache:", self.x_cache, self.y_cache)
                # print(self.us["x"], self.us["y"])
            else:
                self.stop_flag = 0
    
    # def stuck_in_region(self):
    #     if self.step_count > 4:
    #         # if self._distance(self.us["x"], self.us["y"], self.x_cache[-3], self.y_cache[-3]) < 3:
    #         if self.us["x"] in self.x_cache[-7:-5] and self.us["y"] in self.x_cache[-7:-5]:
    #             return True
    #         elif self.memory_actions[-4:-2] == self.memory_actions[-2:]:
    #             if self.memory_actions[-1] in self.feasible_actions:
    #                 self.feasible_actions.remove(self.memory_actions[-2])
    #             return True
    #         else:
    #             return False

    # def out_of_stuck(self):
    #     for (xi, yi, _) in self.targets:
    #         for d in [5, 6, 7, 8, 9, 10, 11, 12]:
    #             if self._distance(self.us["x"], self.us["y"], xi, yi) == d:
    #                 return (xi, yi)

    def _which_player(self, data):
        if data["player1"]["name"] == self.team_name:
            # print("I am Player 1!")
            self.us = data["player1"]
            self.enemy = data["player2"]
        elif data["player2"]["name"] == self.team_name:
            # print("I am Player 2!")
            self.us = data["player2"]
            self.enemy = data["player1"]

    def update_state(self, data):
        self.step_count += 1
        # load us and enemy
        self._which_player(data)
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
        self.memory_actions.append(self.previous_action())

        self.obstacles = [(d["x"], d["y"]) for d in data["walls"]]
        self.targets = [(d["x"], d["y"], d["value"]) for d in data["jobs"]]
        # # Add home as job
        # self.targets += [(self.us["home_x"], self.us["home_y"], 10)]
        # A-star
        # initialize the map when first obtain state
        if self.step_count == 1:
            self.map2d = Array2D(12, 12)
            for xi, yi in self.obstacles:
                self.map2d[yi][xi] = 1
            self.map2d[self.enemy["home_y"]][self.enemy["home_x"]] = 1

    def find_path(self, x_start, y_start, x_end, y_end):
        """Set up to_go_actions
        
        Arguments:
            x_start {[type]} -- [description]
            y_start {[type]} -- [description]
            x_end {[type]} -- [description]
            y_end {[type]} -- [description]
        """
        astar = AStar(self.map2d, Point(x_start, y_start), Point(x_end, y_end))
        if astar.start() is not None:
            path_list = [(p.x, p.y) for p in astar.start()]
            for i, (xi, yi) in enumerate(path_list):
                if i == 0:
                    if xi == x_start:
                        if yi > y_start:
                            self.to_go_actions.append("D")
                        else:
                            self.to_go_actions.append("U")
                    elif yi == y_start:
                        if xi > x_start:
                            self.to_go_actions.append("R")
                        else:
                            self.to_go_actions.append("L")
                else:
                    # Previous location
                    x_pre, y_pre = path_list[i-1]
                    if xi == x_pre:
                        if yi > y_pre:
                            self.to_go_actions.append("D")
                        else:
                            self.to_go_actions.append("U")
                    elif yi == y_pre:
                        if xi > x_pre:
                            self.to_go_actions.append("R")
                        else:
                            self.to_go_actions.append("L")

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

        # Enemy location detection
        # if self.enemy["x"] in surround_x and self.enemy["y"] == ori_y:
        #     if self.enemy["x"] < ori_x and "U" in self.feasible_actions:
        #         self.feasible_actions.remove("U")
        #     elif self.enemy["x"] > ori_x and "D" in self.feasible_actions:
        #         self.feasible_actions.remove("D")
        # elif self.enemy["x"] == ori_x and self.enemy["y"] in surround_y:
        #     if self.enemy["y"] < ori_y and "L" in self.feasible_actions:
        #         self.feasible_actions.remove("L")
        #     elif self.enemy["y"] > ori_y and "R" in self.feasible_actions:
        #         self.feasible_actions.remove("R")
        
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

    def bot_action(self):
        # TODO:
        """Main Step Function for GAME API
        
        Returns:
            str -- [description]
        """
        # feasible actions
        if self.us["x"] == self.us["home_x"] and self.us["y"] == self.us["home_y"]:
            # return self.global_scan()
            self.find_enemy()
            if self._distance(self.us["x"], self.us["y"], self.enemy["x"], self.enemy["y"]) > 8:
                return self.find_enemy_actions.pop(0)
            else:
                return self.global_scan()

        # Must go home
        if self.stop_flag > 1:
            # Clear all the actions
            self.to_go_actions = []
            self.find_path(self.us["y"], self.us["x"], self.us["home_y"], self.us["home_x"])
            return self.to_go_actions.pop(0)
        if self.step_count + self.go_home() == 199 or self.job_count > 9:
            # different axis from a-star
            self.to_go_actions = []
            self.find_path(self.us["y"], self.us["x"], self.us["home_y"], self.us["home_x"])
        # Step 1: first scan
        if self.step_count == 1:
            return self.global_scan()

        if self.non_pickup > 10 and self.job_count > 1:
            self.to_go_actions = []
            self.find_path(self.us["y"], self.us["x"], self.us["home_y"], self.us["home_x"])
            self.to_go_actions.append(self.global_scan())


        if len(self.to_go_actions) != 0:
            if self._distance(self.us["x"], self.us["y"], self.enemy["x"], self.enemy["y"]) < 8:
                self.find_enemy()
                return self.find_enemy_actions.pop(0)
            return self.to_go_actions.pop(0)
        else:
            x_bar, y_bar = self.find_nearest_target()
            self.find_path(self.us["y"], self.us["x"], y_bar, x_bar)
            return self.to_go_actions.pop(0)

    def find_nearest_target(self):
        """Return the location of nearest job
        
        Returns:
            (int, int) -- [description]
        """
        assert self.targets is not None, "You must update the state first!"
        min_dist = 10000
        res = ()
        for (xi, yi, _) in self.targets:
            if xi != 0 or yi != 0 or xi != 11 or yi != 11:
                if self._distance(xi, yi, self.us["x"], self.us["y"]) < min_dist:
                    min_dist = self._distance(xi, yi, self.us["x"], self.us["y"])
                    res = (xi, yi)
        return res

    def find_gain_most(self):
        target_gains = {}
        for (xi, yi, vi) in self.targets:
            target_gains[(xi, yi)] = vi / self._distance(xi, yi, self.us["x"], self.us["y"])
        max_gain = max([v for _, v in target_gains.items()])
        return [k for k, v in target_gains.items() if v == max_gain][0]

    def _distance(self, x1, y1, x2, y2):
        # distance = 0
        # moving_x, moving_y = x1, y1
        # while moving_x != x2 and moving_y != y2:
        #     feasible_action = self.where_to_go(moving_x, moving_y)
        #     moving_x, moving_y = self.action_update(moving_x, moving_y, feasible_action.pop(0))
        #     distance += 1
        # return distance

        return abs(x1 - x2) + abs(y1 - y2)

    def take_actions(self, x, y):
        """Action set to reach the input location (x, y)

        Arguments:
            x {int} -- [description]
            y {int} -- [description]
        
        Returns:
            [list<str>] -- [description]
        """
        to_action = True
        action_set = []
        if to_action:
            moving_x, moving_y = self.us["x"], self.us["y"]
            while to_action:
                # x-axis
                if moving_x - x < 0:
                    action_set.append("D")
                    moving_x += 1
                elif moving_x - x > 0:
                    action_set.append("U")
                    moving_x -= 1
                # y-axis
                if moving_y - y < 0:
                    action_set.append("R")
                    moving_y += 1
                elif moving_y - y > 0:
                    action_set.append("L")
                    moving_y -= 1
                # break condition
                if moving_x == x and moving_y == y:
                    break
        return action_set

    def go_home(self):
        # TODO:
        """Actions to go home
        
        Returns:
            [list] -- [description]
        """
        # final_decision = []
        # ori_x, ori_y = self.us["x"], self.us["y"]
        # while ori_x != self.us["home_x"] and ori_y != self.us["home_y"]:
        #     feasible_actions = self.where_to_go(ori_x, ori_y)
        #     action_set = self.take_actions(self.us["home_x"], self.us["home_y"])
        #     for a in action_set:
        #         if a in feasible_actions:
        #             ori_x, ori_y = self.action_update(ori_x, ori_y, a)
        #             final_decision.append(a)
        # return final_decision 
        return self._distance(self.us["x"], self.us["y"], self.us["home_x"], self.us["home_y"])

    @staticmethod
    def action_update(x, y, action):
        if action == "D":
            x += 1
        elif action == "U":
            x -= 1
        elif action == "L":
            y -= 1
        elif action == "R":
            y += 1
        return x, y

    def previous_action(self):
        if self.step_count > 2:
            if self.us["x"] == self.x_cache[-2]:
                if self.us["y"] > self.y_cache[-2]:
                    return "R"
                elif self.us["y"] < self.y_cache[-2]:
                    return "L"
                else:
                    return "S"
            elif self.us["y"] == self.y_cache[-2]:
                if self.us["x"] > self.x_cache[-2]:
                    return "D"
                elif self.us["x"] < self.x_cache[-2]:
                    return "U"
                else:
                    return "S"


def main():
    bot = Bot(team_name="p1")
    # Env init
    env = Env("", "p1", "p2", CONF, random.Random(20))
    init_state = env.reset()
    # print(init_state)
    env.render()
    bot.load_state(init_state)
    # greedy search
    # x, y = bot.find_nearest_target()
    # p1_action = bot.take_actions(x, y)
    while True:
        try:
            # state = env.step("p1", p1_action.pop())
            aa = bot.bot_action()
            state = env.step("p1", aa)
            print("Action:", aa)
            print("TO GO:", bot.to_go_actions)
        except:
            print("========================")
            feasible_actions = bot.where_to_go(bot.us["x"], bot.us["y"])
            aa = random.choice(feasible_actions)
            print("Action:", aa)
            state = env.step("p1", aa)
        env.render()
        bot.load_state(state)
        print("Stop:", bot.stop_flag)
        print("N_JOBS:", bot.job_count)

        print("Step:", bot.step_count)
        # if bot.us["x"] == bot.us["home_x"] and bot.us["y"] == bot.us["home_y"]:
        #     print("Score:", bot.us["score"])
        print("Values:", bot.us["value"])
        # Player 2
        env.step("p2", random.choice(["U", "R", "D", "L"]))
        if env.done:
            print("**********Final Score:", bot.us["score"])
            break
        time.sleep(0.3)


if __name__ == "__main__":
    main()
