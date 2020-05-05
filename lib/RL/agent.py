import numpy as np
from utils.a_star import AStar, Array2D, Point
from utils.greedy_rule import GreedyBenchmark
import random
import copy

feasible_actions = {"U": 0, "D": 1, "L": 2, "R": 3, "S": 4}
action_symbols = {0: "U", 1: "D", 2: "L", 3: "R", 4: "S"}

class Agent(object):
    def __init__(self, max_capacity, receptive_field_size, model, name, max_steps):
        self.max_capacity = float(max_capacity)
        self.receptive_field_size = receptive_field_size
        # self.last_score_matrix = None
        self.model = model
        self.last_m2_me = 0
        self.last_m1_me = 0
        self.cur_m2_me = 0
        self.cur_m1_me = 0
        self.name = name
        self.steps_left = max_steps
        return

    def rough_plot_observe(self, tensor, plot_func):
        te = np.array(tensor)

        def padding(x):
            if isinstance(x, float):
                x = int(x)
            if isinstance(x, int):
                if x < 0 and x>-10:
                    _str = "-"+str(-x)
                elif x <= -10:
                    _str = "M"+(str(-x)[-1])
                elif x >= 0 and x < 10:
                    _str = " "+str(x)
                elif x >= 10:
                    _str = str(x)[:2]
                else:
                    _str = str(x)[:2]
            else:
                _str = str(x)[:2]
            return _str


        line_buffer = []
        for i in range(te.shape[0]):
            col_buffer = []
            for j in range(te.shape[0]):
                if i == (te.shape[0] - 1) // 2 and j == (te.shape[0] - 1) // 2:
                    col_buffer.append("$$")
                else:
                    obj = te[i][j]
                    char = plot_func(obj)
                    col_buffer.append(padding(char))
            line = "|"+ "|".join(col_buffer) + "|"
            line_buffer.append(line)

        bar = "".join(["-" for _ in range(len(line_buffer[0]))])
        result_str = bar + "\n" + "\n".join(line_buffer) + "\n"+bar
        print(result_str)
        return


    def rough_plot_board(self, board_state, field_size=12):
        board = np.zeros([field_size, field_size])
        for item in board_state['walls']:
            board[item['x']][item['y']] = -1.0
        for item in board_state['jobs']:
            board[item['x']][item['y']] = item['value']

        board[board_state['player1']['x']][board_state['player1']['y']] = 100.0
        board[board_state['player2']['x']][board_state['player2']['y']] = 200.0
        if board_state['player1']['x'] == board_state['player2']['x'] and board_state['player1']['y'] == board_state['player2']['y']:
            board[board_state['player2']['x']][board_state['player2']['y']] = 300.0

        line_buffer = []
        for i in range(field_size):
            col_buffer = []
            for j in range(field_size):
                if board[i][j] == 300.0:
                    symbol = "$$"
                elif board[i][j] == 100.0:
                    symbol = "&1"
                elif board[i][j] == 200.0:
                    symbol = "&2"
                elif board[i][j] < 0:
                    symbol = "##"
                elif board[i][j] > 0:
                    symbol = " "+str(int(board[i][j])) if board[i][j] < 10 else str(int(board[i][j]))
                else:
                    symbol = "  "
                col_buffer.append(symbol)
            line = "|"+ "|".join(col_buffer) + "|"
            line_buffer.append(line)

        bar = "".join(["-" for _ in range(len(line_buffer[0]))])
        result_str = bar + "\n" + "\n".join(line_buffer) + "\n"+bar
        print(result_str)
        return


    def observe_state(self, jobs, walls, me, rival, field_size):
        max_peek_range = self.receptive_field_size

        jobs_dict = dict([(_x["x"] * field_size + _x["y"], _x["value"]) for _x in jobs])
        walls_dict = dict([(_x["x"] * field_size + _x["y"], True) for _x in walls])

        obstacle_channel = np.zeros([2*max_peek_range + 1, 2*max_peek_range + 1])
        package_channel = np.zeros([2*max_peek_range + 1, 2*max_peek_range + 1])
        rival_channel = np.zeros([2*max_peek_range + 1, 2*max_peek_range + 1])
        home_channel = np.zeros([2*max_peek_range + 1, 2*max_peek_range + 1])
        distance_advantage_channel = np.ones([2*max_peek_range + 1, 2*max_peek_range + 1]) * - (2.0 * max_peek_range)

        # -------------------
        map2d_me = Array2D(field_size, field_size)
        map2d_rival = Array2D(field_size, field_size)
        for i in range(field_size):
            for j in range(field_size):
                if i * field_size + j in walls_dict:
                    map2d_me[i][j] = 1
                    map2d_rival[i][j] = 1
                if (i, j) == (rival["home_x"], rival["home_y"]):
                    map2d_me[i][j] = 1
                if (i, j) == (me["home_x"], me["home_y"]):
                    map2d_rival[i][j] = 1


        def distance_advantage(ptx, pty, pkg_score=0):
            if pkg_score > 0:
                self_astar=AStar(map2d_me, Point(ptx, pty), Point(me["x"], me["y"]))
                self_path_list = self_astar.start()
                if self_path_list is None:
                    self_steps = 20
                else:
                    self_steps = len(self_path_list)

                rival_astar = AStar(map2d_rival, Point(ptx, pty), Point(rival["x"], rival["y"]))
                rival_path_list = rival_astar.start()
                if rival_path_list is None:
                    rival_steps = 20
                else:
                    rival_steps = len(rival_path_list)
                advantage = rival_steps - self_steps if self_path_list is not None else -20
            else:
                self_steps = abs(me["x"] - ptx) + abs(me["y"] - pty)
                rival_steps = abs(rival["x"] - ptx) + abs(rival["y"] - pty)
                advantage = rival_steps - self_steps
            return advantage
        # ----------------------

        inventory_me = me["n_jobs"] / self.max_capacity
        inventory_rival = rival["n_jobs"] / self.max_capacity
        me_full = 0.0 if me["n_jobs"] < self.max_capacity else 1.0
        rival_full = 0.0 if rival["n_jobs"] < self.max_capacity else 1.0

        if self.steps_left <= 20 or me["n_jobs"] >= self.max_capacity - 1:
            home_astar=AStar(map2d_me, Point(me["x"], me["y"]), Point(me["home_x"], me["home_y"]))
            self_home_path = home_astar.start()
            if self_home_path is None:
                home_distance = 0
                if (me["x"], me["y"]) == (me["home_x"], me["home_y"]):
                    home_move = (me["home_x"], me["home_y"])
                else:
                    aaaa = 1
            else:
                home_distance = len(self_home_path)
                home_move = self_home_path[0].x, self_home_path[0].y
            dx = home_move[0] - me["x"]
            dy = home_move[1] - me["y"]
        else:
            dx, dy = 0, 0
            home_distance = 10.0

        move_vector = np.zeros([len(feasible_actions)])
        if dx == 1:
            move_vector[feasible_actions["D"]] = 1.0
        elif dx == -1:
            move_vector[feasible_actions["U"]] = 1.0
        elif dy == 1:
            move_vector[feasible_actions["R"]] = 1.0
        elif dy == -1:
            move_vector[feasible_actions["L"]] = 1.0
        else:
            move_vector[feasible_actions["S"]] = 1.0

        spare = 1
        # spare += int(home_distance // 5)

        force_home = min(self.steps_left - home_distance - spare, field_size)
        # --------------------

        def out_range(a):
            return a < 0 or a >= field_size

        for i in range(2*max_peek_range + 1):
            field_x = me["x"] - max_peek_range + i
            for j in range(2*max_peek_range + 1):
                field_y = me["y"] - max_peek_range + j

                # --- walls -----
                if out_range(field_x) or out_range(field_y):
                    obstacle_channel[i, j] = 1.0
                else:
                    if field_x * field_size + field_y in walls_dict or (field_x, field_y)==(rival["home_x"], rival["home_y"]):
                        obstacle_channel[i, j] = 1.0
                    else:
                        obstacle_channel[i, j] = 0.0

                    # --- jobs ------
                    if field_x * field_size + field_y in jobs_dict:
                        pkg_value = jobs_dict[field_x * field_size + field_y]
                        package_channel[i, j] = pkg_value
                    else:
                        pkg_value = 0.0


                    # --- competitor ----
                    if (rival["x"], rival["y"]) == (field_x, field_y):
                        rival_channel[i, j] = 1.0

                    # --- home -----
                    if (me["home_x"], me["home_y"]) == (field_x, field_y):
                        home_channel[i, j] = inventory_me if force_home > 0 else 1.0

                    # ----distance advantage ----
                    dist_advantage = distance_advantage(field_x, field_y, pkg_score=pkg_value)
                    distance_advantage_channel[i, j] = dist_advantage

        obstacle_channel -= 2.0 * home_channel

        result_tensor = np.concatenate([np.expand_dims(obstacle_channel, axis=2),
                                        np.expand_dims(package_channel, axis=2),
                                        np.expand_dims(rival_channel, axis=2),
                                        np.expand_dims(distance_advantage_channel, axis=2),
                                        np.expand_dims(home_channel, axis=2)
                                        ],
                                       axis=2)
        inventory_tensor = np.array([inventory_me, inventory_rival, me_full, rival_full, force_home])
        extra_tensor = np.concatenate([inventory_tensor, move_vector], axis=0)



        return (result_tensor[np.newaxis, :, :, :], extra_tensor[np.newaxis, :])


    def observe_state2(self, jobs, walls, me, rival, field_size):

        return np.zeros([1, 23, 23, 5]), np.zeros([1, 10])

    def move(self, movement, env):
        new_state = env.step(self.name, movement)
        self.steps_left -= 1
        if new_state['player1']['name'] == self.name:
            score, value = new_state['player1']["score"], new_state['player1']["value"]
        else:
            score, value = new_state['player2']["score"], new_state['player2']["value"]

        self.last_m2_me = self.cur_m2_me
        self.last_m1_me = self.cur_m1_me
        self.cur_m2_me = score + value
        self.cur_m1_me = score
        return new_state


    def policy_movement(self, state, epsilon):
        vision, extra = state
        action_Qs = self.model.predict(vision, extra)
        dqn_action_indices = np.argmax(action_Qs, axis=1)

        action_num = len(feasible_actions)

        random_probs = np.ones([1, action_num], dtype=float) * epsilon / float(action_num)

        action_tensor = []
        for bi in range(dqn_action_indices.shape[0]):
            random_probs[bi][dqn_action_indices[bi]] += (1.0 - epsilon)
            probs = random_probs[bi]
            action_index = np.random.choice(np.arange(action_num), p=probs)
            oh_action = np.zeros([action_num])
            oh_action[action_index] = 1.0
            action_tensor.append(oh_action)

        res = np.array(action_tensor)
        return res


    def greedy_stepwise_gain(self, rule_state):
        jobs, walls, me, rival, field_size = rule_state
        obstacles = Array2D(field_size, field_size)
        for item in walls:
            obstacles[item["x"]][item["y"]] = 1
        obstacles[rival["home_x"]][rival["home_y"]] = 1
        obstacles[rival["x"]][rival["y"]] = 1

        move_stats = {}

        x, y = me["x"], me["y"]
        for job in jobs:
            astar = AStar(copy.deepcopy(obstacles), Point(x, y), Point(job["x"], job["y"]))
            path_list = astar.start()
            if path_list is not None:
                pathlen = float(len(path_list))
                next_x, next_y = path_list[0].x, path_list[0].y
                delta_x, delta_y = next_x - x, next_y - y
                if delta_x == -1:
                    move = "U"
                elif delta_x == 1:
                    move = "D"
                elif delta_y == -1:
                    move = "L"
                elif delta_y == 1:
                    move = "R"
                else:
                    move = "S"
            else:
                pathlen = 100.0
                move = random.choice(["U", "D", "L", "R", "S"])
            gain = job["value"] / pathlen

            if move not in move_stats:
                move_stats[move] = []
            move_stats[move].append(gain)

        expectation_move_gain = []
        for k in list(move_stats.keys()):
            valid_gains = list(filter(lambda g: g < 100, move_stats[k]))
            if len(valid_gains) == 0:
                expectation_gain = np.mean(move_stats[k])
            else:
                expectation_gain = np.mean(valid_gains)
            expectation_move_gain.append((k, expectation_gain))

        expectation_move_gain = list(sorted(expectation_move_gain, key=lambda g: g[1]))
        top_gain_move = expectation_move_gain[-1]
        move_index = feasible_actions[top_gain_move[0]]

        res = np.zeros([1, 5])
        res[0][move_index] = 1.0
        return res

    def get_home_policy(self, rule_state):
        jobs, walls, me, rival, field_size = rule_state
        # jobs_dict = dict([(_x["x"] * field_size + _x["y"], _x["value"]) for _x in jobs])
        walls_dict = dict([(_x["x"] * field_size + _x["y"], True) for _x in walls])


        me_full = 0.0 if me["n_jobs"] < self.max_capacity else 1.0

        if self.steps_left <= 20 or me["n_jobs"] >= self.max_capacity - 1:
            # -------------------
            map2d_me = Array2D(field_size, field_size)
            map2d_rival = Array2D(field_size, field_size)
            for i in range(field_size):
                for j in range(field_size):
                    if i * field_size + j in walls_dict:
                        map2d_me[i][j] = 1
                        map2d_rival[i][j] = 1
                    if (i, j) == (rival["home_x"], rival["home_y"]):
                        map2d_me[i][j] = 1
                    if (i, j) == (me["home_x"], me["home_y"]):
                        map2d_rival[i][j] = 1

            home_astar=AStar(map2d_me, Point(me["x"], me["y"]), Point(me["home_x"], me["home_y"]))
            self_home_path = home_astar.start()
            if self_home_path is None:
                home_distance = 0
                if (me["x"], me["y"]) == (me["home_x"], me["home_y"]):
                    home_move = (me["home_x"], me["home_y"])
                else:
                    aaaa = 1
            else:
                home_distance = len(self_home_path)
                home_move = self_home_path[0].x, self_home_path[0].y
            dx = home_move[0] - me["x"]
            dy = home_move[1] - me["y"]
        else:
            dx, dy = 0, 0
            home_distance = 0

        move_vector = np.zeros([len(feasible_actions)])
        if dx == 1:
            move_vector[feasible_actions["D"]] = 1.0
        elif dx == -1:
            move_vector[feasible_actions["U"]] = 1.0
        elif dy == 1:
            move_vector[feasible_actions["R"]] = 1.0
        elif dy == -1:
            move_vector[feasible_actions["L"]] = 1.0
        else:
            move_vector[feasible_actions["S"]] = 1.0

        spare = 1
        # spare += int(home_distance // 5)
        force_home2 = min(self.steps_left - home_distance - spare, field_size) <= 0
        force_home1 = me_full > 0

        return np.array([force_home1 or force_home2]), np.array([move_vector])

    def get_movement(self, state, use_policy=True, rule_state=None, epsilon=0.05, random_move=False, stay=False):

        if use_policy:
            vision, extra_tensor = state
            force_cond1 = extra_tensor[:, 2] > np.zeros_like(extra_tensor)[:, 0]
            force_cond2 = extra_tensor[:, 4] <= np.zeros_like(extra_tensor)[:, 0]
            force_home_cond = np.logical_or(force_cond1, force_cond2)
            home_moves = extra_tensor[:, -5:]
            moves = self.policy_movement(state, epsilon)
        elif random_move:
            # batch_size = 1
            force_home_cond, home_moves = self.get_home_policy(rule_state)
            moves, _ = self.random_move()
        elif stay:
            force_home_cond, home_moves = self.get_home_policy(rule_state)
            moves = np.zeros([1, len(feasible_actions)])
            moves[0][feasible_actions["S"]] = 1.0
        else:
            assert rule_state is not None
            force_home_cond, home_moves = self.get_home_policy(rule_state)
            moves = self.greedy_stepwise_gain(rule_state)

        result_moves = np.where(force_home_cond, home_moves, moves)
        move_indicies = np.argmax(result_moves, axis=1)
        move_symbols = []
        for mv in list(move_indicies):
            symbol = action_symbols[int(mv)]
            move_symbols.append(symbol)
        return result_moves, move_symbols

    def get_gains(self, player):
        pickup_gain = self.cur_m2_me - self.last_m2_me
        deliver_gain = self.cur_m1_me - self.last_m1_me
        return (pickup_gain, deliver_gain)

    def random_move(self):
        # batch size = 1
        move, mv_ind = random.choice(list(feasible_actions.items()))
        move_tensor = np.zeros([1, len(feasible_actions)])
        move_tensor[0][mv_ind] = 1.0
        return move_tensor, [move]

    def hit_wall(self, rule_s, next_rule_s, is_player1, action):
        if is_player1:
            me = rule_s["player1"]
            me_next = next_rule_s['player1']
            rival = rule_s['player2']
            rival_next = next_rule_s['player2']
        else:
            rival = rule_s["player1"]
            rival_next = next_rule_s['player1']
            me = rule_s['player2']
            me_next = next_rule_s['player2']

        if action[0] != feasible_actions["S"] and (me["x"], me["y"])==(me_next["x"], me_next["y"]):
            return True
        else:
            return False
