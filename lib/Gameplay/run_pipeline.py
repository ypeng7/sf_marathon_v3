from lib.server.models import Model
from RL.model import SimpleCNN, QNConfig
from utils.a_star import AStar, Array2D, Point
import random
import numpy as np

class Agent(object):
    def __init__(self, max_capacity, name, max_steps):
        self.max_capacity = float(max_capacity)
        self.last_m2_me = 0  # m2 = score + value
        self.last_m1_me = 0 # m1 = score
        self.cur_m2_me = 0 # 当前m2
        self.name = name
        self.steps_left = max_steps


    def observe_state(self, jobs, walls, me, rival, field_size):
        """
        实现特征工程
        :param jobs:
        :param walls:
        :param me:
        :param rival:
        :param field_size:
        :return: 返回 self.policy_movement(feature_state)的输入参数
        """
        return np.zeros([field_size, field_size])

    def move(self, movement, env):
        """
        :param movement: 上下左右停
        :param env: 环境
        :return:
        """
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

    def get_movement(self, feature_state):
        """
        根据feature state产生上下左右停,
        可以组合不同policy
        :param feature_state:
        :return:
        """
        return "L"



    def get_gains(self, player):
        """
        :param player:
        :return: 计算当前一步拣取包裹的价值, 拾取价值以及送家价值
        """
        pickup_gain = self.cur_m2_me - self.last_m2_me
        deliver_gain = self.cur_m1_me - self.last_m1_me
        return (pickup_gain, deliver_gain)

    def astar_path(self, map, source_x, source_y, dest_x, dest_y):
        """
        astar寻路使用方法见本方法内容
        :param map:
        :param source_x:
        :param source_y:
        :param dest_x:
        :param dest_y:
        :return:
        """
        map = Array2D(12, 12)
        map[2][2] = 1.0
        map[2][3] = 1.0
        map[2][4] = 1.0

        source_x, source_y = 0, 3
        dest_x, dest_y = 4, 3

        astar = AStar(map, Point(source_x, source_y), Point(dest_x, dest_y))
        rival_path_list = astar.start()
        assert rival_path_list is not None  # 注意处理空值
        print(rival_path_list)

        return


class Gameplay(object):
    env_class = Model()
    env = env_class.create_env("round1", "p1", "p2", seed=0)

    def __init__(self):
        self.t = 0
        self.env_class = Gameplay.env_class
        self.env = Gameplay.env

    def run_gameplay(self,):


        env_state = self.env.reset()

        if random.random() < 0.5:
            self.me = Agent(self.env.conf["capacity"], self.env.player1.name, self.env.conf["max_steps"])
            self.rival = Agent(self.env.conf["capacity"], self.env.player2.name, self.env.conf["max_steps"])
            self.is_player1 = True
        else:
            self.rival = Agent(self.env.conf["capacity"], self.env.player1.name, self.env.conf["max_steps"])
            self.me = Agent(self.env.conf["capacity"], self.env.player2.name, self.env.conf["max_steps"])
            self.is_player1 = False

        for _ in range(200): #最大200步
            env_state = self.env.get_state()

            # --- p1 move --------
            if self.is_player1:
                agt1_state = self.me.observe_state(env_state["jobs"], env_state["walls"], env_state["player1"], env_state["player2"], self.env.conf["world_size"])
                agt1_move_symbol = self.me.get_movement(agt1_state)
                env_state = self.me.move(agt1_move_symbol, self.env)
                gain_me = self.me.get_gains(env_state["player1"])
            else:
                agt1_state = self.rival.observe_state(env_state["jobs"], env_state["walls"], env_state["player1"], env_state["player2"], self.env.conf["world_size"])
                agt1_move_symbol = self.rival.get_movement(agt1_state)
                env_state = self.rival.move(agt1_move_symbol, self.env)
                gain_rival = self.rival.get_gains(env_state["player1"])

            # --- p2 move ---------
            if self.is_player1:
                agt2_state = self.rival.observe_state(env_state["jobs"], env_state["walls"], env_state["player2"], env_state["player1"], self.env.conf["world_size"])
                agt2_move_symbol = self.rival.get_movement(agt2_state)
                env_state = self.rival.move(agt2_move_symbol, self.env)
                gain_rival = self.rival.get_gains(env_state["player2"])
            else:
                agt2_state = self.me.observe_state(env_state["jobs"], env_state["walls"], env_state["player2"], env_state["player1"], self.env.conf["world_size"])
                agt2_move_symbol = self.me.get_movement(agt2_state)
                env_state = self.me.move(agt2_move_symbol, self.env)
                gain_me = self.me.get_gains(env_state["player2"])


            print(gain_me, gain_rival)
            print(env_state['player1'], env_state['player2'])

if __name__ == '__main__':
    gp = Gameplay()
    gp.run_gameplay()
