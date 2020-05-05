from lib.server.models import Model
from RL.agent import Agent
from RL.model import SimpleCNN, QNConfig
import random
import numpy as np
import copy
import pickle

project_root = "/Users/st491/LocalStore/Git/ai_battle"
max_step = 200

class TrainProcess(object):
    env_class = Model()
    env = env_class.create_env("round0", "p1", "p2", seed=0)
    feasible_actions = {"U": 0, "D": 1, "L": 2, "R": 3, "S": 4}
    action_symbols = {0: "U", 1: "D", 2: "L", 3: "R", 4: "S"}
    # state = env.reset()
    # games = env_class.get_envs()
    dqn_sizes = (env.conf["world_size"]*2-1, env.conf["world_size"]*2-1, 5, len(feasible_actions))
    cfg = QNConfig(project_root + "/persist")
    q_network = SimpleCNN(cfg, None, dqn_sizes, "dqn")
    q_network.initialize_variables()

    def __init__(self):
        self.t = 0
        self.env_class = TrainProcess.env_class
        self.env = TrainProcess.env
        self.state = self.env.reset()
        self.game = self.env_class.get_envs()[0]
        self.max_episode_steps = self.env.conf["max_steps"]

        if random.random() < 0.5:
            self.me = Agent(self.env.conf["capacity"], 11, TrainProcess.q_network, self.env.player1.name, self.env.conf["max_steps"])
            self.rival = Agent(self.env.conf["capacity"], 11, TrainProcess.q_network, self.env.player2.name, self.env.conf["max_steps"])
            self.is_player1 = True
        else:
            self.rival = Agent(self.env.conf["capacity"], 11, TrainProcess.q_network, self.env.player1.name, self.env.conf["max_steps"])
            self.me = Agent(self.env.conf["capacity"], 11, TrainProcess.q_network, self.env.player2.name, self.env.conf["max_steps"])
            self.is_player1 = False


    def restart_episode(self, is_player1, is_eval=False):
        if not is_eval:
            round_name = "round1"
        else:
            round_name = "eval_round"

        self.env_class.del_env(round_name)
        self.env = self.env_class.create_env(round_name, "p1", "p2", seed=random.randint(0, 10000))
        self.state = self.env.reset()

        if is_player1:
            self.me = Agent(self.env.conf["capacity"], 11, TrainProcess.q_network, self.env.player1.name, self.env.conf["max_steps"])
            self.rival = Agent(self.env.conf["capacity"], 11, TrainProcess.q_network, self.env.player2.name, self.env.conf["max_steps"])
            self.is_player1 = True
        else:
            self.rival = Agent(self.env.conf["capacity"], 11, TrainProcess.q_network, self.env.player1.name, self.env.conf["max_steps"])
            self.me = Agent(self.env.conf["capacity"], 11, TrainProcess.q_network, self.env.player2.name, self.env.conf["max_steps"])
            self.is_player1 = False


    def rival_move(self, epsilons, epsilon_t, rival_is_rl, me_pos, rival_pos, rival_observe=None):
        epsilon = epsilons[min(epsilon_t, len(epsilons)-1)]
        env_state = self.env.get_state()
        agt_rule_state = (env_state["jobs"], env_state["walls"], env_state[rival_pos], env_state[me_pos], self.env.conf["world_size"])
        if rival_is_rl and rival_observe is None:
            agt_state = self.rival.observe_state(env_state["jobs"], env_state["walls"], env_state[rival_pos], env_state[me_pos], self.env.conf["world_size"])
        else:
            agt_state = rival_observe
        agt_movement, agt_move_symbol = self.rival.get_movement(agt_state, use_policy=rival_is_rl, rule_state=agt_rule_state, epsilon=epsilon, stay=False)
        env_state = self.rival.move(agt_move_symbol[0], self.env)
        gain_rival = self.rival.get_gains(env_state[rival_pos])
        return gain_rival


    def me_move(self, epsilons, epsilon_t, observe, me_pos, rival_pos, me_is_random=False):
        epsilon = epsilons[min(epsilon_t, len(epsilons)-1)]
        env_state = self.env.get_state()
        if me_is_random:
            agt_rule_state = (env_state["jobs"], env_state["walls"], env_state[me_pos], env_state[rival_pos], self.env.conf["world_size"])
            agt_movement, agt_move_symbol = self.me.get_movement(None, use_policy=False, rule_state=agt_rule_state, random_move=True)
        else:
            agt_state = observe
            agt_movement, agt_move_symbol = self.me.get_movement(agt_state, epsilon=epsilon)
        env_state = self.me.move(agt_move_symbol[0], self.env)
        gain_me = self.me.get_gains(env_state[me_pos])
        action_me = np.argmax(agt_movement, axis=1)
        return gain_me, action_me


    def mock_one_round(self, epsilons, epsilon_t, local_step, is_p1, rival_is_rl=False):
        replay_buffer = []
        done = False
        if local_step % max_step == max_step-1:
            done=True

        if is_p1==False and local_step % max_step == 0:
            state = self.env.get_state()
            observe = self.me.observe_state(state['jobs'], state['walls'], state['player2'], state['player1'], self.env.conf["world_size"])
            gain_me = (0, 0)
            action_me = np.array([TrainProcess.feasible_actions['S']])
            if rival_is_rl:
                rival_observe = self.rival.observe_state(state['jobs'], state['walls'], state['player1'], state['player2'], self.env.conf["world_size"])
                gain_rival = self.rival_move(epsilons, epsilon_t, True, "player2", "player1", rival_observe=rival_observe)
            else:
                gain_rival = self.rival_move(epsilons, epsilon_t, False, "player2", "player1")
            state_next = self.env.get_state()
            observe_next = self.me.observe_state(state_next['jobs'], state_next['walls'], state_next['player2'], state_next['player1'], self.env.conf["world_size"])
        else:
            me_pos = 'player1' if is_p1 else 'player2'
            rival_pos = 'player2' if is_p1 else 'player1'
            state = self.env.get_state()
            observe = self.me.observe_state(state['jobs'], state['walls'], state[me_pos], state[rival_pos], self.env.conf["world_size"])
            gain_me, action_me = self.me_move(epsilons, epsilon_t, observe, me_pos, rival_pos)
            if rival_is_rl:
                rival_observe = self.rival.observe_state(state['jobs'], state['walls'], state[rival_pos], state[me_pos], self.env.conf["world_size"])
                gain_rival = self.rival_move(epsilons, epsilon_t, True, me_pos, rival_pos, rival_observe=rival_observe)
            else:
                gain_rival = self.rival_move(epsilons, epsilon_t, False, me_pos, rival_pos)
            state_next = self.env.get_state()
            observe_next = self.me.observe_state(state_next['jobs'], state_next['walls'], state_next[me_pos], state_next[rival_pos], self.env.conf["world_size"])

        gains = (gain_me[0], gain_me[1], gain_rival[0])
        replay_buffer.append(copy.deepcopy((state, state_next, observe, observe_next, gains, action_me, is_p1, done)))

        if is_p1==False and local_step % max_step == max_step-1:
            state = self.env.get_state()
            observe = self.me.observe_state(state['jobs'], state['walls'], state['player2'], state['player1'], self.env.conf["world_size"])
            gain_me, action_me = self.me_move(epsilons, epsilon_t, observe, "player2", "player1")
            gain_rival = (0, 0)
            gains = (gain_me[0], gain_me[1], gain_rival[0])
            state_next = self.env.get_state()
            observe_next = self.me.observe_state(state_next['jobs'], state_next['walls'], state_next['player2'], state_next['player1'], self.env.conf["world_size"])
            replay_buffer.append(copy.deepcopy((state, state_next, observe, observe_next, gains, action_me, is_p1, done)))

        return replay_buffer


    def save_buffer(self):
        epsilon_start, epsilon_end, epsilon_decay_steps = 1.0, 0.99, 500000
        epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

        replay_buffer = []
        is_p1 = random.choice([True, False])
        buffer_number = 30000
        for t_buffer in range(buffer_number):
            done = False
            if t_buffer % max_step == max_step-1:
                done=True

            if t_buffer % max_step == 0:
                is_p1 = random.choice([True, False])
                self.restart_episode(is_p1)

            if t_buffer % 1000 == 0:
                print("loading replay buffer: " + str(t_buffer))

            if is_p1==False and t_buffer % max_step == 0:
                state = self.env.get_state()
                observe = self.me.observe_state(state['jobs'], state['walls'], state['player2'], state['player1'], self.env.conf["world_size"])
                gain_me = (0, 0)
                action_me = np.array([TrainProcess.feasible_actions['S']])
                gain_rival = self.rival_move(epsilons, t_buffer, False, "player2", "player1")
                state_next = self.env.get_state()
                observe_next = self.me.observe_state(state_next['jobs'], state_next['walls'], state_next['player2'], state_next['player1'], self.env.conf["world_size"])
            else:
                me_pos = 'player1' if is_p1 else 'player2'
                rival_pos = 'player2' if is_p1 else 'player1'
                state = self.env.get_state()
                observe = self.me.observe_state(state['jobs'], state['walls'], state[me_pos], state[rival_pos], self.env.conf["world_size"])
                gain_me, action_me = self.me_move(epsilons, t_buffer, observe, me_pos, rival_pos)
                gain_rival = self.rival_move(epsilons, t_buffer, False, me_pos, rival_pos)
                state_next = self.env.get_state()
                observe_next = self.me.observe_state(state_next['jobs'], state_next['walls'], state_next[me_pos], state_next[rival_pos], self.env.conf["world_size"])

            gains = (gain_me[0], gain_me[1], gain_rival[0])
            replay_buffer.append(copy.deepcopy((state, state_next, observe, observe_next, gains, action_me, is_p1, done)))

            if is_p1==False and t_buffer % max_step == max_step-1:
                state = self.env.get_state()
                observe = self.me.observe_state(state['jobs'], state['walls'], state['player2'], state['player1'], self.env.conf["world_size"])
                gain_me, action_me = self.me_move(epsilons, t_buffer, observe, "player2", "player1")
                gain_rival = (0, 0)
                gains = (gain_me[0], gain_me[1], gain_rival[0])
                state_next = self.env.get_state()
                observe_next = self.me.observe_state(state_next['jobs'], state_next['walls'], state_next['player2'], state_next['player1'], self.env.conf["world_size"])
                replay_buffer.append(copy.deepcopy((state, state_next, observe, observe_next, gains, action_me, is_p1, done)))

        pickle.dump(replay_buffer, open("../../persist/replay_buffer2.cpkl", "wb"))
        return replay_buffer

    def load_buffer(self):
        replay_buffer = pickle.load(open("../../persist/replay_buffer2.cpkl", "rb"))
        return replay_buffer

    def evaluate_dqn(self):
        print("========start evaluate against benchmark ======")
        ratios = []
        epsilon = 0.0
        for ep_i in range(5):
            for is_player1 in [True, False]:
                self.restart_episode(is_player1=is_player1, is_eval=True)
                r_accu = 0.0
                for t in range(max_step):
                    if ep_i == 0 and is_player1 and t < 20:
                        print_enable = True
                    else:
                        print_enable = False

                    env_state = self.env.get_state()
                    if is_player1:
                        agt1_state = self.me.observe_state(env_state["jobs"], env_state["walls"], env_state["player1"], env_state["player2"], self.env.conf["world_size"])
                        agt1_movement, agt1_move_symbol = self.me.get_movement(agt1_state, epsilon=epsilon)

                        if print_enable:
                            self.me.rough_plot_board(env_state) # (-20 * agt1_state[0][0, :, :, 0] + agt1_state[0][0, :, :, 1], lambda x: x)
                            print("me move is :" + agt1_move_symbol[0])

                        rule_s = copy.deepcopy(env_state)
                        env_state = self.me.move(agt1_move_symbol[0], self.env)
                        rule_s_ = copy.deepcopy(env_state)
                        gain_me = self.me.get_gains(env_state["player1"])
                        action_me = np.argmax(agt1_movement, axis=1)
                    else:
                        agt1_state = None
                        agt1_rule_state = (env_state["jobs"], env_state["walls"], env_state["player1"], env_state["player2"], self.env.conf["world_size"])
                        agt1_movement, agt1_move_symbol = self.rival.get_movement(agt1_state, use_policy=False, rule_state=agt1_rule_state, stay=False)
                        env_state = self.rival.move(agt1_move_symbol[0], self.env)
                        gain_rival = self.rival.get_gains(env_state["player1"])

                    if is_player1:
                        agt2_state = None
                        agt2_rule_state = (env_state["jobs"], env_state["walls"], env_state["player2"], env_state["player1"], self.env.conf["world_size"])
                        agt2_movement, agt2_move_symbol = self.rival.get_movement(agt2_state, use_policy=False, rule_state=agt2_rule_state, stay=False)
                        if print_enable:
                            self.me.rough_plot_board(env_state) # (-20 * agt1_state[0][0, :, :, 0] + agt1_state[0][0, :, :, 1], lambda x: x)
                            print("rival move is :" + agt2_move_symbol[0])
                        env_state = self.rival.move(agt2_move_symbol[0], self.env)
                        gain_rival = self.rival.get_gains(env_state["player2"])
                    else:
                        agt2_state = self.me.observe_state(env_state["jobs"], env_state["walls"], env_state["player2"], env_state["player1"], self.env.conf["world_size"])
                        agt2_movement, agt2_move_symbol = self.me.get_movement(agt2_state, epsilon=epsilon)
                        rule_s = copy.deepcopy(env_state)
                        env_state = self.me.move(agt2_move_symbol[0], self.env)
                        rule_s_ = copy.deepcopy(env_state)
                        gain_me = self.me.get_gains(env_state["player2"])
                        action_me = np.argmax(agt2_movement, axis=1)

                    rs = (gain_me[0], gain_me[1], gain_rival[0])
                    is_p1 = is_player1
                    a = action_me

                    if self.me.hit_wall(rule_s, rule_s_, is_p1, a):
                        wall_penalty = -20.0
                    elif rs[0] > 0 or rs[1] > 0:
                        wall_penalty = 0.0
                    else:
                        wall_penalty = -1.0
                        if a[0] == 4:
                            wall_penalty = -2.0
                    r = (rs[0]-rs[2]) * 0.85 + rs[1] * 0.15 + wall_penalty
                    r_accu += r

                    if print_enable:
                        print("me reward= "+str(r) + " ; rwd_accu= "+str(r_accu))

                print("game episode finished with me=" + str((self.me.cur_m2_me, self.me.cur_m1_me)) + " ; rival="+str((self.rival.cur_m2_me, self.rival.cur_m1_me)) + " ; rwd_accu= "+str(r_accu))
                if self.rival.cur_m1_me > 0:
                    ratios.append(self.me.cur_m1_me/float(self.rival.cur_m1_me))

        print("advantage ratio is : " + str(np.mean(ratios)))
        print("========evaluate finished =====================")
        return


    def start_train(self):
        replay_buffer = self.load_buffer()
        epsilon_start, epsilon_end, epsilon_decay_steps = 0.99, 0.01, 30000
        epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
        batch_size = 64
        gamma = 0.93

        is_p1 = random.choice([True, False])
        for t in range(200000):

            t_eps = max(random.randint(t - 1000, t+1000), 0)
            rival_is_rl = t > 15000
            if t % 3000 == 0 and t > 0:
                print("save_model")
                self.me.model.save_best_models(t, 1000-t//1000)
                print("model saved")
                self.evaluate_dqn()

            if t % max_step == 0:
                is_p1 = random.choice([True, False])
                self.restart_episode(is_p1, is_eval=False)

            buffer_increment = self.mock_one_round(epsilons, t_eps, t, is_p1, rival_is_rl=rival_is_rl)
            if t % max_step == max_step -1:
                state_fin = self.env.get_state()
                me_pos = 'player1' if is_p1 else 'player2'
                rival_pos = 'player2' if is_p1 else 'player1'
                print(state_fin[me_pos]['score'], state_fin[me_pos]['value'], state_fin[rival_pos]['score'], state_fin[rival_pos]['value'])
                print(epsilons[min(t_eps, len(epsilons)-1)])

            if len(replay_buffer) > 30000:
                for _ in range(len(buffer_increment)):
                    replay_buffer.pop(0)
            replay_buffer.extend(buffer_increment)

            samples = random.sample(replay_buffer, batch_size)
            s_batch, se_batch, s_n_batch, se_n_batch, r_batch, a_batch = self.process_batch(samples, batch_size)

            q_next = self.me.model.predict(s_n_batch, se_n_batch)
            best_action_next = np.argmax(q_next, axis=1)

            best_values = np.max(q_next, axis=1)
            superioty = np.tile(np.expand_dims(best_values, axis=1), [1, 5]) - q_next
            batch_superioty = np.mean(np.sum(superioty, axis=1) / 4.0)

            q_target = r_batch + gamma * q_next[np.arange(batch_size), best_action_next]
            loss, action_qs = self.me.model.update(s_batch, se_batch, a_batch, q_target)
            if t % 100 == 0:
                print(t, loss, q_next[0], batch_superioty)




    def process_batch(self, samples, batch_size):
        s_batch = []
        se_batch = []
        s_next_batch = []
        se_next_batch = []
        r_batch = []
        a_batch = []

        for bi in range(batch_size):
            rule_s, rule_s_, s, s_, rs, a, is_p1, done = samples[bi]

            observed, observed_extra = s
            observed_next, observed_next_extra = s_

            # if observed.shape[-1]==5:
            #     observed[:, :, :, 0] -= 2.0 * observed[:, :, :, 4]
            #     observed = observed[:, :, :, :2]
            # if observed_next.shape[-1]==5:
            #     observed_next[:, :, :, 0] -= 2.0 * observed_next[:, :, :, 4]
            #     observed_next = observed_next[:, :, :, :2]

            if self.me.hit_wall(rule_s, rule_s_, is_p1, a):
                wall_penalty = -20.0
            elif rs[0] > 0 or rs[1] > 0:
                wall_penalty = 0.0
            else:
                wall_penalty = -1.0
                if a[0] == 4:
                    wall_penalty = -2.0

            s_batch.append(observed)
            se_batch.append(observed_extra)
            s_next_batch.append(observed_next)
            se_next_batch.append(observed_next_extra)
            a_batch.append(a)

            r = (rs[0]-rs[2]) * 0.85 + rs[1] * 0.15 + wall_penalty

            r_batch.append(r)

        return np.concatenate(s_batch, axis=0), np.concatenate(se_batch, axis=0), \
               np.concatenate(s_next_batch, axis=0), np.concatenate(se_next_batch, axis=0), \
               np.array(r_batch), np.concatenate(a_batch, axis=0)

    def test_astar(self,):
        from utils.a_star import AStar, Array2D, Point
        acc = 0
        for i in range(100000):
            field = Array2D(12, 12)
            walls = []
            for _ in range(26):
                x = random.randint(0, 11)
                y = random.randint(0, 11)
                field[x][y] = 1.0
                walls.append((x, y))

            p1x, p1y = random.randint(0, 11), random.randint(0, 11)
            while (p1x, p1y) in walls:
                p1x, p1y = random.randint(0, 11), random.randint(0, 11)

            p2x, p2y = random.randint(0, 11), random.randint(0, 11)
            while (p2x, p2y) in walls:
                p2x, p2y = random.randint(0, 11), random.randint(0, 11)
            path = AStar(field, Point(p1x, p1y), Point(p2x, p2y))
            pl = path.start()
            if pl is not None:
                a = len(pl)

            if i % (50) == 0:
                acc +=1

            if acc % 20 == 0:
                print(acc)


if __name__ == '__main__':
    mock = TrainProcess()
    # mock.save_buffer()
    # mock.load_buffer()
    # mock.evaluate_dqn(False, False)
    # mock.test_astar()
    mock.start_train()