import numpy as np
import ray
import random
import time
import os
from tqdm import tqdm

from .game import GameHistory


@ray.remote
class ReplayBuffer(object):
    """Reference : DISTRIBUTED PRIORITIZED EXPERIENCE REPLAY
    Algo. 1 and Algo. 2 in Page-3 of (https://arxiv.org/pdf/1803.00933.pdf
    """

    def __init__(self, replay_buffer_id, make_dataset=False, reset_prior=True, config=None):
        self.config = config
        self.soft_capacity = config.window_size
        self.batch_size = config.batch_size
        self.make_dataset = make_dataset
        self.replay_buffer_id = replay_buffer_id
        self.reset_prior = reset_prior
        self.keep_ratio = 1

        self.model_index = 0
        self.model_update_interval = 10

        self.buffer = []
        self.priorities = []
        self.game_look_up = []
        self.tail_index = []
        self.tail_len = 5
        self.tail_ratio = self.config.tail_ratio

        self._eps_collected = 0
        self.base_idx = 0
        self._alpha = config.priority_prob_alpha
        self.transition_top = int(config.transition_num *100* 10 ** 4)#@wjc, changed from 10**6 for testing
        print("transition top is {}".format(self.transition_top),flush=True)
        self.clear_time = 0

    def load_files(self, path=None):
        if path is None:
            path = self.config.exp_path

        dir = os.path.join(path, 'replay', str(self.replay_buffer_id))
        dir='/home/game/jan/buffer'
        print('Loading from ', dir, ' ...')
        assert os.path.exists(dir)

        self.priorities = np.load(os.path.join(dir, 'prior.npy'))
        if self.reset_prior:
            self.priorities = np.ones_like(self.priorities)

        self.game_look_up = np.load(os.path.join(dir, 'game_look_up.npy')).tolist()
        self.base_idx, buffer_len = np.load(os.path.join(dir, 'utils.npy')).tolist()

        assert len(self.priorities) == len(self.game_look_up)
        assert self.game_look_up[-1][0] == self.base_idx + buffer_len - 1

        env = self.config.new_game(0)
        ls=time.time()
        gamebuffer=d2=np.load(os.path.join(dir, "gamebuffer.npy"), allow_pickle=True).item()

        print("load game buffer takes {}".format(time.time()-ls),flush=True)
        for i in tqdm(range(buffer_len)):
            game = GameHistory(env.env.action_space, max_length=self.config.history_length, config=self.config)
            game.load_file(gamebuffer[str(i)])
            self.buffer.append(game)

        print('Load Over.')

    def save_files(self,id):
        dir = os.path.join(self.config.exp_path, 'replay', str(id))
        print('dir: ', dir)
        if not os.path.exists(dir):
            os.makedirs(dir)

        np.save(os.path.join(dir, 'prior.npy'), np.array(self.priorities))
        np.save(os.path.join(dir, 'game_look_up.npy'), np.array(self.game_look_up))
        np.save(os.path.join(dir, 'utils.npy'), np.array([self.base_idx, len(self.buffer)]))


        st=time.time()
        buffer_dict={}
        for i, game in enumerate(self.buffer):
            buffer_dict[str(i)]=game.save_file()
        np.save(os.path.join(dir,'gamebuffer.npy'), buffer_dict)
        print("finish saving {} length buffer with {}s".format(len(self.buffer),time.time()-st),flush=True)

    def save_pools(self, pools, gap_step):
        if self.make_dataset:
            buffer_size = self.size()
            print('Current size: ', buffer_size)

        for (game, priorities) in pools:
            # Only append end game
            # if end_tag:
            # print("in save pool,len(game)=",len(game),flush=True )

            self.save_game(game, True, gap_step, priorities)

    def save_game(self, game, end_tag, gap_steps, priorities=None):
        if end_tag:
            self._eps_collected += 1
            valid_len = len(game)
        else:
            valid_len = len(game) - gap_steps

        if priorities is None:
            max_prio = self.priorities.max() if self.buffer else 1
            self.priorities = np.concatenate((self.priorities, [max_prio for _ in range(valid_len)] + [0. for _ in range(valid_len, len(game))]))
        else:
            assert len(game) == len(priorities), " priorities should be of same length as the game steps"
            priorities = priorities.copy().reshape(-1)
            # priorities[valid_len:len(game)] = 0.
            self.priorities = np.concatenate((self.priorities, priorities))

        self.buffer.append(game)
        self.game_look_up += [(self.base_idx + len(self.buffer) - 1, step_pos) for step_pos in range(len(game))]

        #@wjc
        total = self.get_total_len()
        beg_index = max(total - self.tail_len, 0)
        self.tail_index += [idx for idx in range(beg_index, total)]

    def get_game(self, idx):
        game_id, game_pos = self.game_look_up[idx]
        game_id -= self.base_idx
        game = self.buffer[game_id]
        return game

    def prepare_batch_context(self, batch_size, beta):
        assert beta > 0

        total = self.get_total_len()

        # uniform
        if random.random() < self.config.uniform_ratio:
            _alpha = 0.
        else:
            _alpha = self._alpha

        probs = self.priorities ** _alpha


        probs /= probs.sum()

        while total <=batch_size:
            time.sleep(1)
            print("in replay buffer's prepare_batch_context: total={0}, batch_size={1}".format(self.get_total_len(),batch_size),flush=True)
        indices_lst = np.random.choice(total, batch_size, p=probs, replace=False)

        weights_lst = (total * probs[indices_lst]) ** (-beta)
        weights_lst /= weights_lst.max()

        game_lst = []
        game_pos_lst = []

        for idx in indices_lst:
            game_id, game_pos = self.game_look_up[idx]
            game_id -= self.base_idx
            game = self.buffer[game_id]

            game_lst.append(game)
            game_pos_lst.append(game_pos)

        make_time = [time.time() for _ in range(len(indices_lst))]

        context = (game_lst, game_pos_lst, indices_lst, weights_lst, make_time)
        return context

    def update_priorities(self, batch_indices, batch_priorities, make_time):
        for i in range(len(batch_indices)):
            if make_time[i] > self.clear_time:
                idx, prio = batch_indices[i], batch_priorities[i]
                self.priorities[idx] = prio

    def remove_to_fit(self):
        current_size = self.size()

        total_transition = self.get_total_len()
        if total_transition > self.transition_top:
            print('Remove fit , current size(len(buffer)): {0}, total_transition(len(priority)): {1}/1e5 '.format(current_size,self.get_total_len()),flush=True)
            index = 0
            for i in range(current_size):
                total_transition -= len(self.buffer[i])
                if total_transition <= self.transition_top * self.keep_ratio:
                    index = i
                    break

            if total_transition >= self.config.batch_size:
                self._remove(index + 1)

    def _remove(self, num_excess_games):
        excess_games_steps = sum([len(game) for game in self.buffer[:num_excess_games]])
        del self.buffer[:num_excess_games]
        self.priorities = self.priorities[excess_games_steps:]
        del self.game_look_up[:excess_games_steps]
        self.base_idx += num_excess_games

        self.clear_time = time.time()

    def clear_buffer(self):
        del self.buffer[:]

    def size(self):
        return len(self.buffer)

    def episodes_collected(self):
        return self._eps_collected

    def get_batch_size(self):
        return self.batch_size

    def get_priorities(self):
        return self.priorities

    def get_total_len(self):
        return len(self.priorities)

    def over(self):
        if self.make_dataset:
            return self.get_total_len() >= self.transition_top
        return False
