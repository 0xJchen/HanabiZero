
@ray.remote(num_gpus=gpu_num,num_cpus=1)
class BatchWorker(object):
    def __init__(self, worker_id, replay_buffer, storage, batch_storage, config):
        self.worker_id = worker_id
        self.replay_buffer = replay_buffer
        self.storage = storage
        self.batch_storage = batch_storage
        self.config = config

        if config.amp_type == 'nvidia_apex':
            self.target_model = amp.initialize(config.get_uniform_network().to(self.config.device))
        else:
            self.target_model = config.get_uniform_network()
            self.target_model.to('cuda')
        self.target_model.eval()

        self.last_model_index = -1
        self.batch_max_num = 40#from 20
        self.beta_schedule = LinearSchedule(config.training_steps + config.last_steps, initial_p=config.priority_prob_beta, final_p=1.0)

    def run(self):
        start = False
        # print("batch worker initialize",flush=True)
        while True:
            # wait for starting
            if not start:
                start = ray.get(self.storage.get_start_signal.remote())
                time.sleep(1)
                # print("batch worker waiting replay_buffer.get_total_len.remote()) >= config.start_window_size",flush=True)

                continue
            # TODO: use latest weights for policy reanalyze
            ray_data_lst = [self.storage.get_counter.remote(), self.storage.get_target_weights.remote()]
            trained_steps, target_weights = ray.get(ray_data_lst)

            beta = self.beta_schedule.value(trained_steps)
            batch_context = ray.get(self.replay_buffer.prepare_batch_context.remote(self.config.batch_size, beta))

            #@wjc
            game_lst, game_pos_lst, _, _, _=batch_context
            #temporarily save the obs list

            if trained_steps >= self.config.training_steps + self.config.last_steps:
                # print("batchworker finished working",flush=True)
                break

            new_model_index = trained_steps // self.config.target_model_interval
            if new_model_index > self.last_model_index:
                self.last_model_index = new_model_index
            else:
                target_weights = None

            if self.batch_storage.get_len() < self.batch_max_num:
                #should be zero as no batch is pushed
                # print("batch storage size={0}/20 ".format(self.batch_storage.get_len()),flush=True)
                try:
                    batch = self.make_batch(batch_context, self.config.revisit_policy_search_rate, weights=target_weights, batch_num=2)
                    # print("batch worker finish makeing batch, start to push",flush=True)
                    #if self.batch_storage.is_full():
                     #   print("{} is sleeping, buffer={}".format(self.worker_id,self.batch_storage.get_len()),flush=True)
                    self.batch_storage.push(batch)
                except:
                    print('=====================>Data is deleted...')
                    #assert False

    def split_context(self, batch_context, split_num):#game_lst, game_pos_lst, indices_lst, weights_lst, make_time_lst = batch_context
        batch_context_lst = []

        context_num = len(batch_context)
        batch_size = len(batch_context[0])
        split_size = batch_size // split_num

        assert split_size * split_num == batch_size

        for i in range(split_num):
            beg_index = split_size * i
            end_index = split_size * (i + 1)

            _context = []
            for j in range(context_num):
                _context.append(batch_context[j][beg_index:end_index])

            batch_context_lst.append(_context)

        return batch_context_lst

    def concat_batch(self, batch_lst):
        batch = [[] for _ in range(8 + 1)]

        for i in range(len(batch_lst)):
            for j in range(len(batch_lst[0])):
                if i == 0:
                    batch[j] = batch_lst[i][j]
                else:
                    batch[j] = np.concatenate((batch[j], batch_lst[i][j]), axis=0)
        return batch

    def make_batch(self, batch_context, ratio, weights=None, batch_num=2):
        batch_context_lst = self.split_context(batch_context, batch_num)#game_lst, game_pos_lst, indices_lst, weights_lst, make_time_lst = batch_context
        batch_lst = []
        for i in range(len(batch_context_lst)):

            batch_lst.append(self._make_batch(batch_context_lst[i], ratio, weights))
        return self.concat_batch(batch_lst)

    def _make_batch(self, batch_context, ratio, weights=None):
        if weights is not None:
            self.target_model.set_weights(weights)
            self.target_model.to('cuda')
            self.target_model.eval()
            # print('weight is not none! change weights!',flush=True)
        game_lst, game_pos_lst, indices_lst, weights_lst, make_time_lst = batch_context
        batch_size = len(indices_lst)
        obs_lst, action_lst, mask_lst = [], [], []
        for i in range(batch_size):
            game = game_lst[i]
            game_pos = game_pos_lst[i]
            _actions = game.actions[game_pos:game_pos + self.config.num_unroll_steps].tolist()
            _mask = [1. for i in range(len(_actions))]
            _mask += [0. for _ in range(self.config.num_unroll_steps - len(_mask))]
            _actions += [np.random.randint(0, game.action_space_size) for _ in range(self.config.num_unroll_steps - len(_actions))]

            obs_lst.append(game_lst[i].obs(game_pos_lst[i], extra_len=self.config.num_unroll_steps, padding=True))#<===== problem

            action_lst.append(_actions)
            mask_lst.append(_mask)
        re_num = int(batch_size * ratio)#ratio is config.revisit_policy_search_rate
        obs_lst = prepare_observation_lst(obs_lst, image_based=self.config.image_based)
        batch = [obs_lst, action_lst, mask_lst, [], [], [], indices_lst, weights_lst, make_time_lst]
        if self.config.reanalyze_part == 'paper':
            re_value, re_reward, re_policy = prepare_multi_target(self.replay_buffer, indices_lst[:re_num],
                                                                  make_time_lst[:re_num],
                                                                  game_lst[:re_num], game_pos_lst[:re_num],
                                                                  self.config, self.target_model)
            batch[3].append(re_reward)
            batch[4].append(re_value)
            batch[5].append(re_policy)
            # only value
            if re_num < batch_size:
                re_value, re_reward, re_policy = prepare_multi_target_only_value(game_lst[re_num:], game_pos_lst[re_num:],
                                                                                 self.config, self.target_model)
                batch[3].append(re_reward)
                batch[4].append(re_value)
                batch[5].append(re_policy)
        elif self.config.reanalyze_part == 'none':
            re_value, re_reward, re_policy = prepare_multi_target_none(game_lst[:], game_pos_lst[:],
                                                                                 self.config, self.target_model)
            batch[3].append(re_reward)
            batch[4].append(re_value)
            batch[5].append(re_policy)
        else:
            assert self.config.reanalyze_part == 'all'
            re_value, re_reward, re_policy = prepare_multi_target(self.replay_buffer, indices_lst, make_time_lst,
                                                                  game_lst, game_pos_lst, self.config,
                                                                  self.target_model)
            batch[3].append(re_reward)
            batch[4].append(re_value)
            batch[5].append(re_policy)
        for i in range(len(batch)):
            if i in range(3, 6):
                batch[i] = np.concatenate(batch[i])
            else:
                batch[i] = np.asarray(batch[i])

        return batch

def test_mcts(config, summary_writer=None, model_path=None):
    model = config.get_uniform_network()
    target_model = config.get_uniform_network()
    latest_model = config.get_uniform_network()
    if model_path:
        print('resume model from path: ', model_path)
        weights = torch.load(model_path)

        model.load_state_dict(weights)
        target_model.load_state_dict(weights)
        latest_model.load_state_dict(weights)

    storage = SharedStorage.remote(model, target_model, latest_model)

    batch_storage = BatchStorage(8, 16)

    replay_buffer = ReplayBuffer.remote(replay_buffer_id=0, config=config)

    batch_workers = [BatchWorker.remote(idx, replay_buffer, storage, batch_storage, config)
                     for idx in range(config.batch_actor)]

    dw=DataWorker(0, config, storage, replay_buffer).run_multi()
    # # self-play
    # workers = [DataWorker.remote(rank, config, storage, replay_buffer) for rank in range(0, config.num_actors)]
    # workers = [worker.run_multi.remote() for worker in workers]
    # # ray.get(replay_buffer.random_init_trajectory.remote(200))

    # # batch maker
    # workers += [batch_worker.run.remote() for batch_worker in batch_workers]
    # # test
    # # workers += [_test.remote(config, storage)]
    # # train
    # final_weights = _train(model, target_model, latest_model, config, storage, replay_buffer, batch_storage, summary_writer)
    # # wait all
    # ray.wait(workers)

    return model, final_weights

def super(config, data_path, summary_writer=None):
    storage = SharedStorage.remote(config.get_uniform_network())
    assert (config.batch_size // config.replay_number * config.replay_number) == config.batch_size
    replay_buffer_lst = [
        ReplayBuffer.remote(num_replays=config.replay_number, config=config, replay_buffer_id=i, make_dataset=True) for
        i in range(config.replay_number)]

    ray.get([replay_buffer.load_files.remote(data_path) for replay_buffer in replay_buffer_lst])

    _train(config, storage, replay_buffer_lst, summary_writer, None)

    return config.get_uniform_network().set_weights(ray.get(storage.get_weights.remote()))


def make_dataset(config, model):
    storage = SharedStorage.remote(config.get_uniform_network())
    storage.set_weights.remote(model.get_weights())
    replay_buffer_lst = [
        ReplayBuffer.remote(num_replays=config.replay_number, config=config, make_dataset=True, replay_buffer_id=i) for
        i in range(config.replay_number)]

    workers = [DataWorker.remote(rank, config, storage, replay_buffer_lst[rank % config.replay_number]) for rank in
               range(0, config.num_actors)]
    workers = [worker.run_multi.remote() for worker in workers]

    ray.wait(workers)

    data_workers = [replay_buffer.save_files.remote() for replay_buffer in replay_buffer_lst]
    ray.get(data_workers)
