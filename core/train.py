import os
import ray
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import L1Loss
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from .reanalyze_worker import  BatchWorker_CPU,BatchWorker_GPU

from .replay_buffer import ReplayBuffer
from .test import test
import time
import numpy as np
from .test import _test
from .selfplay_worker import DataWorker
from .storage import SharedStorage
from .log import _log
from .storage import BatchStorage
try:
    from apex import amp
except:
    pass
###

def consist_loss_func(f1, f2):
    f1 = F.normalize(f1, p=2., dim=-1, eps=1e-5)
    f2 = F.normalize(f2, p=2., dim=-1, eps=1e-5)
    return -(f1 * f2).sum(dim=1)


def adjust_lr(config, optimizer, step_count, scheduler):
    if step_count < config.lr_warm_step:
        lr = config.lr_init * step_count / config.lr_warm_step
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        if config.lr_type is 'cosine':
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
        else:

            tmp_lr = config.lr_init * config.lr_decay_rate ** ((step_count - config.lr_warm_step) // config.lr_decay_steps)
            if tmp_lr >= 0.0001:
                lr=tmp_lr
            else:
                lr=0.0001
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    return lr

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def update_weights(model, batch, optimizer, replay_buffer, config, scaler, vis_result=False):
    #total_transitions = ray.get(replay_buffer.get_total_len.remote())
    total_transitions=0

    inputs_batch, targets_batch = batch
    obs_batch_ori, action_batch, mask_batch, indices, weights_lst, make_time = inputs_batch
    target_reward, target_value, target_policy = targets_batch

    if config.image_based:
        obs_batch_ori = torch.from_numpy(obs_batch_ori).to(config.device).float() / 255.0
        obs_batch = obs_batch_ori[:, 0: config.stacked_observations * config.image_channel, :, :]
        obs_target_batch = obs_batch_ori[:, config.image_channel:, :, :]
    else:
        obs_batch_ori = torch.from_numpy(obs_batch_ori).to(config.device).float()
        obs_batch = obs_batch_ori[:, 0: config.stacked_observations * config.image_channel, :]
        obs_target_batch = obs_batch_ori[:, config.image_channel:, :]

    if config.use_augmentation:
        # TODO: use different augmentation in target observations respectively
        obs_batch = config.transform(obs_batch)
        obs_target_batch = config.transform(obs_target_batch)

    action_batch = torch.from_numpy(action_batch).to(config.device).unsqueeze(-1).long()
    mask_batch = torch.from_numpy(mask_batch).to(config.device).float()
    target_reward = torch.from_numpy(target_reward).to(config.device).float()
    target_value = torch.from_numpy(target_value).to(config.device).float()
    target_policy = torch.from_numpy(target_policy).to(config.device).float()
    weights = torch.from_numpy(weights_lst).to(config.device).float()

    batch_size = obs_batch.size(0)
    assert batch_size == config.batch_size == target_reward.size(0)
    metric_loss = torch.nn.L1Loss()

    # transform targets to categorical representation
    # Reference:  Appendix F
    other_log = {}
    other_dist = {}

    other_loss = {
        'l1': -1,
        'l1_1': -1,
        'l1_-1': -1,
        'l1_0': -1,
    }
    for i in range(config.num_unroll_steps):
        key = 'unroll_' + str(i + 1) + '_l1'
        other_loss[key] = -1
        other_loss[key + '_1'] = -1
        other_loss[key + '_-1'] = -1
        other_loss[key + '_0'] = -1
    ratio_reward_1 = (target_reward == 1).sum().item() / (batch_size * 5)
    ratio_reward_0 = (target_reward == 0).sum().item() / (batch_size * 5)
    ratio_reward_ne1 = (target_reward == -1).sum().item() / (batch_size * 5)
    other_loss['target_reward_1'] = ratio_reward_1
    other_loss['target_reward_0'] = ratio_reward_0
    other_loss['target_reward_-1'] = ratio_reward_ne1

    transformed_target_reward = config.scalar_transform(target_reward)
    target_reward_phi = config.reward_phi(transformed_target_reward)

    transformed_target_value = config.scalar_transform(target_value)
    target_value_phi = config.value_phi(transformed_target_value)

    with autocast():
        value, _, policy_logits, hidden_state = model.initial_inference(obs_batch.reshape(batch_size, -1))
    scaled_value = config.inverse_value_transform(value)
    if vis_result:
        state_lst = hidden_state.detach().cpu().numpy()

    predicted_rewards = []
    # Note: Following line is just for logging.
    if vis_result:
        predicted_values, predicted_policies = scaled_value.detach().cpu(), torch.softmax(policy_logits, dim=1).detach().cpu()

    # Reference: Appendix G
    value_priority = L1Loss(reduction='none')(scaled_value.squeeze(-1), target_value[:, 0])
    value_priority = value_priority.data.cpu().numpy() + config.prioritized_replay_eps
    reward_priority = []

    value_loss = config.scalar_value_loss(value, target_value_phi[:, 0])
    policy_loss = -(torch.log_softmax(policy_logits, dim=1) * target_policy[:, 0]).sum(1)
    reward_loss = torch.zeros(batch_size, device=config.device)
    consistency_loss = torch.zeros(batch_size, device=config.device)

    target_reward_cpu = target_reward.detach().cpu()
    final_indices = indices > total_transitions * 0.95
    gradient_scale = 1 / config.num_unroll_steps
    with autocast():
        for step_i in range(config.num_unroll_steps):
            value, reward, policy_logits, hidden_state = model.recurrent_inference(hidden_state, action_batch[:, step_i])

            beg_index = config.image_channel * step_i
            end_index = config.image_channel * (step_i + config.stacked_observations)

            if config.consistency_coeff > 0:
                #will not run here
                assert False
                _, _, _, presentation_state = model.initial_inference(obs_target_batch[:, beg_index:end_index, :].reshape(batch_size, -1))
                if config.consist_type is 'contrastive':
                    temp_loss = model.contrastive_loss(hidden_state, presentation_state) * mask_batch[:, step_i]
                else:
                    dynamic_proj = model.project(hidden_state, with_grad=True)
                    observation_proj = model.project(presentation_state, with_grad=False)
                    temp_loss = consist_loss_func(dynamic_proj, observation_proj) * mask_batch[:, step_i]

                other_loss['consist_' + str(step_i + 1)] = temp_loss.mean().item()
                consistency_loss += temp_loss

            policy_loss += -(torch.log_softmax(policy_logits, dim=1) * target_policy[:, step_i + 1]).sum(1)
            value_loss += config.scalar_value_loss(value, target_value_phi[:, step_i + 1])
            reward_loss += config.scalar_reward_loss(reward, target_reward_phi[:, step_i])
            hidden_state.register_hook(lambda grad: grad * 0.5)

            scaled_rewards = config.inverse_reward_transform(reward.detach())

            l1_prior = torch.nn.L1Loss(reduction='none')(scaled_rewards.squeeze(-1), target_reward[:, step_i])
            reward_priority.append(l1_prior.detach().cpu().numpy())
            if vis_result:
                scaled_rewards_cpu = scaled_rewards.detach().cpu()

                predicted_values = torch.cat((predicted_values, config.inverse_value_transform(value).detach().cpu()))
                # scaled_rewards = config.inverse_reward_transform(reward)
                predicted_rewards.append(scaled_rewards_cpu)
                predicted_policies = torch.cat((predicted_policies, torch.softmax(policy_logits, dim=1).detach().cpu()))
                state_lst = np.concatenate((state_lst, hidden_state.detach().cpu().numpy()))

                key = 'unroll_' + str(step_i + 1) + '_l1'

                reward_indices_0 = (target_reward_cpu[:, step_i].unsqueeze(-1) == 0)
                reward_indices_n1 = (target_reward_cpu[:, step_i].unsqueeze(-1) == -1)
                reward_indices_1 = (target_reward_cpu[:, step_i].unsqueeze(-1) == 1)

                target_reward_base = target_reward_cpu[:, step_i].reshape(-1).unsqueeze(-1)

                other_loss[key] = metric_loss(scaled_rewards_cpu, target_reward_base)
                if reward_indices_1.any():
                    other_loss[key + '_1'] = metric_loss(scaled_rewards_cpu[reward_indices_1], target_reward_base[reward_indices_1])
                if reward_indices_n1.any():
                    other_loss[key + '_-1'] = metric_loss(scaled_rewards_cpu[reward_indices_n1], target_reward_base[reward_indices_n1])
                if reward_indices_0.any():
                    other_loss[key + '_0'] = metric_loss(scaled_rewards_cpu[reward_indices_0], target_reward_base[reward_indices_0])

                if final_indices.any():
                    # last 5% data
                    key = '5%_' + key

                    target_reward_cpu_test = target_reward_cpu[final_indices]
                    scaled_rewards_cpu_test = scaled_rewards_cpu[final_indices]
                    target_reward_base_test = target_reward_base[final_indices]

                    reward_indices_0 = (target_reward_cpu_test[:, step_i].unsqueeze(-1) == 0)
                    reward_indices_n1 = (target_reward_cpu_test[:, step_i].unsqueeze(-1) == -1)
                    reward_indices_1 = (target_reward_cpu_test[:, step_i].unsqueeze(-1) == 1)

                    if reward_indices_1.any():
                        other_loss[key + '_1'] = metric_loss(scaled_rewards_cpu_test[reward_indices_1], target_reward_base_test[reward_indices_1])
                    if reward_indices_n1.any():
                        other_loss[key + '_-1'] = metric_loss(scaled_rewards_cpu_test[reward_indices_n1], target_reward_base_test[reward_indices_n1])
                    if reward_indices_0.any():
                        other_loss[key + '_0'] = metric_loss(scaled_rewards_cpu_test[reward_indices_0], target_reward_base_test[reward_indices_0])

    # ----------------------------------------------------------------------------------
    # optimize
    loss = (config.consistency_coeff * consistency_loss + config.policy_loss_coeff * policy_loss + config.value_loss_coeff * value_loss + config.reward_loss_coeff * reward_loss)

    weighted_loss = (weights * loss).mean()

    # L2 reg
    parameters = model.parameters()
    if config.amp_type == 'torch_amp':
        with autocast():
            total_loss = weighted_loss
            total_loss.register_hook(lambda grad: grad * gradient_scale)
    else:
        total_loss = weighted_loss
        total_loss.register_hook(lambda grad: grad * gradient_scale)
    optimizer.zero_grad()

    if config.amp_type == 'nvidia_apex':
        with amp.scale_loss(total_loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    elif config.amp_type == 'none':
        total_loss.backward()
    elif config.amp_type == 'torch_amp':
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)

    torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
    if config.amp_type == 'torch_amp':
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    # ----------------------------------------------------------------------------------
    # update priority
    reward_priority = np.mean(reward_priority, 0)
    new_priority = (1 - config.priority_reward_ratio) * value_priority + config.priority_reward_ratio * reward_priority
    replay_buffer.update_priorities.remote(indices, new_priority, make_time)

    # packing data for logging
    loss_data = (total_loss.item(), weighted_loss.item(), loss.mean().item(), 0, policy_loss.mean().item(),
                 reward_loss.mean().item(), value_loss.mean().item(), consistency_loss.mean())
    if vis_result:
        reward_w_dist, representation_mean, dynamic_mean, reward_mean = model.get_params_mean()
        other_dist['reward_weights_dist'] = reward_w_dist
        other_log['representation_weight'] = representation_mean
        other_log['dynamic_weight'] = dynamic_mean
        other_log['reward_weight'] = reward_mean

        # reward l1 loss
        reward_indices_0 = (target_reward_cpu[:, :config.num_unroll_steps].reshape(-1).unsqueeze(-1) == 0)
        reward_indices_n1 = (target_reward_cpu[:, :config.num_unroll_steps].reshape(-1).unsqueeze(-1) == -1)
        reward_indices_1 = (target_reward_cpu[:, :config.num_unroll_steps].reshape(-1).unsqueeze(-1) == 1)

        target_reward_base = target_reward_cpu[:, :config.num_unroll_steps].reshape(-1).unsqueeze(-1)

        predicted_rewards = torch.stack(predicted_rewards).transpose(1, 0).squeeze(-1)
        if final_indices.any():
            predicted_rewards_test = predicted_rewards[final_indices].reshape(-1).unsqueeze(-1)
        predicted_rewards = predicted_rewards.reshape(-1).unsqueeze(-1)
        other_loss['l1'] = metric_loss(predicted_rewards, target_reward_base)
        if reward_indices_1.any():
            other_loss['l1_1'] = metric_loss(predicted_rewards[reward_indices_1], target_reward_base[reward_indices_1])
        if reward_indices_n1.any():
            other_loss['l1_-1'] = metric_loss(predicted_rewards[reward_indices_n1], target_reward_base[reward_indices_n1])
        if reward_indices_0.any():
            other_loss['l1_0'] = metric_loss(predicted_rewards[reward_indices_0], target_reward_base[reward_indices_0])

        if final_indices.any():
            # last 5% data
            target_reward_cpu_test = target_reward_cpu[final_indices]
            target_reward_base_test = target_reward_cpu[final_indices, :config.num_unroll_steps].reshape(-1).unsqueeze(-1)

            reward_indices_0 = (target_reward_cpu_test[:, :config.num_unroll_steps].reshape(-1).unsqueeze(-1) == 0)
            reward_indices_n1 = (target_reward_cpu_test[:, :config.num_unroll_steps].reshape(-1).unsqueeze(-1) == -1)
            reward_indices_1 = (target_reward_cpu_test[:, :config.num_unroll_steps].reshape(-1).unsqueeze(-1) == 1)

            other_loss['5%_l1'] = metric_loss(predicted_rewards_test, target_reward_base_test)
            if reward_indices_1.any():
                other_loss['5%_l1_1'] = metric_loss(predicted_rewards_test[reward_indices_1], target_reward_base_test[reward_indices_1])
            if reward_indices_n1.any():
                other_loss['5%_l1_-1'] = metric_loss(predicted_rewards_test[reward_indices_n1], target_reward_base_test[reward_indices_n1])
            if reward_indices_0.any():
                other_loss['5%_l1_0'] = metric_loss(predicted_rewards_test[reward_indices_0], target_reward_base_test[reward_indices_0])

        td_data = (new_priority, target_reward.detach().cpu().numpy(), target_value.detach().cpu().numpy(),
                   transformed_target_reward.detach().cpu().numpy(), transformed_target_value.detach().cpu().numpy(),
                   target_reward_phi.detach().cpu().numpy(), target_value_phi.detach().cpu().numpy(),
                   predicted_rewards.detach().cpu().numpy(), predicted_values.detach().cpu().numpy(),
                   target_policy.detach().cpu().numpy(), predicted_policies.detach().cpu().numpy(), state_lst,
                   other_loss, other_log, other_dist)
        priority_data = (weights, indices)
    else:
        td_data, priority_data = _, _

    return loss_data, td_data, priority_data, scaler


def _train(model, target_model, latest_model, config, shared_storage, replay_buffer, batch_storage, summary_writer, snapshot):

    # ----------------------------------------------------------------------------------
    model = model.to(config.device)
    target_model = target_model.to(config.device)

    model.train()
    target_model.eval()
    latest_model.eval()

    optimizer = optim.SGD(model.parameters(), lr=config.lr_init, momentum=config.momentum,
                           weight_decay=config.weight_decay)

    # optimizer = optim.RMSprop(model.parameters(), lr=config.lr_init, momentum=config.momentum,
    #                        weight_decay=config.weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=config.lr_init, momentum=config.momentum,
    #                        weight_decay=config.weight_decay)
    # optimizer = optim.Adam(model.parameters(),lr=config.lr_init,eps=1e-5)


    if config.amp_type == 'nvidia_apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.opt_level)
    scaler = GradScaler()
    # ----------------------------------------------------------------------------------

    if config.lr_type is 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=config.training_steps + config.last_steps - config.lr_warm_step)
    else:
        scheduler = None

    if config.use_augmentation:
        config.set_transforms()

    # wait for all replay buffer to be non-empty
    last=0
    mv=2
    while not (ray.get(replay_buffer.get_total_len.remote()) >= config.start_window_size):
        cur=ray.get(replay_buffer.get_total_len.remote())
        print("waiting in _train,buffer size ={} /{}, speed={:.1f}".format(cur,config.start_window_size,(cur-last)/mv),flush=True)
        last=cur
        time.sleep(mv)
        pass
    print('Begin training...')
    shared_storage.set_start_signal.remote()

    step_count = 0
    lr = 0.

    recent_weights = model.get_weights()
    time_100k=time.time()
    _interval=config.debug_interval

    while step_count < config.training_steps + config.last_steps:
    # while True:
        if step_count % 200 == 0:
            replay_buffer.remove_to_fit.remote()
        if step_count in snapshot:
            print("============>replay buffer start")
            op_dir=os.path.join(config.exp_path, 'replay',str(step_count))
            if not os.path.exists(op_dir):
                os.makedirs(op_dir)
            replay_buffer.save_files.remote(id=step_count)
            op_dir = os.path.join(config.exp_path, 'replay',str(step_count))
            print(op_dir,os.path.exists(op_dir))
            print(os.path.join(op_dir,'op.pt'))
            torch.save(optimizer.state_dict(),os.path.join(op_dir,'op.pt'))
            print("============>replay buffer finish")
        batch = batch_storage.pop()
        if batch is None:
            time.sleep(0.5)
            continue
        shared_storage.incr_counter.remote()
        lr = adjust_lr(config, optimizer, step_count, scheduler)

        if step_count % config.checkpoint_interval == 0:
            shared_storage.set_weights.remote(model.get_weights())


        if step_count % config.target_model_interval == 0:
            shared_storage.set_target_weights.remote(recent_weights)
            recent_weights = model.get_weights()

        if config.use_latest_model:
            soft_update(latest_model, model.detach(), tau=1)
            shared_storage.set_latest_model.remote(latest_model.get_weights())

        if step_count % config.vis_interval == 0:
            vis_result = True
        else:
            vis_result = False

        if config.amp_type == 'torch_amp':
            if step_count >= 1:
                scaler = scaler_prev
            log_data = update_weights(model, batch, optimizer, replay_buffer, config, scaler, True)
            scaler_prev = log_data[3]
        else:
            log_data = update_weights(model, batch, optimizer, replay_buffer, config, scaler, vis_result)

        if step_count % config.log_interval == 0:
            _log(config, step_count, log_data[0:3], model, replay_buffer, lr, shared_storage, summary_writer, vis_result)

        step_count += 1


        if step_count%_interval==0:

            _time=time.time()-time_100k
            print("===>step={} ;cost [{:.2f}] s/{}steps; <==>[{:.2f}] s/1klr".format(step_count,_time,_interval,_time/(_interval/1000)),flush=True)
            time_100k=time.time()

        if step_count % config.save_ckpt_interval == 0:
            model_path = os.path.join(config.model_dir, 'model_{}.p'.format(step_count))
            torch.save(model.state_dict(), model_path)

    shared_storage.set_weights.remote(model.get_weights())
    return model.get_weights()





def train(config, summary_writer=None, model_path=None):
    model = config.get_uniform_network()
    target_model = config.get_uniform_network()
    latest_model = config.get_uniform_network()
    #assert model_path is not None
    if model_path:
        print('resume model from path: ', model_path,flush=True)
        weights = torch.load(model_path)

        model.load_state_dict(weights)
        target_model.load_state_dict(weights)
        latest_model.load_state_dict(weights)

    storage = SharedStorage.remote(model, target_model, latest_model)

    batch_storage = BatchStorage(20, 30,'learn batch')
    mcts_storage = BatchStorage(20, 30,'context batch')

    replay_buffer = ReplayBuffer.remote(replay_buffer_id=0, config=config)


    workers=[]
    # reanalyze workers
    cpu_workers = [BatchWorker_CPU.remote(idx, replay_buffer, storage, batch_storage, mcts_storage, config) for idx in range(config.cpu_actor)]
    workers += [cpu_worker.run.remote() for cpu_worker in cpu_workers]
    gpu_workers = [BatchWorker_GPU.remote(idx, replay_buffer, storage, batch_storage, mcts_storage, config) for idx in range(config.gpu_actor)]
    workers += [gpu_worker.run.remote() for gpu_worker in gpu_workers]


    # self-play
    #num_actors=2
    data_workers = [DataWorker.remote(rank, config, storage, replay_buffer) for rank in range(config.num_actors)] #changed to 1 actor
    workers += [worker.run_multi.remote() for worker in data_workers]

    workers += [_test.remote(config, storage)]
    # train
    snapshot=[]#save snapshot of replay buffer, optimizer at {} training steps
    final_weights = _train(model, target_model, latest_model, config, storage, replay_buffer, batch_storage, summary_writer, snapshot)
    # wait all
    ray.wait(workers)

    return model
