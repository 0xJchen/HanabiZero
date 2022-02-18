# HanabiZero

Solve cooperative imperfect information multi-agent game "Hanabi" with SoTA model-based reinforcement learning methods from scratch through self-play and without human knowledge. Build on top of [EfficientZero](https://github.com/YeWR/EfficientZero)

Motivation: this project can be understood in a broader background: *(In CTDE regime,) How do Model-based RL work in Partially-observable (or moreover, stochastic) environment?*. Model-free methods like `actor-critic` can simply train an oracle `critic` that takes input as the gloval state. While there's no such equivalence in MBRL.

Direcly train with global-state or oracle-regression (we proposed) reaches ~ 23/25. Currently debugging whether there's something wrong with the codebase (in one sence, training on global state should reaches at least SOTA performance, which is at least `24+`). This branch currently contains the code of training with either `global` or `local` states.

## Train

```
python3 ../main.py --env Hanabi-Full --case hanabi --opr train --seed 10 --num_gpus 4 --num_cpus 110 --force\
  --cpu_actor 6 --gpu_actor 16 \
  --p_mcts_num 24\
  --extra 2022 \
  --use_priority \
  --use_max_priority \
  --revisit_policy_search_rate 0.999 \
  --amp_type 'torch_amp' \
  --reanalyze_part 'paper' \
  --info 'decay_n_op' \
  --actors 40 \
  --simulations 50 \
  --batch_size 256 \
  --val_coeff 0.25 \
  --td_step 5 \
  --debug_interval 100 \
  --lr 0.1 \
  --decay_rate 0.1 \
  --decay_step 200000 \
  --stack 1 \
  --optim 'sgd' 
```

Some tweaking parameters:
- compuational budget: `--num_gpus 4 --num_cpus 110`
- reanalyze-bottleneck: `--cpu_actor 6 --gpu_actor 16`
- parallel mcts instance: `p_mcts_num`. Note: increase this may greatly increase the experience collect speed, but as one pass corresponds to one history policy, this may lead to stale experience in the replay buffer. To increase the replay buffer flash speed, plz consider **1. increase actors** and **2. tuning `p_mcts_num`**
- prioritize replay: `use_priority`. Currently prioritizing the latest experience.
- network architecture: using larger model (over-parameterized) `representation, dynamics, prediction` modules lead to faster convergence.
- actors: # of parallel actors to collect experience. Restricted by the GPU memory.
- `gpu_num` in `reanal.py:15`. Currently, `actor` and `worker` share the same amount of gpu determined by `gpu_num`. On `RTX 3090` the most compatible budget is `0.06/card`
- learning rate `lr` and decay `decay_rate`, `decay_step`. First using large lr `0.1`, then gradually decay. In practice, I found it stuck at game score (15/25, known as a policy saddle point also observed in other hanabi algorithms). Decay it by `0.1` gradually lead to imrpoved performance. When capped at `0.0001`, the agent is capable of reaching `23/25`.
- stacked frame `stack`. While tackling the problem of partial observability, stack image requires a larger representation network. When using global regression-like techniques or simply testing with global observation, no stacking image works fine. Note, there are 2 successful hanabi algorithms: `[R2D2](https://github.com/facebookresearch/hanabi_SAD/tree/main/pyhanabi)` uses RNN for state representation while `MAPPO` uses single frame as input state. On the other hand, by default using global state for debugging now. Simly using local state not seems to work here. 
- optimizer `optim`. I found `rmsprop` not working, `sgd` is enough. `adam` may stuck at local optim when squeezing the last performance. Other techniques like `cos annealing, cyclic lr` are possible alternative choices.

Other supported modes (besides `train`) including: 1. load a model then test. 2. save snapshot of `replay buffer` and `optimizer` during training 3. load these snapshots and continue training. The Logging directory can be found automatically with `sh eval.sh`, which takes `info`'s value in the script as input.

On `4*RTX3090`, training on `Hanabi-Small` takes roughly 4 hours to reach `9/10`, and on `Hanabi-Full` takes roughly more than a day to reach `23/25`. The default script takes `~ 160s` for 1k learner steps, with an [replay ratio](https://acsweb.ucsd.edu/~wfedus/pdf/replay.pdf) of `0.008`.  

## environment

- option 1: using docker.
- option 2: install `requirements.txt` manually

- remember to install the requirements for `Hanabi` Env in `./env`. Also, after modifying the environment itself, rebuild it with `cd env/hanabi && rm -rf build && mkdir build && cd build && cmake .. && make `.
- after modifying `core/ctree`, rebuild with `cd core/ctree && sh make.sh`
