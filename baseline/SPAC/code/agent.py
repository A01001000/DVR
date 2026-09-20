import torch
from torchvision import datasets, transforms
import numpy as np
import h5py
import time
import os
from collections import namedtuple

import utils
from config import Config as cfg
from memory import ReplayMemory

Memory = namedtuple('Memory', ['s', 'a', 'r', 's_', 'd'])

class Agent:
    def __init__(self, brain, env, summary_writer, device=None):
        super(Agent, self).__init__()
        self.writer = summary_writer
        self.device = device
        self.brain = brain
        self.env = env

        self.memories = ReplayMemory(cfg.MEMORY_SIZE, look_forward_steps=0)

    def feed_memory(self, s, a, r, s_, d):
        memory = Memory(s, a, r, s_, d)
        return self.memories.store(memory)

    def run(self):
        tic = time.time()
        print('start training!')

        total_step = 1
        global_ep = 0
        update_planner = False
        update_actor = False
        zero_reward_streak = 0
        # buffer_s, buffer_a, buffer_v, buffer_r, buffer_h = [], [], [], [], []
        while global_ep < cfg.MAX_GLOBAL_EP:

            ep_reward = 0
            step = 0
            episode_values = []


            # score_thr = 0.3 if global_ep < 1000 else 0.2

            s, init_score = self.env.reset()
            
            if not np.isfinite(init_score) or init_score == 0:
                print(f"FATAL WARNING at EP {global_ep}: Initial score is {init_score}. The reward function might be failing. Stopping training.")
                break

            # print(s.shape)
            if global_ep % 20 == 0:
                self.env.save_init()

            while True:
                latent, field = self.brain.choose_action(s)
                
                field_mag_min = field.min().item()
                field_mag_max = field.max().item()
                print(f"  Step {step}: Deformation Field Magnitude | Min: {field_mag_min:.4f}, Max: {field_mag_max:.4f}")

                v = self.brain.get_value(s, latent)
                r, s_, done, score = self.env.step(field, global_ep)
                
                print(f"  Step {step}: Score after step: {score:.4f} | Reward: {r:.4f}")
                if r == 0 and score != init_score and global_ep > cfg.EP_BOOTSTRAP:
                    print(f"  - WARNING: Reward is zero but score changed. This might indicate a problem.")
                
                if r == 0:
                    zero_reward_streak += 1
                else:
                    zero_reward_streak = 0 # Reset streak if we get a valid reward

                if zero_reward_streak > 50: # After 50 consecutive zero rewards
                    print(f"🚨 FATAL WARNING at EP {global_ep}, Step {step}: The agent has received a zero reward 50 times in a row.")
                    print("This indicates the model is not learning or the reward function is stuck.")
                    print("It is highly recommended to stop training and investigate.")
                    # You can force it to stop by uncommenting the next line:
                    # global_ep = cfg.MAX_GLOBAL_EP 
                    zero_reward_streak = 0 # Reset to avoid spamming the log
                    
                self.feed_memory(utils.numpy(s, device=self.device),
                                 utils.numpy(latent, device=self.device), r,
                                 utils.numpy(s_, device=self.device), done)

                ep_reward = r if step == 0 else (ep_reward * 0.99 + r * 0.01)

                episode_values.append(v)

                if done:
                    print("EP: ", global_ep,
                          "\tReward = ", ep_reward,
                          "\tSteps = ", step,
                          '\tinit_score = ', init_score,
                          '\tdone_score = ', score,
                          '\tDone!')

                # update planner when critic updated N times
                if total_step % (cfg.FREQUENCY_PLANNER*cfg.UPDATE_GLOBAL_ITER) == 0:
                    update_planner = True
                else:
                    update_planner = False

                if total_step % (cfg.FREQUENCY_VAE*cfg.UPDATE_GLOBAL_ITER) == 0:
                    update_actor = True
                else:
                    update_actor = False

                if global_ep >= cfg.EP_BOOTSTRAP and total_step % cfg.UPDATE_GLOBAL_ITER == 0:
                    # Only update if we have at least some memories
                    loss = None
                    if len(self.memories) > 0:
                        if len(self.memories) < cfg.SAMPLE_BATCH_SIZE:
                            samples = self.memories.sample(len(self.memories))
                        else:
                            samples = self.memories.sample(cfg.SAMPLE_BATCH_SIZE)

                        loss = self.brain.optimize(update_planner, update_actor, samples)

                    if update_actor and global_ep % 10 == 0 and loss is not None:
                        critic = min(loss['critic1'].item(), loss['critic2'].item())
                        reg = loss['reg'].item()

                        print('ep-{}-{:2d}:'.format(global_ep, step),
                              f' -- critic: {critic:.4f}, alpha: {self.brain.alpha:.4f},',
                              f'reg: {reg:.4f}, reward: {r:.3f},',
                              f'score: {score:.3f}, init_score: {init_score:.3f}')


                if step < 31 and global_ep % 20 == 0:
                    self.env.save_process(step)
                s = s_
                total_step += 1
                step += 1

                if done or global_ep % 1000 == 0:
                    self.save_model(global_ep)

                if done or step >= cfg.MAX_EP_STEPS: #or global_ep < cfg.EP_BOOTSTRAP
                    self.writer.add_info(step, episode_values, ep_reward)
                    global_ep += 1
                    if global_ep % 100 == 0 and global_ep > 0:
                        self.save_model(global_ep) 

                    if global_ep < cfg.EP_BOOTSTRAP:
                        print(global_ep, f'{init_score:.4f}',
                            f'{score:.4f}')

                    break

        self.writer.close()
        toc = time.time()
        time_elapse = toc - tic
        h = time_elapse // 3600
        m = time_elapse % 3600 // 60
        s = time_elapse % 3600 % 60
        print('cast time %.0fh %.0fm %.0fs' % (h, m, s))
        print('training finished!')


    def save_model(self, global_ep):
        if global_ep > 500 and global_ep % 100 == 0:
            print(f"------------------------Saving model {global_ep}------------------")
            self.brain.save_model(global_ep, cfg.MODEL_PATH)
            print(f"------------------------Model saved!------------------")



























