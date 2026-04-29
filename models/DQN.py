import random
import numpy as np
import nasim
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch
import torch.optim  as optim
import torch.nn.functional as F 
from pprint import pprint


class ReplayBuffer:
    def __init__(self, capacity, n_dims, device="cpu"):
        self.capacity = capacity
        self.n_dims = n_dims
        self.device = device
        self.s_buff = np.zeros((capacity, *n_dims), dtype=np.float32)
        self.a_buff = np.zeros((capacity, 1), dtype=np.int64)
        self.next_s_buff = np.zeros((capacity, *n_dims), dtype =np.float32)
        self.r_buff = np.zeros(capacity , dtype = np.float32)
        self.done = np.zeros(capacity , dtype = np.float32)
        self.ptr = 0
        self.size = 0

    def store(self, s, a, next_s, r, done):
        self.s_buff[self.ptr] = s
        self.a_buff[self.ptr] = a
        self.next_s_buff[self.ptr] = next_s
        self.r_buff[self.ptr] = r
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    
    def sample_batch(self, batch_size):
        idxs = np.random.choice(self.size, batch_size)
        batch = [
            self.s_buff[idxs],
            self.a_buff[idxs],
            self.next_s_buff[idxs],
            self.r_buff[idxs],
            self.done[idxs]
        ]
        return [torch.tensor(x, device = self.device) for x in batch]
    
class DQN(nn.Module):
    def __init__(self, input_dim, layers, num_actions):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim[0], layers[0])])
        for l in range(1, len(layers)):
            self.layers.append(nn.Linear(layers[l-1], layers[l]))
        self.out = nn.Linear(layers[-1], num_actions)
    
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.out(x)
    
    def save_DQN(self, path):
        torch.save(self.state_dict(), path)

    def load_DQN(self, path):
        self.load_state_dict(torch.load(path))

    def get_action(self, x):
        with torch.no_grad():
            if len(x.shape) == 1:
                x = x.view(1, -1)
            return self.forward(x).max(1)[1]

class DQNAgent:
    def __init__(self, 
                 env,
                 seed = 0,
                 lr = 1e-3,
                 training_steps = 100000,
                 replay_size = 10000,
                 final_epsilon = 0.05,
                 batch_size = 64,
                 exploration_steps = 10000,
                 gamma = 0.99,
                 hidden_sizes = [64, 64],
                 target_update_freq = 1000,
                 verbose = False,
                 **kwargs
                ):
        assert env.flat_actions
        self.verbose = verbose
        if self.verbose:
            print("Initializing DQN Agent...")
            pprint(locals())
        self.env = env
        if seed is not None:
            
            np.random.seed(seed)
        #parameters for training
        self.lr = lr
        self.num_actions = self.env.action_space.n 
        self.obs_dim = self.env.observation_space.shape

        self.exploration_steps = exploration_steps
        self.final_epsilon = final_epsilon
        self.epsilon_schedule = np.linspace(1.0, self.final_epsilon, self.exploration_steps)
        
        self.batch_size = batch_size
        self.training_steps = training_steps
        self.discount = gamma
        self.steps_done = 0
        
        #Neural network and optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else  "cpu")
        
        self.dqn = DQN(input_dim = self.obs_dim, 
                       layers = hidden_sizes, 
                       num_actions= self.num_actions).to(self.device)

        if self.verbose:
            print("DQN Architecture:")
            print(self.dqn)
        self.target_dqn = DQN(input_dim = self.obs_dim, 
                                layers = hidden_sizes, 
                                num_actions= self.num_actions).to(self.device)
        self.target_update_freq = target_update_freq
        self.optimizer = optim.Adam(self.dqn.parameters(), lr = self.lr)
        self.loss_fn = nn.SmoothL1Loss()

        #setup replay buffer

        self.replay = ReplayBuffer(replay_size, self.obs_dim, self.device)

        self.logger = SummaryWriter()
        
    
    def get_epsilon(self):
        if self.steps_done < self.exploration_steps:
            return self.epsilon_schedule[self.steps_done]
        else:
            return self.final_epsilon

    def get_greedy_action(self, o, epsilon):
        if random.random() > epsilon:
            o = torch.from_numpy(o).float().to(self.device)
            return self.dqn.get_action(o).cpu().item()
        return random.randint(0, self.num_actions-1)
        
    def optimize(self):
        batch = self.replay.sample_batch(self.batch_size)
        states, actions, next_states, rewards, done = batch

        q_vals_raw = self.dqn(states)
        q_val = q_vals_raw.gather(1, actions).squeeze()

        # get max q value for next states from target network
        with torch.no_grad():
            target_q_vals_raw = self.target_dqn(next_states)
            target_q_val = target_q_vals_raw.max(1)[0]
            target = rewards + self.discount * (1-done) * target_q_val
        
        td_error = self.loss_fn(q_val, target)
        # optimize the model
        self.optimizer.zero_grad()
        td_error.backward()
        self.optimizer.step()

        if (self.steps_done % self.target_update_freq) == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())
        q_vals_max = q_vals_raw.max(1)[0]
        mean_v = q_vals_max.mean().item()
        return td_error.item(), mean_v

    def train(self):
        if self.verbose:
            print("Starting training...")
        num_episodes = 0
        training_steps_remaining = self.training_steps

        while self.steps_done < self.training_steps:
            ep_result = self.run_train_episode(training_steps_remaining)
            episode_reward, ep_step, done, goal_reached = ep_result
            training_steps_remaining -= ep_step
            num_episodes += 1
            self.logger.add_scalar("Epsilon", self.get_epsilon(), self.steps_done)
            self.logger.add_scalar("Episode Reward", episode_reward, self.steps_done)   
            self.logger.add_scalar("Episode Length", ep_step, self.steps_done)
            self.logger.add_scalar("Episode Goal Reached", done, self.steps_done)
            
            if num_episodes % 10 == 0 and self.verbose:
                print(f"\nEpisode {num_episodes}:")
                print(f"\tsteps done = {self.steps_done} / "
                      f"{self.training_steps}")
                print(f"\treturn = {episode_reward}")
                print(f"\tgoal = {goal_reached}")

        self.logger.close()
        if self.verbose:
            print("Training complete")
            print(f"\nEpisode {num_episodes}:")
            print(f"\tsteps done = {self.steps_done} / {self.training_steps}")
            print(f"\treturn = {episode_reward}")
            print(f"\tgoal = {goal_reached}")


            
    def run_train_episode(self, limit_steps):
        done = False
        env_limit_reached = 0
        state, _ = self.env.reset()

        episode_reward = 0
        step = 0
        while not done and not env_limit_reached and step < limit_steps:
            epsilon = self.get_epsilon()
            action = self.get_greedy_action(state, epsilon)

            next_state, reward, done, env_limit_reached, _= self.env.step(action)
            self.replay.store(state, action, next_state, reward, done)
            self.steps_done += 1
            loss, mean_v = self.optimize()
            self.logger.add_scalar("Loss", loss, self.steps_done)
            self.logger.add_scalar("Mean_V", mean_v, self.steps_done)
            state = next_state
            episode_reward += reward 
            step += 1
            
        
        return episode_reward, step, done, self.env.goal_reached()

    def run_eval_episode(self,
                         env=None,
                         render=False,
                         eval_epsilon=0.05,
                         render_mode="human"):
        if env is None:
            env = self.env

        original_render_mode = env.render_mode
        env.render_mode = render_mode

        o, _ = env.reset()
        done = False
        env_step_limit_reached = False

        steps = 0
        episode_return = 0

        line_break = "="*60
        if render:
            print("\n" + line_break)
            print(f"Running EVALUATION using epsilon = {eval_epsilon:.4f}")
            print(line_break)
            env.render()
            input("Initial state. Press enter to continue..")

        while not done and not env_step_limit_reached:
            a = self.get_greedy_action(o, eval_epsilon)
            next_o, r, done, env_step_limit_reached, _ = env.step(a)
            o = next_o
            episode_return += r
            steps += 1
            if render:
                print("\n" + line_break)
                print(f"Step {steps}")
                print(line_break)
                print(f"Action Performed = {env.action_space.get_action(a)}")
                env.render()
                print(f"Reward = {r}")
                print(f"Done = {done}")
                print(f"Step limit reached = {env_step_limit_reached}")
                input("Press enter to continue..")

                if done or env_step_limit_reached:
                    print("\n" + line_break)
                    print("EPISODE FINISHED")
                    print(line_break)
                    print(f"Goal reached = {env.goal_reached()}")
                    print(f"Total steps = {steps}")
                    print(f"Total reward = {episode_return}")

        env.render_mode = original_render_mode
        return episode_return, steps, env.goal_reached()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, help="benchmark scenario name")
    parser.add_argument("--render_eval", action="store_true",
                        help="Renders final policy")
    parser.add_argument("-o", "--partially_obs", action="store_true",
                        help="Partially Observable Mode")
    parser.add_argument("--hidden_sizes", type=int, nargs="*",
                        default=[64, 64],
                        help="(default=[64. 64])")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default=0.001)")
    parser.add_argument("-t", "--training_steps", type=int, default=20000,
                        help="training steps (default=20000)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="(default=32)")
    parser.add_argument("--target_update_freq", type=int, default=1000,
                        help="(default=1000)")
    parser.add_argument("--seed", type=int, default=0,
                        help="(default=0)")
    parser.add_argument("--replay_size", type=int, default=100000,
                        help="(default=100000)")
    parser.add_argument("--final_epsilon", type=float, default=0.05,
                        help="(default=0.05)")
    parser.add_argument("--init_epsilon", type=float, default=1.0,
                        help="(default=1.0)")
    parser.add_argument("--exploration_steps", type=int, default=10000,
                        help="(default=10000)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="(default=0.99)")
    parser.add_argument("--quite", action="store_false",
                        help="Run in Quite mode")
    args = parser.parse_args()

    env = nasim.make_benchmark(args.env_name,
                               args.seed,
                               fully_obs=not args.partially_obs,
                               flat_actions=True,
                               flat_obs=True)
    dqn_agent = DQNAgent(env, verbose=args.quite, **vars(args))
    dqn_agent.train()
    dqn_agent.run_eval_episode(render=args.render_eval)