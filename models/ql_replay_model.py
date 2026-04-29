import random 
import nasim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from pprint import pprint

class ReplayMemory:
    def __init__(self, capacity, n_dims):
        self.capacity  = capacity
        self.n_dims = n_dims
        self.s_buffer = np.zeros((capacity, *n_dims), dtype = np.float32)
        self.a_buffer = np.zeros((capacity,), dtype = np.int32)
        self.next_s_buffer = np.zeros((capacity, *n_dims), dtype = np.float32)
        self.r_buffer = np.zeros((capacity,), dtype = np.float32)
        self.done_buffer = np.zeros((capacity,), dtype = np.float32)
        self.ptr, self.size = 0, 0
    def store(self, s, a, r, next_s, done):
        self.s_buffer[self.ptr] = s
        self.a_buffer[self.ptr] = a
        self.next_s_buffer[self.ptr] = next_s
        self.r_buffer[self.ptr] = r
        self.done_buffer[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample_batch(self, batch_size):
        sample_idxs = np.random.choice(self.size, batch_size)
        batch = [
            self.s_buffer[sample_idxs],
            self.a_buffer[sample_idxs],
            self.next_s_buffer[sample_idxs],
            self.r_buffer[sample_idxs],
            self.done_buffer[sample_idxs]
        ]  
        return batch
        
class TabularQLearning:
    def __init__(self, num_actions):
        self.q_func = dict()
        self.num_actions = num_actions

    def __call__(self, s):
        return self.forward(s)
    
    def forward(self, s):
        key = str(s.astype(int)) if isinstance(s, np.ndarray) else s

        if key not in self.q_func:
            self.q_func[key] = np.zeros(self.num_actions, dtype = np.float32)
        return self.q_func[key]
        
    def forward_batch(self, s_batch):
        return np.asarray([self.forward(s) for s in s_batch])
    
    def update(self, s_batch, a_batch, q_val_batch):
        for s, a, q_val in zip(s_batch, a_batch, q_val_batch):
            q_state = self.forward(s)
            q_state[a] += q_val

    
    def display(self):
        pprint(self.q_func)
    
    def get_action(self, s):
        return int(self.forward(s).argmax())
    
class QLearningAgent():
    def __init__(self,
                 env,
                 seed = 42,
                 lr = 0.001,
                 training_steps = 10000,
                 batch_size = 32,
                 replay_size = 1000,
                 final_epsilon = 0.05,
                 exploration_steps = 10000,
                 gamma = 0.9,
                 verbose = True,
                 **kwargs):
        assert env.flat_actions # flat action is 1d array of action space, which is required for tabular Q-learning
        self.verbose = verbose
        if self.verbose:
            print("\nRunning Tabular Q-Learning with config:")
            pprint(locals())
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
        self.env = env
        self.lr = lr
        self.training_steps = training_steps
        self.final_epsilon = final_epsilon
        self.exploration_steps = exploration_steps
        self.discount = gamma
        self.num_actions = self.env.action_space.n
        self.q_func = TabularQLearning(num_actions = self.num_actions)
        self.logger = SummaryWriter()

        self.steps_done = 0 # Training follows step, not episode

        self.epsilon_schedule = np.linspace(1.0, self.final_epsilon, self.exploration_steps)
        self.batch_size = batch_size
    
        self.replay = ReplayMemory(capacity = replay_size, n_dims = self.env.observation_space.shape)
    def get_epsilon(self):
        """ Get epsilon for current step, following linear decay schedule"""
        if self.steps_done < self.exploration_steps:
            return self.epsilon_schedule[self.steps_done]
        else:
            return self.final_epsilon
    
    def get_greedy_action(self, s, epsilon = None):
        """ Using greedy function to get action"""
        if random.random() > epsilon:
            return self.q_func.get_action(s)
        else:
            return random.randint(0, self.num_actions - 1)
        
    def optimize(self):
        """ Update Q-function using Bellman equation"""
        batch = self.replay.sample_batch(self.batch_size)
        s_batch, a_batch, next_s_batch, r_batch, done_batch = batch
        
        # get q_vals for each state and the action performed in that state
        q_vals_raw = self.q_func.forward_batch(s_batch)
        q_vals = np.take_along_axis(q_vals_raw, a_batch[:, None], axis = 1).squeeze()

        # get target q val = max val of next state 
        q_vals_raw_next = self.q_func.forward_batch(next_s_batch)
        q_vals_next = q_vals_raw_next.max(axis = 1)

        target = r_batch + self.discount *(1 - done_batch) * q_vals_next
        td_error = target - q_vals
        td_delta = self.lr * td_error

        self.q_func.update(s_batch, a_batch, td_delta)
        q_vals_max = q_vals_raw.max(axis = 1)
        mean_val = q_vals_max.mean().item()
        mean_td_error = np.abs(td_error).mean().item()
        return mean_td_error, mean_val

    
    def __call__(self):
        return self.train()
    
    def train(self):
        if self.verbose:
            print("\nStart training...")
        
        num_episodes = 0

        training_step_remaining = self.training_steps
        while self.steps_done < self.training_steps:
            ep_reward, ep_steps, goal = self.run_train_episode(training_step_remaining)

            num_episodes += 1
            training_step_remaining -= ep_steps
            self.logger.add_scalar("episode", num_episodes, self.steps_done)
            self.logger.add_scalar(
                "epsilon", self.get_epsilon(), self.steps_done
            )
            self.logger.add_scalar(
                "episode_return", ep_reward, self.steps_done
            )
            self.logger.add_scalar(
                "episode_steps", ep_steps, self.steps_done
            )
            self.logger.add_scalar(
                "episode_goal_reached", int(goal), self.steps_done
            )


    def run_train_episode(self, step_remaining):
        done = False
        env_step_limit_reached = 0

        step = 0
        ep_reward = 0

        s, _ = self.env.reset()

        while not done and not env_step_limit_reached and step < step_remaining:
            epsilon = self.get_epsilon()
            a = self.get_greedy_action(s, epsilon)
            self.steps_done += 1
            next_s, r, done, env_step_limit_reached, _ = self.env.step(a)
            self.replay.store(s, a, r, next_s, done)
            if self.replay.size >= self.batch_size:
                mean_td_error, mean_s =self.optimize()
            else:
                mean_td_error, mean_s = 0, 0
            self.logger.add_scalar("train/mean_td_error", mean_td_error, self.steps_done)
            self.logger.add_scalar("train/s_value", mean_s, self.steps_done)
            s = next_s
            step += 1
            ep_reward += r
        return ep_reward, step, self.env.goal_reached()
    
    def run_eval_episode(self, 
                         env = None,
                         render = False,
                         eval_epsilon = 0.05,
                         render_mode = "human"):
        if env is None:
            env = self.env
        original_render_mode = env.render_mode

        env.render_mode = render_mode

        s, _ = env.reset()
        done = False
        env_step_limit_reached = False

        steps = 0
        episode_return = 0

        line_break = "=" *30
        if render:
            print("\n" + line_break)
            print(f"Running EVALUATION using epsilon = {eval_epsilon:.4f}")
            print(line_break)
            env.render()
            input("Initial state. Press enter to continue..")

        while not done and not env_step_limit_reached:
            a = self.get_greedy_action(s, eval_epsilon)
            next_s, r, done, env_step_limit_reached, _ = env.step(a)
            episode_return += r
            s = next_s
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

    


def main():
    print("Hello from rl-projects!")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, help="benchmark scenario name")
    parser.add_argument("--render_eval", action="store_true",
                        help="Renders final policy")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default=0.001)")
    parser.add_argument("-t", "--training_steps", type=int, default=10000,
                        help="training steps (default=10000)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="(default=32)")
    parser.add_argument("--seed", type=int, default=0,
                        help="(default=0)")
    parser.add_argument("--replay_size", type=int, default=100000,
                        help="(default=100000)")
    parser.add_argument("--final_epsilon", type=float, default=0.05,
                        help="(default=0.05)")
    parser.add_argument("--init_epsilon", type=float, default=1.0,
                        help="(default=1.0)")
    parser.add_argument("-e", "--exploration_steps", type=int, default=10000,
                        help="(default=10000)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="(default=0.99)")
    parser.add_argument("--quite", action="store_false",
                        help="Run in Quite mode")
    args = parser.parse_args()

    env = nasim.make_benchmark(args.env_name, 
                               seed = args.seed,
                               fully_obs = True,
                               flat_obs = True)
    agent = QLearningAgent(env,
                            verbose = args.quite,
                            seed = args.seed,
                            lr = args.lr,
                            training_steps = args.training_steps,
                            final_epsilon = args.final_epsilon,
                            exploration_steps = args.exploration_steps,
                            gamma = args.gamma
                            )
    
    agent.train()
    print("q_table after training.")
    agent.q_func.display()
    agent.run_eval_episode(render = args.render_eval)


if __name__ == "__main__":
    main()
        