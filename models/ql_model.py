import numpy as np
import random
from pprint import pprint
import nasim
from torch.utils.tensorboard import SummaryWriter


class TaubularQfunction:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.q_func = dict()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # convert state to a hashable key and lazily init if missing
        key = str(x.astype(int)) if isinstance(x, np.ndarray) else x
        if key not in self.q_func:
            self.q_func[key] = np.zeros(self.num_actions, dtype=np.float32)
        return self.q_func[key]

    def forward_batch(self, x_batch):
        return np.asarray([self.forward(x) for x in x_batch])

    def update(self, s, a, q_val):
        q_state = self.forward(s)
        q_state[a] += q_val

    def get_action(self, s):
        return int(self.forward(s).argmax())

    def display(self):
        pprint(self.q_func)


class QLearningAgent:
    def __init__(
        self,
        env,
        seed=42,
        lr=0.001,
        training_steps=10000,
        final_epsilon=0.05,
        exploration_steps=10000,
        gamma=0.09,
        verbose=True,
        **kwargs,
    ):
        assert env.flat_actions
        self.verbose = verbose
        if self.verbose:
            print("\nRunning Tabular Q-Learning with config:")
            pprint(locals())

        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

        # environment setups
        self.env = env
        self.num_actions = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape

        # logger setups
        self.logger = SummaryWriter()

        self.lr = lr
        self.exploration_steps = exploration_steps
        self.final_epsilon = final_epsilon
        self.epsilon_schedule = np.linspace(
            1.0, self.final_epsilon, self.exploration_steps
        )
        self.discount = gamma
        self.training_steps = training_steps
        self.steps_done = 0

        # Tabular Q_func
        #
        self.qfunc = TaubularQfunction(self.num_actions)
    def __call__(self):
        return self.train()

    def get_epsilon(self):
        if self.steps_done < self.exploration_steps:
            return self.epsilon_schedule[self.steps_done]
        return self.final_epsilon

    def get_greedy_action(self, s, epsilon):
        if random.random() > epsilon:
            return self.qfunc.get_action(s)
        else:
            return random.randint(0, self.num_actions - 1)

    def optimize(self, s, a, next_s, r, done):
        # get q_val for state and action performed in that state
        q_val_raw = self.qfunc.forward(s)
        q_val = q_val_raw[a]

        # get target q val = max val of next state
        q_val_ns = self.qfunc.forward(next_s).max()
        td_target = r + self.discount * (1 - done) * q_val_ns

        # Calculate error and update
        td_error = td_target - q_val
        td_delta = self.lr * td_error

        self.qfunc.update(s, a, td_delta)

        s_value = q_val_raw.max()
        return td_error, s_value
    
    def train(self):
        if self.verbose:
            print("\nStart training...")

        num_episodes = 0
        training_step_remaining = self.training_steps

        while self.steps_done < self.training_steps:
            ep_results = self.run_train_episode(training_step_remaining)
            ep_reward, ep_steps , goal = ep_results

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


        
    def run_train_episode(self, limit_step):
        done = False
        env_step_limit_reached = 0

        step = 0
        ep_reward = 0

        s, _ = self.env.reset()

        while not done and not env_step_limit_reached and step < limit_step:
            a  = self.get_greedy_action(s, self.get_epsilon())
            self.steps_done += 1

            # get feedback from environment
            next_s, r, done, env_step_limit_reached, _ = self.env.step(a)

            td_error, s_value = self.optimize(s, a, next_s, r, done)
            self.logger.add_scalar("train/td_error", td_error, self.steps_done)
            self.logger.add_scalar("train/s_value", s_value, self.steps_done)
            s = next_s
            ep_reward += r
            step += 1
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
    agent.qfunc.display()
    agent.run_eval_episode(render = args.render_eval)


if __name__ == "__main__":
    main()
