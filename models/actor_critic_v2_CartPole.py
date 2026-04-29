import numpy

import torch
import torch.nn.functional as F
import torch.distributions as distributions
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# env
train_env = gym.make("CartPole-v1")
eval_env = gym.make("CartPole-v1")
# seed
SEED = 42
torch.manual_seed(SEED)


# model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        return self.fc2(x)


class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()

        self.actor = actor
        self.critic = critic

    def forward(self, state):
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        return action_pred, value_pred


def train_one_episode(env, policy, discounted_factor, optimizer, device):
    policy.train()
    rewards = []
    values = []
    log_probs_action = []
    entropies = []
    episode_reward = 0
    done = False
    truncated = False

    state, _ = env.reset()

    while not done and not truncated:
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        action_pred, value_pred = policy(state)

        action_prob = F.softmax(action_pred, dim=-1)

        dist = distributions.Categorical(action_prob)

        action = dist.sample()
        log_prob_action = dist.log_prob(action)
        entropy = dist.entropy()

        state, reward, done, truncated, _ = env.step(action.item())
        # save
        rewards.append(reward)
        log_probs_action.append(log_prob_action)
        values.append(value_pred.squeeze(0))  # output se la tensor[1, 1]
        entropies.append(entropy)
        episode_reward += reward
    # chuyen tu list cac tensor sang [tensor]
    log_probs_action = torch.cat(log_probs_action)
    values = torch.cat(values)
    entropies = torch.cat(entropies)

    returns = calculate_returns(rewards, discounted_factor, device)
    advantages = calculate_advantages(returns, values)
    loss = update_policy(
        advantages, log_probs_action, returns, values, entropies, optimizer
    )
    return loss, episode_reward


def calculate_returns(rewards, discounted_factor, device):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + discounted_factor * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)

    return (returns - returns.mean()) / returns.std()


def calculate_advantages(returns, values):
    advantages = returns - values
    return (advantages - advantages.mean()) / advantages.std()


def update_policy(advantages, log_probs_action, returns, values, entropies, optimizer):
    returns = returns.detach()
    advantages = advantages.detach()

    policy_loss = -(advantages * log_probs_action).sum()
    value_loss = F.smooth_l1_loss(returns, values)

    optimizer.zero_grad()

    loss = policy_loss + value_loss * 0.5 - entropies.mean() * 0.01

    loss.backward()

    optimizer.step()
    return loss.item()


def evaluate(env, policy, device):

    policy.eval()

    done = False
    truncated = False
    episode_reward = 0

    state, _ = env.reset()

    while not done and not truncated:

        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():

            action_pred, _ = policy(state)

            action_prob = F.softmax(action_pred, dim=-1)

        action = torch.argmax(action_prob, dim=-1)

        state, reward, done, truncated, _ = env.step(action.item())

        episode_reward += reward

    return episode_reward


if __name__ == "__main__":
    device = torch.device("cpu")
    input_dim = train_env.observation_space.shape[0]
    hidden_dim = 64
    output_dim = train_env.action_space.n

    import argparse
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--actor_lr", type=float, default=1e-3)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--num_episodes", type=int, default=250)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n_runs", type=int, default=5)

    args = parser.parse_args()

    actor_lr = args.actor_lr
    critic_lr = args.critic_lr
    discounted_factor = args.gamma
    max_episodes = args.num_episodes
    n_runs = args.n_runs
    train_rewards = []
    test_rewards = []

    for run in range(n_runs):
        actor = MLP(input_dim, hidden_dim, output_dim)
        critic = MLP(input_dim, hidden_dim, 1)

        policy = ActorCritic(actor, critic)
        policy = policy.to(device)

        optimizer = optim.Adam(
            [
                {"params": actor.parameters(), "lr": actor_lr},
                {"params": critic.parameters(), "lr": critic_lr},
            ]
        )
        for episode in tqdm.tqdm(range(max_episodes), desc=f"Run: {run}"):
            loss, train_rewards = train_one_episode(
                train_env, policy, discounted_factor, optimizer, device
            )
            eval_reward = evaluate(eval_env, policy, device)
