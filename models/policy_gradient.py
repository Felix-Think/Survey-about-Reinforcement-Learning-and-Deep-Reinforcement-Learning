import numpy as np
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import torch.distributions as distributions
import cv2
import gymnasium as gym
import tqdm

# Setup random seeds
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


# Model
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.softmax(x)


# init weights
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


# apply weight initialization


EPOCHS = 1000
DISCOUNT = 0.99


def caculate_reward(rewards, discount_factor, device, normalize=True):
    returns = []
    R = 0

    for reward in reversed(rewards):
        R = reward + R * discount_factor
        returns.insert(0, R)
    returns = torch.tensor(returns).to(device)
    if normalize:
        returns = (returns - returns.mean()) / returns.std()

    return returns


def optimize(returns, log_probs_action, optimizer):
    returns = returns.detach()  # xoa ra khoi computational graph

    loss = -(log_probs_action * returns).sum()
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    return loss.item()


device = torch.device("cpu")


def train(env, policy, episodes, discount, optimizer, device):
    episodes_reward = []
    losses = []
    for episode in tqdm.tqdm(range(episodes)):
        policy.train()
        log_probs_action = []
        rewards = []
        done = False
        truncated = False
        episode_reward = 0
        eval_reward = 0
        state, _ = env.reset()

        while not done and not truncated:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action_probs = policy(state)
            dict = distributions.Categorical(action_probs)
            action = dict.sample()
            log_prob_action = dict.log_prob(action)
            state, reward, done, truncated, _ = env.step(action.item())
            log_probs_action.append(log_prob_action)
            rewards.append(reward)
            episode_reward += reward
        eval_reward = evaluate(env, policy, device)

        log_probs_action = torch.cat(log_probs_action)
        rewards = caculate_reward(rewards, discount, device, normalize=True)
        episodes_reward.append(episode_reward)

        loss = optimize(rewards, log_probs_action, optimizer)

        losses.append(loss)
        print(
            f"Episode{episode}| loss: {loss} | episode_reward{episode_reward} | eval_reward{eval_reward}"
        )
    torch.save(policy.state_dict(), "bestmodel.pth")


def evaluate(env, policy, device):
    policy.eval()
    done = False
    truncated = False
    episode_reward = 0
    state, _ = env.reset()

    while not done and not truncated:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            action_pred = policy(state)
        action = torch.argmax(action_pred, dim=-1)
        state, reward, done, truncated, _ = env.step(action.item())
        episode_reward += reward
    return episode_reward


def test(env, policy, episodes, device, render=True):
    policy.eval()
    done = False
    truncated = False
    episode_reward = 0
    frames = []
    for episode in range(episodes):
        done = False
        truncated = False
        episode_reward = 0
        state, _ = env.reset()
        while not done and not truncated:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)

            with torch.no_grad():
                action_pred = policy(state)
            action = torch.argmax(action_pred, dim=-1)
            state, reward, done, truncated, _ = env.step(action.item())
            episode_reward += reward
            if render:
                frame = env.render()
                # Vẽ số episode lên frame
                frame = cv2.putText(
                    frame,
                    f"Episode: {episode}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                frames.append(frame)
        print(f"Ep{episode} | Reward{episode_reward}")
    height, width = frames[0].shape[:2]
    out = cv2.VideoWriter(
        "video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height)
    )

    for frame in frames:
        out.write(frame)  # CV2 cần BGR → cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    out.release()
    print("✅ Video saved!")


# env
env = gym.make("CartPole-v1")
test_env = gym.make("CartPole-v1", render_mode="rgb_array")

# Config
INPUT_DIM = env.observation_space.shape[0]
HIDDEN_DIM = 128
OUTPUT_DIM = env.action_space.n
LEARNING_RATE = 1e-2
DROPOUT = 0.3


policy = PolicyNetwork(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT)
policy.apply(init_weights)
policy = policy.to(device)
policy.load_state_dict(torch.load("bestmodel.pth"))
# optimizer
optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

TRAIN_EPISODES = 30000
TEST_EPISODES = 50
# train(env,policy,episodes=TRAIN_EPISODES,discount=DISCOUNT,optimizer=optimizer,device=device)


test(test_env, policy, episodes=TEST_EPISODES, device=device)
