import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
import torch.distributions as distributions 

import numpy as np
import gymnasium as gym 

# ---------------------------------------------------------
# 1. KHỞI TẠO KIẾN TRÚC MẠNG NEURAL (Giống hệt PPO để dễ so sánh)
# ---------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor 
        self.critic = critic 
    
    def forward(self, x):
        action_logits = self.actor(x)
        value = self.critic(x)
        return action_logits, value 

def init_weight(m):
    if type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)

# ---------------------------------------------------------
# 2. HÀM THU THẬP DỮ LIỆU (Gom đủ một Batch lớn rồi mới dừng)
# ---------------------------------------------------------
def collect_trajectories(env, policy, min_steps, gamma):
    policy.eval()
    states, actions, returns = [], [], []
    episode_rewards = []
    
    steps_collected = 0
    
    # Lặp cho tới khi gom đủ số steps tối thiểu (VD: 2048 steps)
    while steps_collected < min_steps:
        ep_states, ep_actions, ep_rewards = [], [], []
        state, _ = env.reset()
        done, truncated = False, False
        
        while not done and not truncated:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_logits, _ = policy(state_tensor)
            
            dist = distributions.Categorical(logits=action_logits)
            action = dist.sample()
            
            state, reward, done, truncated, _ = env.step(action.item())
            
            ep_states.append(state_tensor)
            ep_actions.append(action)
            ep_rewards.append(reward)
            
            steps_collected += 1
            
        episode_rewards.append(sum(ep_rewards))
        
        # Tính Monte Carlo Return cho TỪNG episode độc lập
        R = 0
        ep_returns = []
        for r in reversed(ep_rewards):
            R = r + gamma * R
            ep_returns.insert(0, torch.tensor([R], dtype=torch.float32))
            
        # Đưa vào mảng dữ liệu tổng
        states.extend(ep_states)
        actions.extend(ep_actions)
        returns.extend(ep_returns)
        
    return states, actions, returns, episode_rewards

# ---------------------------------------------------------
# 3. THUẬT TOÁN AWR (Advantage-Weighted Regression)
# ---------------------------------------------------------
def train_awr(policy, optimizer, states, actions, returns, beta, weight_clip, epochs):
    policy.train()
    
    # Chuyển list thành tensor
    states = torch.cat(states)
    actions = torch.cat(actions)
    returns = torch.cat(returns)
    
    # Chuẩn hóa returns giúp mạng Critic học ổn định hơn
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    total_actor_loss = 0
    total_critic_loss = 0
    
    for _ in range(epochs):
        action_logits, values = policy(states)
        values = values.squeeze(-1)
        
        # Bước 1: Tính toán Trọng số AWR (Weights)
        with torch.no_grad(): # CHÚ Ý: Chặn gradient qua Value function khi làm trọng số
            advantages = returns - values
            weights = torch.exp(advantages / beta)
            weights = torch.clamp(weights, max=weight_clip) # Chống bùng nổ gradient
            
        # Bước 2: Tính Actor Loss (Weighted NLL)
        # Thay vì maximize Prob như PPO, AWR đóng vai trò như một bài toán Supervised Learning có trọng số
        dist = distributions.Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        actor_loss = - (weights * log_probs).mean()
        
        # Bước 3: Tính Critic Loss (MSE Loss)
        critic_loss = F.mse_loss(values, returns)
        
        # Tổng hợp và cập nhật gradient
        loss = actor_loss + 0.5 * critic_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_actor_loss += actor_loss.item()
        total_critic_loss += critic_loss.item()
        
    return total_actor_loss / epochs, total_critic_loss / epochs


# ---------------------------------------------------------
# 4. CHẠY VÀ KIỂM THỬ (MAIN)
# ---------------------------------------------------------
if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    INPUT_DIM = env.observation_space.shape[0]
    HIDDEN_DIM = 64
    OUTPUT_DIM = env.action_space.n 

    actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    critic = MLP(INPUT_DIM, HIDDEN_DIM, 1)

    policy = ActorCritic(actor, critic)
    policy.apply(init_weight)
    
    # Khởi tạo trọng số layer cuối nhỏ lại để model có sự khám phá ở đầu tốt hơn
    policy.actor.net[-1].weight.data *= 0.01

    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    # SIÊU THAM SỐ (Hyperparameters)
    MAX_ITERATIONS = 100
    STEPS_PER_ITER = 2048
    DISCOUNT_FACTOR = 0.99
    BETA = 1.0           # Temperature cho AWR (Tham số quan trọng nhất của AWR)
    WEIGHT_CLIP = 20.0   # Chặn trọng số (Weight)
    AWR_EPOCHS = 10

    print("BẮT ĐẦU HUẤN LUYỆN AWR (ADVANTAGE-WEIGHTED REGRESSION)...\n")
    for iteration in range(1, MAX_ITERATIONS + 1):
        
        # 1. Chơi game để thu thập kinh nghiệm (Gom đủ ít nhất 2048 steps)
        states, actions, returns, ep_rewards = collect_trajectories(env, policy, STEPS_PER_ITER, DISCOUNT_FACTOR)
        
        # 2. Học từ kinh nghiệm thu thập được bằng AWR
        actor_loss, critic_loss = train_awr(policy, optimizer, states, actions, returns, BETA, WEIGHT_CLIP, AWR_EPOCHS)
        
        # 3. In kết quả đánh giá
        mean_reward = np.mean(ep_rewards)
        print(f"| Iteration: {iteration:3} | Episodes Done: {len(ep_rewards):2} | Mean Reward: {mean_reward:5.1f} | A_Loss: {actor_loss:5.2f} | C_Loss: {critic_loss:5.2f} |")
            
        if mean_reward >= 500:
            print(f"\\n=> THÀNH CÔNG: Đã đạt ngưỡng 500 điểm tuyệt đối ở vòng lặp thứ {iteration}!")
            break
