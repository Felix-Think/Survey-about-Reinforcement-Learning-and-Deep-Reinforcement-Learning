import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
import torch.distributions as distributions 

import numpy as np
import gymnasium as gym 

# ---------------------------------------------------------
# 1. REPLAY BUFFER (Dựa trên code DQN của bạn)
# ---------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity, n_dims, device="cpu"):
        self.capacity = capacity
        self.n_dims = n_dims
        self.device = device
        self.s_buff = np.zeros((capacity, *n_dims), dtype=np.float32)
        # Sửa lại shape (capacity,) để chứa 1 hành động (Discrete) thay vì (capacity, 1)
        self.a_buff = np.zeros((capacity,), dtype=np.int64) 
        self.next_s_buff = np.zeros((capacity, *n_dims), dtype=np.float32)
        self.r_buff = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def store(self, s, a, next_s, r, done):
        self.s_buff[self.ptr] = s
        self.a_buff[self.ptr] = a
        self.next_s_buff[self.ptr] = next_s
        self.r_buff[self.ptr] = r
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample_batch(self, batch_size):
        idxs = np.random.choice(self.size, batch_size)
        batch = [
            self.s_buff[idxs],
            self.a_buff[idxs],
            self.next_s_buff[idxs],
            self.r_buff[idxs],
            self.done[idxs]
        ]
        return [torch.tensor(x, device=self.device) for x in batch]

# ---------------------------------------------------------
# 2. KIẾN TRÚC MẠNG NEURAL (Giữ nguyên)
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
        return self.actor(x), self.critic(x)

def init_weight(m):
    if type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)

# Cập nhật từ từ (Soft update) cho Target Network giống thuật toán SAC/DDPG
def soft_update(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

# ---------------------------------------------------------
# 3. THUẬT TOÁN AWR OFF-POLICY (Sử dụng Buffer)
# ---------------------------------------------------------
def train_awr_off_policy(policy, target_policy, optimizer, buffer, batch_size, gamma, beta, weight_clip):
    if buffer.size < batch_size:
        return 0.0, 0.0

    policy.train()
    
    # 1. Rút ngẫu nhiên từ Replay Buffer (Off-policy)
    states, actions, next_states, rewards, dones = buffer.sample_batch(batch_size)
    
    action_logits, values = policy(states)
    values = values.squeeze(-1)
    
    # 2. Tính 1-step TD Target (Return)
    # Vì lấy data lẻ tẻ, ta dùng Target Network để dự đoán tương lai giống hệt DQN
    with torch.no_grad():
        _, next_values = target_policy(next_states)
        next_values = next_values.squeeze(-1)
        # Công thức TD(0): Return = reward + gamma * V(s') * (1 - done)
        returns = rewards + gamma * next_values * (1.0 - dones)
        
        # 3. Tính Advantage và Trọng số AWR
        advantages = returns - values
        weights = torch.exp(advantages / beta)
        weights = torch.clamp(weights, max=weight_clip)
        
    # Chuẩn hóa returns giúp Critic học mượt hơn
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
    # 4. Tính Actor Loss (Weighted NLL)
    dist = distributions.Categorical(logits=action_logits)
    log_probs = dist.log_prob(actions)
    actor_loss = - (weights * log_probs).mean()
    
    # 5. Tính Critic Loss (MSE)
    critic_loss = F.mse_loss(values, returns)
    
    loss = actor_loss + 0.5 * critic_loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return actor_loss.item(), critic_loss.item()

# ---------------------------------------------------------
# 4. CHẠY VÀ KIỂM THỬ
# ---------------------------------------------------------
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    INPUT_DIM = env.observation_space.shape
    HIDDEN_DIM = 64
    OUTPUT_DIM = env.action_space.n 

    actor = MLP(INPUT_DIM[0], HIDDEN_DIM, OUTPUT_DIM)
    critic = MLP(INPUT_DIM[0], HIDDEN_DIM, 1)
    policy = ActorCritic(actor, critic)
    policy.apply(init_weight)
    policy.actor.net[-1].weight.data *= 0.01

    # Tạo Target Policy Network để dự đoán V(s') ổn định (Giống hệt cách DQN làm)
    target_actor = MLP(INPUT_DIM[0], HIDDEN_DIM, OUTPUT_DIM)
    target_critic = MLP(INPUT_DIM[0], HIDDEN_DIM, 1)
    target_policy = ActorCritic(target_actor, target_critic)
    target_policy.load_state_dict(policy.state_dict())
    target_policy.eval()

    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    
    # SIÊU THAM SỐ
    BUFFER_CAPACITY = 50000
    BATCH_SIZE = 256
    GAMMA = 0.99
    BETA = 1.0           
    WEIGHT_CLIP = 20.0   
    TAU = 0.005 # Tốc độ cập nhật Target Network
    MAX_EPISODES = 500

    buffer = ReplayBuffer(BUFFER_CAPACITY, INPUT_DIM)
    ep_rewards_history = []

    print("BẮT ĐẦU HUẤN LUYỆN AWR (OFF-POLICY - DÙNG REPLAY BUFFER)...\n")
    for episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        done, truncated = False, False
        ep_reward = 0
        
        while not done and not truncated:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Khám phá môi trường dựa trên phân phối xác suất
            with torch.no_grad():
                action_logits, _ = policy(state_tensor)
            dist = distributions.Categorical(logits=action_logits)
            action = dist.sample().item()
            
            next_state, reward, done, truncated, _ = env.step(action)
            
            # CẤT VÀO BUFFER
            buffer.store(state, action, next_state, reward, float(done))
            
            # LẤY NGẪU NHIÊN TỪ BUFFER ĐỂ HỌC LIÊN TỤC (Giống DQN)
            actor_loss, critic_loss = train_awr_off_policy(
                policy, target_policy, optimizer, buffer, BATCH_SIZE, GAMMA, BETA, WEIGHT_CLIP
            )
            
            # Từ từ sao chép trọng số sang mạng Target
            soft_update(policy, target_policy, TAU)
            
            state = next_state
            ep_reward += reward
            
        ep_rewards_history.append(ep_reward)
        mean_reward = np.mean(ep_rewards_history[-25:])
        
        if episode % 10 == 0:
            print(f"| Tập: {episode:3} | Điểm TB: {mean_reward:5.1f} | A_Loss: {actor_loss:5.2f} | C_Loss: {critic_loss:5.2f} |")
            
        if mean_reward >= 475:
            print(f"\n=> THÀNH CÔNG: Đã đạt ngưỡng 475 điểm ở tập thứ {episode}!")
            break
