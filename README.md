RL Projects for Cybersecurity
=============================

Tuyển tập các mô hình RL/DRL áp dụng cho bài toán an ninh mạng (network penetration / cybersecurity) trên bộ môi trường NASim. Hiện tại repo đã hoàn thiện 2 model chính:

- **Tabular Q-Learning** với **Experience Replay** trên NASim benchmark.

---

## 1. Chuẩn bị môi trường với uv

- Cần Python ≥ 3.13.
- Các bước:
  ```bash
  # Tạo venv tại .venv
  uv venv
  # Kích hoạt
  source .venv/bin/activate
  # Cài dependencies từ pyproject.toml
  uv pip install -e .
  ```
  Nếu chưa cài uv: `pip install uv` hoặc xem hướng dẫn tại https://github.com/astral-sh/uv.

---

## 2. Chạy training

```bash
python models/ql_replay_model.py <env_name> [các_cờ]
```

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `env_name` | — | Tên benchmark, ví dụ `tiny`, `small`, `medium` |
| `--lr` | `0.001` | Learning rate |
| `-t --training_steps` | `10000` | Số bước train tối đa |
| `--batch_size` | `32` | Kích thước batch sampled từ replay buffer |
| `--replay_size` | `100000` | Dung lượng replay buffer (số experience tối đa) |
| `--gamma` | `0.99` | Hệ số chiết khấu (discount factor) |
| `-e --exploration_steps` | `10000` | Số bước để anneal epsilon từ 1.0 → `final_epsilon` |
| `--final_epsilon` | `0.05` | Giá trị epsilon tối thiểu (phần trăm exploration cuối) |
| `--seed` | `0` | Random seed |
| `--render_eval` | — | Render episode đánh giá sau khi train |
| `--quite` | — | Tắt log in ra console (mặc định: bật) |

**Ví dụ:**
```bash
# Train 5000 bước với learning rate 0.005
python models/ql_replay_model.py tiny --lr 0.005 -t 5000 --render_eval

# Train đầy đủ trên small scenario
python models/ql_replay_model.py small -t 50000 --batch_size 64
```

---

## 3. Theo dõi log bằng TensorBoard

- Trong khi train, các scalar được ghi vào thư mục `runs/`.
- Mở TensorBoard:
  ```bash
  tensorboard --logdir runs
  ```
- Mở trình duyệt tại http://localhost:6006.

**Các metrics được log:**

| Metric | Mô tả |
|--------|-------|
| `episode` | Số episode đã train |
| `epsilon` | Giá trị epsilon hiện tại |
| `episode_return` | Tổng reward của episode |
| `episode_steps` | Số bước trong episode |
| `episode_goal_reached` | Episode có đạt mục tiêu không (0/1) |
| `train/mean_td_error` | Trung bình TD error của batch |
| `train/s_value` | Trung bình max Q-value của batch |

---

## 4. Cấu trúc chính

```
rl-projects/
├── README.md
├── pyproject.toml
├── models/
│   └── ql_replay_model.py   # Q-Learning + Experience Replay
│   └── ql_model.py          # (chưa dùng, Q-Learning cơ bản)
└── runs/                     # TensorBoard logs
```

**Trong `ql_replay_model.py`:**

| Thành phần | Mô tả |
|-----------|-------|
| `ReplayMemory` | Ring buffer lưu experience `(s, a, r, s', done)` |
| `TabularQLearning` | Q-table dạng dict, tra cứu bằng string key |
| `QLearningAgent` | Agent hoàn chỉnh: train, eval, replay, logging |

**Luồng training:**
```
1. collect transition: env.step(a) → store vào ReplayMemory
2. sample batch: khi replay.size >= batch_size → optimize từ batch
3. update Q-table: TD error từ batch ngẫu nhiên (không chỉ transition vừa thu)
4. log TensorBoard: td_error, s_value, episode metrics
```

---

## 5. Mở rộng

### Thêm mô hình mới (DQN, Policy Gradient, ...)

- Giữ nguyên cách tạo env: `nasim.make_benchmark(env_name, seed, fully_obs=True, flat_obs=True)`
- `flat_obs=True` và `fully_obs=True` là **bắt buộc** cho Tabular Q-Learning
- Nếu dùng DQN: thay `TabularQLearning` bằng `QNetwork` (torch.nn.Module)
- ReplayMemory **tái sử dụng được** cho DQN (không cần sửa)

### NASim benchmark scenarios

| Scenario | Mô tả | Độ khó |
|---------|-------|--------|
| `tiny` | Mạng nhỏ, 1 subnet | Dễ |
| `small` | Mạng nhỏ, vài hosts | Trung bình |
| `medium` | Mạng trung bình | Khó |
| `large` | Mạng lớn | Rất khó |

---

## 6. Giải thích Experience Replay

**Không có Replay** (online Q-Learning):
```
s1 → train → s2 → train → s3 → train → ...
(training samples rất correlated → học không ổn định)
```

**Có Replay** (Experience Replay):
```
s1 → store    s2 → store    s3 → store    s4 → store    ...
                                          ↓
                               sample random batch [s1, s3, s5, s2]
                                          ↓
                               train từ batch (decorrelated)
```

**Lợi ích:**
- Break correlation giữa consecutive samples
- Reuse experience hiệu quả hơn (1 experience train nhiều lần)
- Batch training ổn định hơn, converge nhanh hơn
