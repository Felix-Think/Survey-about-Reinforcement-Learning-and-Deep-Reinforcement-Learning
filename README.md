RL Projects for Cybersecurity
=============================

Tuyển tập các mô hình RL/DRL áp dụng cho bài toán an ninh mạng (network penetration/ cybersecurity) trên bộ môi trường NASim. Hiện tại repo đã có tác tử Q-Learning bảng (tabular) chạy trên các benchmark scenario của NASim (ví dụ: `tiny`, `small`, …).

1) Chuẩn bị môi trường với uv
-----------------------------
- Cần Python ≥ 3.13 (uv sẽ tự tạo venv tương ứng).
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

2) Chạy training Q-Learning trên NASim
--------------------------------------
```bash
python main.py <env_name> [các_cờ]
```
Các tham số quan trọng:
- `env_name` (bắt buộc): tên benchmark, ví dụ `tiny`.
- `--lr` (float, mặc định 0.001): learning rate.
- `-t/--training_steps` (int, mặc định 10000): số bước train tối đa.
- `-e/--exploration_steps` (int, mặc định 10000): số bước anneal epsilon.
- `--final_epsilon` (float, mặc định 0.05): epsilon cuối.
- `--gamma` (float, mặc định 0.99): hệ số chiết khấu.
- `--seed` (int, mặc định 0)
- `--render_eval` (flag): render episode đánh giá sau khi train.

Ví dụ:
```bash
python main.py tiny --lr 0.005 --training_steps 5000 --render_eval
```

3) Theo dõi log bằng TensorBoard
--------------------------------
- Trong khi train, các scalar được ghi vào thư mục `runs/`.
- Mở TensorBoard:
  ```bash
  tensorboard --logdir runs
  ```
- Mở trình duyệt tại địa chỉ hiển thị (thường http://localhost:6006).

4) Cấu trúc chính
-----------------
- `main.py`: script chạy Q-Learning tabular trên NASim; parse tham số, train, log TensorBoard, đánh giá.
- `models/ql_model.py`: hiện đang chứa cùng logic Q-Learning; có thể tách/bổ sung mô hình khác tại đây.
- `pyproject.toml`: khai báo phụ thuộc (gymnasium, nasim, numpy, torch, pandas).

5) Ghi chú thêm
---------------
- `--render_eval` dùng chế độ `env.render()` của NASim để xem tương tác sau khi train.
- Nếu thêm mô hình mới, giữ nguyên interface tạo env: `nasim.make_benchmark(env_name, seed, fully_obs=True, flat_obs=True)`.
- Kết quả học (bảng Q) hiện được in ra sau train để tiện kiểm tra nhanh.
