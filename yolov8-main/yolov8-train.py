import torch
from ultralytics.nn.tasks import DetectionModel

# 注册安全类（关键步骤）
torch.serialization.add_safe_globals([DetectionModel])

# 覆盖PyTorch加载逻辑
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False  # 强制禁用安全模式
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # 使用类构造函数

# 训练配置
model.train(
    data='yolo-bvn.yaml',
    workers=0,
    epochs=30,
    conf=0.1,
    batch=16
)