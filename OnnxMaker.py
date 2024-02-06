import torch
import torch.onnx
from MovingFighting_DQN import Qnet

# 모델을 CPU로 로드하고 inference 모드로 설정
device = torch.device("cpu")
model = Qnet().to(device)
checkpoint = torch.load("C:/ML/SavedModels/20240204201636/ckpt.pth", map_location=device)
model.load_state_dict(checkpoint["network"])
model.eval()  # inference 모드로 설정

# 모델을 입력 데이터에 맞게 설정
# 예를 들어, 입력 크기가 (1)인 경우:
dummy_input = torch.randn(1,4)

# ONNX로 모델 변환
onnx_path = "C:/Unity_project/RL_Fighting/Assets/OnnxModels/MovingModel.onnx"
input_names = ["x"]  # 입력 노드의 이름 지정
output_names = ["y"]  # 출력 노드의 이름 지정
#dynamic_axes = {'state': {0: 'batch'}, 'action': {14: 'batch'}}
export_params=True
do_constant_folding=True
torch.onnx.export(model, dummy_input,onnx_path,opset_version=10,input_names = ["x"],output_names = ["y"])
