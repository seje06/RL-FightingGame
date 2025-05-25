# RL for fighting Game
파이썬과 유니티로 강화학습을 통해 간단히 격투게임의 AI를 구현하기 위한 환경 구성에 대한 것.

참고한 자료로는 "바닥부터 배우는 강화학습", "파이토치와 유니티 ML-Agents로 배우는 강화학습" 책들이다.

### 설명

#### 추천 폴더구조

![image](https://github.com/seje06/Trio/assets/124812852/8e3854e6-db29-4e32-84c9-b5036d1d5172)

C드라이브 바로 아래에 생성하면 된다.
RLVSCode엔 py파일과 json파일들(ActionBuffer,MovingBuffer), SavedModels엔 학습된 파일들, Unity_Program엔 빌드된 게임들이 있다.
ml-agents-release은 다운한 mlagents파일이다.

          
## python(3.9.10)

### 코드들

- [OnnxMaker](OnnxMaker.py)
- [Moving_DQN](MovingFighting_DQN.py)
- [Action_DQN](ActionFighting_DQN.py)

### 설명

OnnxMaker는 해당 딥러닝 모델을 Onnx파일로 만들어준다. 자기 환경에 맞게 세부값 변경 해야함.

#### 버전 다운을 위한 명령어(터미널)
- pip install mlagents==0.30.0
- pip install protobuf==3.20.2
- pip install torch==2.2.0 (1.8.0~)
- pip install torch torchvision torchaudio --index-url https://download.python.org/whl/cu117 (cuda사용시 명령어.url띄워쓰기https해야함)

## unity(2022.3~)

### 설명

#### 버전

- ML Agents Release : 20 [다운 바로가기](https://github.com/Unity-Technologies/ml-agents/releases/tag/release_20)
- ML Agents Extensions : 0.6.1
- ML Agents : 2.3.0

## Games for Environment

- [MovingFighting](../game/MovingFighting.zip)
- [ActionFighting](../game/ActionFighting.zip)

