import torch
import torch.nn as nn

D = nn.Sequential(
    nn.Linear(784, 128),    # Generative model의 결과물인 784의 벡터(28*28)가 입력임
    nn.ReLU(),
    nn.Linear(128, 1),      # 0(가짜) 인지, 1(진짜) 인지 Binary Classification 타겟임.
    nn.Sigmoid()
)

G = nn.Sequential(
    nn.Linear(100, 128),    # 길이 100 짜리 벡터(Latent=Random vector) 가 입력임.
    nn.ReLU(),
    nn.Linear(128, 784),
    nn.Tanh()               # 이미지를 -1 ~ +1 사이로 normalization. 
                            # 이것도 학습되니까 굳이 안해도된다?
)

criterion = nn.BCELoss()    # Binary Cross Entropy Loss (h(x), y)
# -y * logh(x) - ( 1 - y ) * log(1 - h(x))

# 두 모델이 충돌하기 때문에, optimizer를 따로 구현함.
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.01)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.01)

# x is a tensor of shape (batch_size, 784)
# z is a tensor of shape (batch_size, 100=latent)

while True:
    # Training D model
    loss = criterion(D(x), 1) + criterion(D(G(z)), 0)
    # 앞에 term은 1이 나와야 됌. (진짜인줄로 알아야 하니까)
    # 뒤에 term은 0이 나와야 함. (log 1 - D(G(z)) 에서 0이어야 최대값이니까)
    loss.backward()         # 모든 가중치에 대해 gradient 계산 들어감.
    d_optimizer.step()      # gradient descent 학습 부분임.

    # training G model
    loss = criterion(D(G(z)), 1)
    loss.backward()
    g_optimizer.step()      # G model에 대해서만 학습하게끔.
