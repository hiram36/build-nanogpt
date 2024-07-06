import torch

model = torch.nn.Sequential(
    torch.nn.Linear(16,32),
    torch.nn.GELU(),
    torch.nn.Linear(32,1)
)

batch_size = 4
torch.random.manual_seed(43)
x = torch.randn(batch_size, 16)
y = torch.randn(batch_size, 1)

# batch_size=4， 计算1次
model.zero_grad()
y_hat = model(x)
loss = torch.nn.functional.mse_loss(y_hat, y)
loss.backward()
'''
loss[i] = (y[i]-y[i]_hat)**2

LOSS = (1/batch_size) * (
(y[0]-y[0]_hat)**2+
(y[1]-y[1]_hat_**2+
(y[2]-y[2]_hat)**2+
(y[3]-y[3]_hat)**2
)

'''
val = model[0].weight.grad.view(-1)[:10]
print(val)
print(loss)

#  batch_size=1， 计算4次
model.zero_grad()
for step in range(batch_size):
    y_hat = model(x[step])
    loss = torch.nn.functional.mse_loss(y_hat, y[step])
    # loss = (y[i]-y[i]_hat)**2
    # 梯度的累积约等于loss的求和
    # 每个循环的loss约为1个mse损失
    # 这里均方误差mse丢失了（ 实际应该为 (1/batch_size)cl * loss）
    loss.backward()

val = model[0].weight.grad.view(-1)[:10]
print(val)
print(loss)

#  batch_size=1， 计算4次
model.zero_grad()
for step in range(batch_size):
    y_hat = model(x[step])
    loss = torch.nn.functional.mse_loss(y_hat, y[step])
    # 梯度的累积约等于loss的求和
    # 每个循环的loss约为1个mse损失
    # 这里均方误差mse丢失了（ 1/n * loss）
    loss = loss / batch_size
    loss.backward()

val = model[0].weight.grad.view(-1)[:10]
print(val)
print(loss)