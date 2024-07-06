

import torch
import time
import math
import os

from model import GPT, GPTConfig
from my_data_loader import DataLoaderLite, get_tiny_datasets, TinyDataLoaderLite

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

def get_device_type():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"device: {device}")

    return device

def setup():
    init_process_group(backend='nccl')

def cleanup():
    destroy_process_group()

'''
DDP内部机制:
每个 GPU 都由一个进程控制，这些 GPU 可以都在同一个节点上 (单机)，也可以分布在多个节点上 (多机)。
每个进程都执行相同的任务，并且每个进程都与所有其他进程通信。
进程或者说 GPU 之间只传递梯度，这样网络通信就不再是瓶颈
'''
# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
# torchrun --standalone --nproc-per-node=gpu train_gpt2.py
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # 主机内GPU编号,0,1,...,n
    ddp_world_size = int(os.environ['WORLD_SIZE']) # 全局进程数 1个进程控制1个GPU， 等于显卡数
    device = f'cuda:{ddp_local_rank}'

    torch.cuda.set_device(device) # DDP：DDP backend初始化
    #init_process_group(backend='nccl')
    setup()

    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = get_device_type()

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 50
#max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

model = GPT(GPTConfig(vocab_size=50304))

ckpt_path = None
# dist.get_rank()  进程号
if dist.get_rank() == 0 and ckpt_path is not None:
    model.load_state_dict(torch.load(ckpt_path))

model.to(device=device_type)
use_compile = True
# 减少python开销和GPU读写
if torch.cuda.is_available():
    if use_compile:
        model = torch.compile(model=model)

if ddp:
    # 每个独立GPU上的backward结束，每个独立GPU都会拥有所有参数的梯度
    model = DDP(module=model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

# gpt3-small 的训练batch_size=0.5M，取2^19，大约为500000
total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
# 如果GPU显存不足，建议调低B值和T值
B = 4 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"===> calculated gradient accumulation steps: {grad_accum_steps}")

#train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
#val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

filepath = 'input.txt'
train_loader = TinyDataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = TinyDataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

# 运行的内核, 不同的级别对应不同的精度
torch.set_float32_matmul_precision('high')

'''
import tiktoken
enc = tiktoken.get_encoding('gpt2')
filepath = 'input.txt'
text = get_tiny_datasets(filepath=filepath)
text = text[:1000]
tokens = enc.encode(text=text)
buf = torch.tensor(tokens[:B*T+1]).to(device=device_type)
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)
logits, loss = model(x, y)
#print(f'logits===> {logits}')
print(f'loss===> {loss:.4f}')
'''
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

lr = 6e-4
train_steps = 50
betas = (0.9, 0.95)
eps = 1e-8
#optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, eps=eps)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=lr, device_type=device_type)

# create the log directory we will write checkpoints to and log to
log_dir = "log124M_10B"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad() #梯度清零
    loss_accum = 0.0 # loss累加
    # forward-backward梯度累积grad_accum_steps， 然后在所有这些累积后进行一次参数更新
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x = x.to(device=device_type)
        y = y.to(device=device_type)
        # 向后梯度同步require_backward_grad_sync
        # added after video, this field is also used by the forward pass.
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        
        if torch.cuda.is_available():
            # softmax layernorm 这些还是 float32
            # 矩阵乘法这些会转换成精度bfloat16
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward() # 反向传播

    if ddp:

        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    '''
    clip_grad_norm_: 范数裁剪
    基于范数的裁剪方法其核心思想是：
    ①先计算所有参数梯度各自的范数；
    ②然后再计算第①步中得到的各个参数梯度范数的范数；
    ③进一步根据给定的最大范数同第②步得到的范数计算得到一个缩放系数；
    ④最后再将该系数作用于原始各个参数的梯度得到最终裁剪后的结果。
    '''
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step() # 参数更新

    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work

    t1 = time.time()
    time_used = (t1-t0)*1000 #ms
    dt = t1 - t0
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt

    if master_process:
        print(f'step: {step} | loss: {loss.item():.4f} | lr: {lr:.4e} | norm: {norm:.4f} | time_used: {time_used:.4f}ms | tokens/sec: {tokens_per_sec:.0f}') #loss.item(), loss被送回CPU，并转换为浮点数，用于打印
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.4f}\n")

# DDP:
# 1. save模型的时候，和DP模式一样，有一个需要注意的点：保存的是model.module而不是model。
#    因为model其实是DDP model，参数是被`model=DDP(model)`包起来的。
# 2. 只需要在进程0上保存一次就行了，避免多次保存重复的东西。
ckpt_path = 'checkpoints/nanogpt-124M.ckpt'
if dist.get_rank() == 0:
    if ddp:
        torch.save(raw_model.state_dict(), ckpt_path)
    else:
        torch.save(model.module.state_dict(), ckpt_path)

# 同步进程的运行状态，只有0号进程保存完成了model，再继续后面的运行
dist.barrier()

if ddp:
    # 撤销进程组，释放资源
    #destroy_process_group()
    cleanup()

import sys; sys.exit(0)


