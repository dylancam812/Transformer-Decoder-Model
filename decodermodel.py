import torch
import torch.nn as nn
from torch.nn import functional as F

batch = 64 
block = 256 
maxIter = 5000
Interval = 500
learningRate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
evalIters = 200
embed = 384
head = 6
layer = 6
dropout = 0.2
torch.manual_seed(812)

with open('data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocabSize = len(chars)
stoi = { c:i for i,c in enumerate(chars) }
itos = { i:c for i,c in enumerate(chars) }
encoder = lambda s: [stoi[c] for c in s] 
decoder = lambda l: ''.join([itos[i] for i in l]) 
data = torch.tensor(encoder(text), dtype=torch.long)
n = int(0.9*len(data)) 
trainData = data[:n]
valData = data[n:]

def get_batch(split):
    data = trainData if split == 'train' else valData
    ix = torch.randint(len(data) - block, (batch,))
    x = torch.stack([data[i:i+block] for i in ix])
    y = torch.stack([data[i+1:i+block+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(evalIters)
        for k in range(evalIters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ Single head self-attention """

    def __init__(self, headSize):
        super().__init__()
        self.key = nn.Linear(embed, headSize, bias=False)
        self.query = nn.Linear(embed, headSize, bias=False)
        self.value = nn.Linear(embed, headSize, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block, block)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei)
        v = self.value(x) 
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):
    """ Multi head self-attention """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ Linear layer """

    def __init__(self, embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed, 4 * embed),
            nn.ReLU(),
            nn.Linear(4 * embed, embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block """

    def __init__(self, embed, head):
        super().__init__()
        head_size = embed // head
        self.sa = MultiHeadAttention(head, head_size)
        self.ffwd = FeedFoward(embed)
        self.ln1 = nn.LayerNorm(embed)
        self.ln2 = nn.LayerNorm(embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    """ Decoder model """

    def __init__(self):
        super().__init__()
        self.tokenEmbeddingTable = nn.Embedding(vocabSize, embed)
        self.positionEmbeddingTable = nn.Embedding(block, embed)
        self.blocks = nn.Sequential(*[Block(embed, head=head) for _ in range(layer)])
        self.ln_f = nn.LayerNorm(embed) 
        self.lm_head = nn.Linear(embed, vocabSize)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tokenEmbed = self.tokenEmbeddingTable(idx) 
        positionEmbed = self.positionEmbeddingTable(torch.arange(T, device=device)) 
        x = tokenEmbed + positionEmbed 
        x = self.blocks(x) 
        x = self.ln_f(x) 
        logits = self.lm_head(x) 
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1) 
            next = torch.multinomial(probs, numSamples=1) 
            idx = torch.cat((idx, next), dim=1) 
        return idx

model = GPTLanguageModel()
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate)

for iter in range(maxIter):
    if iter % Interval == 0 or iter == maxIter - 1:
        losses = estimate_loss()
        print(f"step {iter}: training loss {losses['train']:.4f}, validation loss {losses['val']:.4f}")
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decoder(m.generate(context, max_new_tokens=500)[0].tolist()))
