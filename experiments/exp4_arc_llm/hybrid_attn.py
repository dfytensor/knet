"""Loop 4: 混合注意力 = 局部窗口 softmax ⊕ 全局 GLA. 测它是否 Pareto 占优两纯版.
  短程靠 softmax(窗口 W, band mask, O(T·W)); 长程靠 GLA(O(T)). 总成本仍 O(T).
  假设: 短上下文(seq256)混合 < GLA(借 softmax 精度); 长上下文(1024/2048)混合 ≈ GLA 且 << softmax.
对照: pure softmax(O(T²)), pure GLA, hybrid(W=128).
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, os, sys, csv, time, argparse
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
DEV='cuda'; OUT=os.path.dirname(os.path.abspath(__file__))
CACHE=os.environ.get('KINT_CACHE', r'F:\OpenASH2605\train_60m\cache\pt_cache_openash_512_openash.pt')
sys.path.insert(0, OUT)
from arch_compare import SoftmaxAttn, GlaAttn, DenseFFN, VOCAB


class LocalSoftmax(nn.Module):
    """窗口 W 的因果 softmax(band mask). 朴素实现带状掩码; 融合 kernel 下真 O(T·W)."""
    def __init__(self, d, h, W):
        super().__init__(); self.W=W; self.inner=SoftmaxAttn(d,h)
    def forward(self,x):
        B,T,d=x.shape; W=self.W
        ao=self.inner(x)                                  # 先算全 softmax
        # 用 band mask 重新算? SoftmaxAttn 内部是全因果。这里改成窗口: 重做一个带状版本
        return self._windowed(x)
    def _windowed(self,x):
        B,T,d=x.shape; H=self.inner.h; hd=self.inner.hd; W=self.W
        q=self.inner.q(x).view(B,T,H,hd).transpose(1,2); k=self.inner.k(x).view(B,T,H,hd).transpose(1,2); v=self.inner.v(x).view(B,T,H,hd).transpose(1,2)
        a=(q@k.transpose(-1,-2))/math.sqrt(hd)
        idx=torch.arange(T,device=x.device); rel=idx[None,:]-idx[:,None]            # rel[i,j]=j-i
        mask=(rel>0)|(rel<-W)                                                         # 上三角(未来) 或 超过 W 步以前 => 屏蔽
        a=a.masked_fill(mask,-float('inf'))
        return self.inner.o((F.softmax(a,-1)@v).transpose(1,2).reshape(B,T,d))


class Block(nn.Module):
    def __init__(self,d,h,attn_t,d_ffn,W=128):
        super().__init__(); self.n1=nn.LayerNorm(d); self.n2=nn.LayerNorm(d)
        self.ffn=DenseFFN(d,d_ffn)
        if attn_t=='softmax': self.attn=SoftmaxAttn(d,h); self.hybrid=False
        elif attn_t=='gla': self.attn=GlaAttn(d,h); self.hybrid=False
        else:
            self.attn_local=LocalSoftmax(d,h,W); self.attn_gla=GlaAttn(d,h); self.hybrid=True
    def forward(self,x):
        if self.hybrid:
            x=x+self.attn_local(self.n1(x))+self.attn_gla(self.n1(x))
        else:
            x=x+self.attn(self.n1(x))
        f,_=self.ffn(self.n2(x))
        return x+f

class LM(nn.Module):
    def __init__(self,attn_t,d=256,h=8,L=6,d_ffn=1024,W=128,max_len=2100):
        super().__init__(); self.em=nn.Embedding(VOCAB,d); self.pos=nn.Embedding(max_len,d)
        self.blocks=nn.ModuleList([Block(d,h,attn_t,d_ffn,W) for _ in range(L)])
        self.norm=nn.LayerNorm(d); self.head=nn.Linear(d,VOCAB,bias=False)
    def forward(self,x):
        h=self.em(x)+self.pos(torch.arange(x.size(1),device=x.device))
        for b in self.blocks: h=b(h)
        return self.head(self.norm(h)), None

class DS(Dataset):
    def __init__(s,d,se): s.d,s.se=d,se
    def __len__(s): return len(s.d)
    def __getitem__(s,i): return s.d[i][:s.se+1]
    @staticmethod
    def collate(it): p=pad_sequence(it,batch_first=True,padding_value=0); return p[:,:-1],p[:,1:]

def run(name,attn_t,steps=1000,seq=256,batch=32,lr=3e-4,wd=0.1,n_val=2000,W=128):
    torch.manual_seed(0)
    data=torch.load(CACHE,weights_only=False); val_d,tr_d=data[:n_val],data[n_val:300000]
    model=LM(attn_t,d=256,h=8,L=6,W=W,max_len=seq+8).to(DEV)
    print(f'[{name}] attn={attn_t} seq={seq} W={W} params={sum(p.numel() for p in model.parameters()):,}',flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    tr=DataLoader(DS(tr_d,seq),batch_size=batch,shuffle=True,num_workers=0,collate_fn=DS.collate,drop_last=True)
    vloader=DataLoader(DS(val_d,seq),batch_size=batch,shuffle=False,num_workers=0,collate_fn=DS.collate)
    warm=int(0.05*steps)
    def vmet():
        model.eval();t=0.0;c=0
        with torch.no_grad():
            for x,y in vloader:
                x,y=x.to(DEV),y.to(DEV); lo,_=model(x)
                l=F.cross_entropy(lo.reshape(-1,VOCAB),y.reshape(-1),ignore_index=0,reduction='sum');t+=float(l);c+=int((y!=0).sum())
        model.train(); return t/c
    t0=time.time()
    for st in range(1,steps+1):
        for g in opt.param_groups: g['lr']=lr*min(1.0,st/warm)
        x,y=next(iter(tr));x,y=x.to(DEV),y.to(DEV)
        lo,_=model(x); loss=F.cross_entropy(lo.reshape(-1,VOCAB),y.reshape(-1),ignore_index=0)
        opt.zero_grad(); loss.backward(); opt.step()
        if st%200==0:
            vl=vmet()
            if st%400==0 or st<=200: print(f'  [{name}] s{st} val_ppl={math.exp(vl):.2f}',flush=True)
    vl=vmet(); dt=time.time()-t0
    print(f'[{name}] DONE seq={seq} val_ppl={math.exp(vl):.2f} ({dt:.0f}s)\n',flush=True)
    return math.exp(vl)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--attn',default='hybrid',choices=['softmax','gla','hybrid'])
    ap.add_argument('--seq',type=int,default=256); ap.add_argument('--steps',type=int,default=1000)
    ap.add_argument('--W',type=int,default=128); ap.add_argument('--batch',type=int,default=32)
    a=ap.parse_args()
    run(a.attn+'_'+str(a.seq), a.attn, steps=a.steps, seq=a.seq, batch=a.batch, W=a.W)
