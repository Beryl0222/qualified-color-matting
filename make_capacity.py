"""按固定种子生成容量输入，不计算算子输出。"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
BASE=Path(__file__).resolve().parent

def generate(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(261126)
    arrays={};meta={'format_version':1,'source':'本地合成容量输入','seed':261126}
    H=W=128
    alpha=np.broadcast_to(np.linspace(0,1,W),(H,W)).copy();F=np.zeros((H,W,3))+[.7,.3,.15];B=np.zeros((H,W,3))+[.05,.2,.8]
    F[:,96:]=[.4001,.4,.4];B[:,96:]=[.4,.4,.4]
    arrays['image']=alpha[...,None]*F+(1-alpha[...,None])*B;arrays['synthetic_foreground']=F;arrays['synthetic_background']=B;arrays['synthetic_alpha']=alpha
    arrays['valid']=np.ones((H,W),dtype='bool');arrays['valid'][60:64,60:64]=False
    meta.update(tile=16,epsilon=.001,lambda_color=.02,scribbles=[dict(id='bg',kind='background',pixels=[[r,0] for r in range(H)]),dict(id='fg',kind='foreground',pixels=[[r,W-1] for r in range(H)]),dict(id='cross-tile',kind='foreground',pixels=[[r,63] for r in range(40,80)])],new_backgrounds=[[1,1,1],[.1,.7,.2]])
    np.savez(output/'arrays.npz',**arrays)
    meta['arrays']={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in arrays.items()}
    (output/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return meta

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='生成可复现的合成容量输入')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();meta=generate(args.output)
    print(json.dumps({'数组数':len(meta['arrays']),'资料SHA256':hashlib.sha256((Path(args.output)/'arrays.npz').read_bytes()).hexdigest()},ensure_ascii=False))
