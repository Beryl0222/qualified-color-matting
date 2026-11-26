"""读取并核对本地合成资料，不计算产品结果。"""
from pathlib import Path
import hashlib,json,math
import numpy as np

BASE=Path(__file__).resolve().parent

def _finite(value):
    if isinstance(value,dict):
        for v in value.values():_finite(v)
    elif isinstance(value,list):
        for v in value:_finite(v)
    elif isinstance(value,float):
        if not math.isfinite(value):raise ValueError('资料含非有限数值')

def validate():
    d=json.loads((BASE/'data/materials.json').read_text(encoding='utf-8'))
    events=json.loads((BASE/'data/events.json').read_text(encoding='utf-8'))
    _finite(d)
    assert d['metadata']['format_version']==1
    assert isinstance(events,list) and len(events)>5
    assert all(isinstance(e.get('op'),str) for e in events)
    a=np.asarray(d['image']);assert a.shape==np.asarray(d['synthetic_foreground']).shape
    assert np.asarray(d['valid']).shape==a.shape[:2] and 0<=a.min()<=a.max()<=1
    al=np.asarray(d['synthetic_alpha'])[...,None]
    assert np.allclose(a,al*np.asarray(d['synthetic_foreground'])+(1-al)*np.asarray(d['synthetic_background']))
    ids=[x['id'] for x in d['scribbles']];assert len(ids)==len(set(ids))
    assert len(d['new_backgrounds'])==2
    return d,events

if __name__=='__main__':
    d,events=validate()
    raw=(BASE/'data/materials.json').read_bytes()+(BASE/'data/events.json').read_bytes()
    print(json.dumps({'产品':d['metadata']['product'],'资料核对':'通过','操作输入数':len(events),'资料SHA256':hashlib.sha256(raw).hexdigest()},ensure_ascii=False))
