import os, json, io, numpy as np, torch
from concurrent.futures import ThreadPoolExecutor
import requests
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
SP='/private/tmp/claude-501/-Users-ajit-Projects-Career-change/2594a4a1-49ad-4214-9ef7-56481f666b97/scratchpad'
IMGDIR=SP+'/elec_imgs'; os.makedirs(IMGDIR, exist_ok=True)
urls=json.load(open(SP+'/elec_item_urls.json'))
ids=list(urls.keys()); HDR={'User-Agent':'Mozilla/5.0'}
def fetch(it):
    fp=os.path.join(IMGDIR,it+'.jpg')
    if os.path.exists(fp) and os.path.getsize(fp)>0:
        return it, True
    try:
        r=requests.get(urls[it],headers=HDR,timeout=15)
        if r.status_code==200 and r.content:
            open(fp,'wb').write(r.content); return it, True
    except Exception: pass
    return it, False
ok=[]
with ThreadPoolExecutor(max_workers=16) as ex:
    for n,(it,good) in enumerate(ex.map(fetch, ids)):
        if good: ok.append(it)
        if (n+1)%400==0: print(f'  fetched {n+1}/{len(ids)} ok={len(ok)}',flush=True)
print('cached images:',len(ok),flush=True)
model=CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
proc=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
keys=[]; feats=[]; B=64
def load(it):
    try: return Image.open(os.path.join(IMGDIR,it+'.jpg')).convert('RGB')
    except Exception: return None
with torch.no_grad():
    for s in range(0,len(ok),B):
        chunk=ok[s:s+B]; ims=[load(k) for k in chunk]
        good=[(k,im) for k,im in zip(chunk,ims) if im is not None]
        if not good: continue
        inp=proc(images=[im for _,im in good],return_tensors='pt')
        out=model.get_image_features(pixel_values=inp['pixel_values'])
        f=out if torch.is_tensor(out) else getattr(out,'image_embeds',getattr(out,'pooler_output'))
        f=f/(f.norm(dim=-1,keepdim=True)+1e-8)
        feats.append(f.cpu().numpy().astype(np.float32)); keys+=[k for k,_ in good]
        if (s//B)%5==0: print(f'  embedded {len(keys)}',flush=True)
F=np.concatenate(feats,0) if feats else np.zeros((0,512),np.float32)
np.savez(SP+'/elec_clip.npz', ids=np.array([int(k) for k in keys]), feats=F)
print(f'CLIP_DONE embedded={F.shape}',flush=True)
