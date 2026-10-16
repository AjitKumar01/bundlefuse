import os, csv, json, re
import pyarrow.parquet as pq
from collections import defaultdict
from huggingface_hub import hf_hub_download
SP='/private/tmp/claude-501/-Users-ajit-Projects-Career-change/2594a4a1-49ad-4214-9ef7-56481f666b97/scratchpad'
root='BundleRec/dataset/electronic'
bi=defaultdict(list); r=csv.reader(open(root+'/bundle_item.csv')); next(r)
for b,it in r: bi[int(b)].append(int(it))
items=sorted({i for v in bi.values() for i in v})
m={}; rr=csv.reader(open(root+'/item_idx_mapping.csv')); next(rr)
for row in rr:
    if len(row)>=2: m[int(row[0])]=row[1]
item2asin={i:m[i] for i in items if i in m}
asin2item=defaultdict(list)
for i,a in item2asin.items(): asin2item[a].append(i)
sset=set(item2asin.values())
print('items',len(items),'asins',len(sset),flush=True)
asin2url={}
# metadata shards
for k in range(10):
    fn=f'raw_meta_Electronics/full-{k:05d}-of-00010.parquet'
    p=os.path.join(SP,fn)
    if not os.path.exists(p):
        p=hf_hub_download('McAuley-Lab/Amazon-Reviews-2023',fn,repo_type='dataset',local_dir=SP)
    t=pq.ParquetFile(p).read(columns=['parent_asin','images'])
    pa=t.column('parent_asin').to_pylist(); ic=t.column('images').to_pylist()
    for a,im in zip(pa,ic):
        if a in sset and a not in asin2url and im:
            url = (im.get('hi_res') or im.get('large') or im.get('thumb') or [None])
            url = next((u for u in url if u), None)
            if url: asin2url[a]=url
    try: os.remove(p)
    except: pass
    print(f'shard{k}: cumulative asin-with-url={len(asin2url)}',flush=True)
# backfill with the 452 direct urls (upscale _US40_ -> _SL1000_)
ru=csv.reader(open(root+'/img_urls.csv')); nb=0
for row in ru:
    if len(row)>=2 and row[0].isdigit():
        it=int(row[0]); a=item2asin.get(it)
        if a and a not in asin2url:
            u=re.sub(r'_AC_US\d+_', '_AC_SL1000_', row[1]); asin2url[a]=u; nb+=1
print('backfilled from direct urls:', nb,flush=True)
cov=len([i for i in items if item2asin.get(i) in asin2url])
print(f'FINAL item coverage: {cov}/{len(items)} ({100*cov/len(items):.1f}%)',flush=True)
json.dump({str(i):asin2url[item2asin[i]] for i in items if item2asin.get(i) in asin2url}, open(SP+'/elec_item_urls.json','w'))
print('HARVEST_DONE',flush=True)
