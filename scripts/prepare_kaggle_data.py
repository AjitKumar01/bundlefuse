"""Package the eight project dataset directories, preserving bytes and loader paths."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DATASETS={
 'pog':'bundlemllm/datasets/pog','pog_dense':'bundlemllm/datasets/pog_dense',
 'spotify':'bundlemllm/datasets/spotify','spotify_sparse':'bundlemllm/datasets/spotify_sparse',
 'electronic':'BundleRec/dataset/electronic','food':'BundleRec/dataset/food',
 'clothing':'BundleRec/dataset/clothing','dunnhumby':'dunnhumby'}


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def prepare(args):
    args.stage.mkdir(parents=True,exist_ok=True)
    archive=args.stage/'bundlefuse-data.zip'
    if archive.exists():raise FileExistsError(f'Archive already exists: {archive}')
    entries=[];totals=defaultdict(int)
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for dataset,relative in DATASETS.items():
            source=Path(os.environ.get('DUNN_DIR') or args.data_root/relative) if dataset=='dunnhumby' else args.data_root/relative
            files=sorted(p for p in source.rglob('*') if p.is_file() and not any(part.startswith('.') for part in p.relative_to(source).parts))
            if not files:raise RuntimeError(f'Missing/empty dataset: {dataset}')
            for path in files:
                if path.is_symlink():raise RuntimeError(f'Unexpected symbolic link: {path}')
                name=str(Path(relative)/path.relative_to(source))
                before=path.stat();digest=hashlib.sha256();size=0
                with path.open('rb') as src,z.open(name,'w',force_zip64=True) as dst:
                    for block in iter(lambda:src.read(8*1024*1024),b''):
                        digest.update(block);dst.write(block);size+=len(block)
                after=path.stat()
                if size!=before.st_size or (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                    raise RuntimeError(f'Source changed while packaging: {name}')
                entries.append({'dataset':dataset,'path':name,'bytes':size,'sha256':digest.hexdigest()})
                totals[dataset]+=size
            print('PACKED',dataset,len(files),'files',totals[dataset],'bytes',flush=True)
        # Preserve upstream attribution without treating software licenses as data licenses.
        for folder in ('bundlemllm','BundleRec'):
            for filename in ('README.md','LICENSE'):
                path=args.data_root/folder/filename
                if path.is_file():z.write(path,str(Path('upstream-notes')/folder/filename))
        manifest={'schema_version':1,'datasets':list(DATASETS),'file_count':len(entries),
                  'total_original_bytes':sum(totals.values()),'bytes_by_dataset':dict(totals),
                  'files':entries,'notes':'Complete local dataset directories. No model checkpoints or credentials.'}
        z.writestr('DATA_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    write(args.stage/'DATA_MANIFEST.json',manifest)
    print('VERIFYING all decompressed file hashes',flush=True)
    with zipfile.ZipFile(archive) as z:
        for row in entries:
            digest=hashlib.sha256();size=0
            with z.open(row['path']) as src:
                for block in iter(lambda:src.read(8*1024*1024),b''):digest.update(block);size+=len(block)
            if size!=row['bytes'] or digest.hexdigest()!=row['sha256']:
                raise RuntimeError('Archive verification failed: '+row['path'])
    readme='''# BundleFuse project datasets

Collection of all eight dataset directories used by BundleFuse:
POG, POG Dense, Spotify, Spotify Sparse, Electronic, Food, Clothing, and Dunnhumby.
All local source files are preserved byte-for-byte, including supplied feature tensors,
training/validation/test files, auxiliary interaction files and original-ID mappings.
No fitted BundleFuse checkpoints or account credentials are included.

## Paths

The archive contains these original relative paths:
- bundlemllm/datasets/{pog,pog_dense,spotify,spotify_sparse}/
- BundleRec/dataset/{electronic,food,clothing}/
- dunnhumby/

Kaggle extracts this archive under bundlefuse-data/. Use that folder as the data root.
If downloading the raw archive, extract it to a directory of your choice and use the
directory containing bundlemllm/, BundleRec/ and dunnhumby/. Point BUNDLEFUSE_DATA_ROOT at that root
before importing BundleFuse. DATA_MANIFEST.json lists original sizes and SHA-256 hashes.
Preserve names and bytes; do not convert the supplied text, CSV, NumPy or PyTorch files.

## Sources and ownership

POG/Spotify supplied feature and split files originate from the Bundle-MLLM data collection:
https://huggingface.co/datasets/xhLiu/BundleConstruction
Upstream project: https://github.com/Xiaohao-Liu/Bundle-MLLM
Electronic/Food/Clothing originate from BundleRec:
https://github.com/BundleRec/bundle_recommendation
Dunnhumby files were supplied in the local project data directory; this collection does
not assert a new source license for them.

Upstream README/citation material is retained under upstream-notes/. The included
Bundle-MLLM MIT notice is its software notice and is not a claim that every dataset file
has that license. Original ownership, data licenses and use conditions remain applicable.
This collection grants no additional redistribution rights. Kaggle metadata uses
Other to avoid assigning a new blanket open license to these mixed-source datasets.
'''
    (args.stage/'README.md').write_text(readme)
    write(args.stage/'dataset-metadata.json',{
        'title':'BundleFuse Project Datasets','id':args.owner+'/bundlefuse-project-datasets',
        'subtitle':'Eight bundle-completion datasets with original splits and features',
        'description':readme,'licenses':[{'name':'other'}]})
    digest=hashlib.sha256()
    with archive.open('rb') as src:
        for block in iter(lambda:src.read(8*1024*1024),b''):digest.update(block)
    receipt={'file_count':len(entries),'datasets':list(DATASETS),
             'original_bytes':manifest['total_original_bytes'],'archive_bytes':archive.stat().st_size,
             'archive_sha256':digest.hexdigest(),'verified':True,'stage':str(args.stage.resolve()),
             'kaggle_ref':args.owner+'/bundlefuse-project-datasets','visibility':'public' if args.public else 'private'}
    write(args.stage/'PACKAGING_RECEIPT.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=Path(os.environ.get('BUNDLEFUSE_DATA_ROOT') or ROOT.parent))
    p.add_argument('--stage',type=Path,default=Path('/private/tmp/bundlefuse-kaggle-20261104'))
    p.add_argument('--owner',default='ajit1729')
    p.add_argument('--public',action='store_true',help='record intent to create a public Kaggle dataset')
    prepare(p.parse_args())
