"""Draw the figures in ../ARCHITECTURE.md.

PYTHONPATH=. .venv/bin/python experiments/neural_relations/figures/make_figures.py --extract   # read learned values from checkpoints
python experiments/neural_relations/figures/make_figures.py                                   # draw from the saved values
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,FancyBboxPatch,Rectangle
import numpy as np

OUT=Path(__file__).parent;RESULTS=Path('results/neural_relations');LEARNED=OUT/'learned.json'
INPUT='#e8eef7';LEARN='#fde8c8';FIXED='#eeeeee';OUTPUT='#dcefdc';INK='#222222';BLUE='#2f6db5';ORANGE='#c8701a';GREEN='#3c8d4f';RED='#b8322a'
plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans'})
NAMES=dict(pog='POG',pog_dense='POG_DENSE',spotify_sparse='Spotify_sparse')


def box(ax,x,y,w,h,text,color=FIXED,size=10.5):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.02,rounding_size=0.08',facecolor=color,edgecolor=INK,linewidth=1.1))
    ax.text(x,y,text,ha='center',va='center',fontsize=size,color=INK,linespacing=1.35)


def arrow(ax,a,b,color=INK,rad=0.):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color=color,linewidth=1.3,connectionstyle=f'arc3,rad={rad}',shrinkA=0,shrinkB=0))


def legend(ax,x,y,dx=2.3):
    for i,(c,t) in enumerate(((INPUT,'input'),(LEARN,'trainable'),(FIXED,'fixed or cross-fitted from data'),(OUTPUT,'output'))):
        ax.add_patch(Rectangle((x+i*dx,y-.11),.3,.22,facecolor=c,edgecolor=INK,linewidth=.8))
        ax.text(x+i*dx+.4,y,t,va='center',fontsize=9.5)


def extract():
    """Read learned relations from the seed 0 v3.1 checkpoints (validation-side quantities only)."""
    import torch
    from experiments.neural_relations import run as r3
    from experiments.neural_lowrank import run as base
    out={'inspect':{d:json.loads((RESULTS/d/'v31_loo_seed0'/'inspect.json').read_text()) for d in NAMES}}
    with torch.inference_mode():
        for name,top in (('pog',None),('spotify_sparse',24)):
            ds,x,c,_,_=base.data(name);model=base.load(ds,x,c,json.loads((RESULTS/name/'v31_loo_seed0'/'selected.json').read_text()))
            weight=out['inspect'][name]['gate_mean']['label0']
            labels=model.label0;size=model.label_size0;n=int(labels.max())+1
            P=model.label_left0;Q=model.label_right0;M=model.label_query[0]
            lowrank=(P@M@Q.T).numpy()
            diag=np.array([model.diagonal[0](torch.tensor([[np.log1p(1.),np.log(10.),float(size[g])]],dtype=torch.float32)).item() for g in range(n)])
            relation=weight*(lowrank+np.diag(diag))
            freq=np.bincount(labels[model.frequency>0].numpy(),minlength=n)
            keep=np.argsort(-freq)[:top] if top else np.argsort(-freq)[:30]
            out[name]=dict(matrix=relation[np.ix_(keep,keep)].tolist(),labels=[int(k) for k in keep])
    LEARNED.write_text(json.dumps(out))


def overview():
    fig,ax=plt.subplots(figsize=(14,8.4));ax.set_xlim(0,14);ax.set_ylim(0,8.4);ax.axis('off')
    ax.text(.25,8.1,'Inputs (every dataset)',fontsize=11,weight='bold')
    box(ax,1.6,7.2,2.7,.8,'Content $x_i$ (128-d)\nimage, text or audio',INPUT,10)
    box(ax,1.6,5.6,2.7,.8,'Labels $g_\\ell(i)$\ncategory; artist, album',INPUT,10)
    box(ax,1.6,4.0,2.7,.8,'Training bundles\nco-purchase counts $N$',INPUT,10)
    ax.text(4.0,8.1,'Three relation families',fontsize=11,weight='bold')
    box(ax,6.3,7.2,4.4,1.05,'Content relation $C$\n$\\varphi(S)^{\\top}\\psi(c)$, rank 64, asymmetric\nviews: content $z$, partner history $h$',LEARN,9.8)
    box(ax,6.3,5.6,4.4,1.25,'Label relation $L_\\ell$, per label level\ndiagonal: $\\delta_\\ell$(matches, $|S|$, label size)\n+ low rank: $\\bar U_S\\,M_\\ell\\,V[g_\\ell(c)]^{\\top}$',LEARN,9.8)
    box(ax,6.3,3.85,4.4,1.25,'Co-purchase graph $G$\n$\\theta_1\\log(1{+}yA)+\\theta_2\\log(1{+}yAA\')+\\rho(f_c)$\n$A=D^{-\\alpha}ND^{-\\beta}$, learned $\\alpha,\\beta,\\gamma$',LEARN,9.8)
    box(ax,6.3,2.25,4.4,.85,'$U,V$ and the 2-hop $LR^{\\top}$: cross-fitted SVDs\n(a training bundle never sees itself)',FIXED,9.5)
    arrow(ax,(2.95,7.2),(4.1,7.2));arrow(ax,(2.95,5.6),(4.1,5.6));arrow(ax,(2.95,4.0),(4.1,3.95))
    arrow(ax,(2.95,4.0),(4.1,7.0),color='#888888');arrow(ax,(2.95,3.9),(4.1,2.3),color='#888888')
    arrow(ax,(6.3,2.68),(6.3,3.22),color='#888888');arrow(ax,(5.0,2.68),(5.0,4.95),color='#888888',rad=-.25)
    box(ax,11.3,6.6,2.9,1.2,'Query gate $w(S)$\nsoftplus MLP on $\\log|S|$,\nseed history, share cold',LEARN,9.8)
    box(ax,11.3,4.3,2.9,1.25,'$s(c\\mid S)=w_C\\,C$\n$+\\sum_\\ell w_\\ell\\,L_\\ell + w_G\\,G$',FIXED,11)
    box(ax,11.3,2.4,2.2,.9,'Rank, drop seeds,\ntop 20',OUTPUT,10)
    for y in (7.2,5.6,3.85):arrow(ax,(8.5,y),(9.85,4.3))
    arrow(ax,(11.3,6.0),(11.3,4.93),color=BLUE);arrow(ax,(11.3,3.67),(11.3,2.85))
    legend(ax,.25,.6,dx=3.3)
    fig.savefig(OUT/'overview.png',dpi=170,bbox_inches='tight');plt.close(fig)


def label_relations(learned):
    fig,axes=plt.subplots(1,2,figsize=(13.5,6.2),gridspec_kw=dict(wspace=.3))
    for ax,(name,title) in zip(axes,(('pog','POG: 30 most-sold categories'),('spotify_sparse','Spotify_sparse: 24 most-played artists'))):
        m=np.asarray(learned[name]['matrix']);lim=np.abs(m).max()
        im=ax.imshow(m,cmap='RdBu_r',vmin=-lim,vmax=lim);ax.set_title(title,fontsize=11)
        ax.set_xlabel('candidate label');ax.set_ylabel('seed label');ax.set_xticks([]);ax.set_yticks([])
        fig.colorbar(im,ax=ax,fraction=.046,pad=.03,label='score added per matching seed')
    fig.text(.5,.02,'Learned label relation for one seed, gate weight included: diagonal (same label) plus cross-fitted low rank. Rows and columns are labels sorted by sales.\n'
             'POG learns a negative diagonal (avoid the seeds\' categories) and signed off-diagonal structure (which categories go together).\n'
             'Spotify learns a strong positive diagonal (same artist) and almost nothing off it. Same family, opposite solutions.',ha='center',va='top',fontsize=10)
    fig.savefig(OUT/'label_relations.png',dpi=170,bbox_inches='tight');plt.close(fig)


def ablation(learned):
    parts=[('diagonal','label diagonal'),('lowrank','label low rank'),('content','content'),('graph','graph (all)'),('hop2','graph 2-hop'),('prior','popularity')]
    fig,ax=plt.subplots(figsize=(11,4.6));x=np.arange(len(parts));w=.26
    for i,(d,color) in enumerate((('pog',ORANGE),('pog_dense','#e3b07a'),('spotify_sparse',BLUE))):
        a=learned['inspect'][d]['ablation_ndcg20'];full=a['full']
        v=[100*(full-a[p])/full for p,_ in parts]
        ax.bar(x+(i-1)*w,v,w,color=color,label=NAMES[d])
    ax.axhline(0,color=INK,linewidth=.8);ax.set_xticks(x,[t for _,t in parts]);ax.set_ylabel('% of validation NDCG@20 lost\nwhen the part is removed')
    ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
    ax.set_title('Seed 0, validation only. Negative bars: removing the part helps slightly.',fontsize=10)
    fig.savefig(OUT/'ablation.png',dpi=170,bbox_inches='tight');plt.close(fig)


def evolution():
    rows=dict(pog=[.0231,.0294,.0346,.0377,.0431,.0417,.0404],pog_dense=[.2701,.3013,.3140,.3141,.3146,.3124,.2849],spotify_sparse=[None,None,.111,.2205,.2480,.2556,.2248])
    labels=['original','category','v1','v2','v3','v3.1']
    fig,axes=plt.subplots(1,3,figsize=(14,4.3),gridspec_kw=dict(wspace=.3))
    for ax,(d,v) in zip(axes,rows.items()):
        xs=[i for i,y in enumerate(v[:-1]) if y is not None];ys=[v[i] for i in xs]
        ax.plot(xs,ys,'o-',color=BLUE);ax.axhline(v[-1],color=ORANGE,linestyle='--',label='BundleFuse')
        ax.set_xticks(range(6),labels,fontsize=9.5);ax.set_title(NAMES[d],fontsize=11);ax.spines[['top','right']].set_visible(False)
        if d=='spotify_sparse':ax.annotate('free label embeddings\noverfit after epoch 2',(4,.248),(1.6,.17),fontsize=8.5,arrowprops=dict(arrowstyle='->',color='#666666'))
    axes[0].set_ylabel('best validation NDCG@20, seed 0');axes[0].legend(frameon=False,fontsize=9)
    fig.savefig(OUT/'evolution.png',dpi=170,bbox_inches='tight');plt.close(fig)


def cross_fitting():
    fig,ax=plt.subplots(figsize=(12.5,4.4));ax.set_xlim(0,12.5);ax.set_ylim(0,4.4);ax.axis('off')
    colors=['#cfe0f3','#f6d7b4','#d6ecd2','#ead3ef','#f3d0cf']
    for k in range(5):
        y=3.6-.62*k;ax.text(.2,y,f'SVD {k+1}',va='center',fontsize=10)
        for j in range(5):
            ax.add_patch(Rectangle((1.3+j*.9,y-.22),.85,.44,facecolor='white' if j==k else colors[j],edgecolor='#999999',hatch='//' if j==k else None))
        ax.text(6.0,y,f'built without fold {k+1}',va='center',fontsize=9.5)
    for j in range(5):ax.text(1.72+j*.9,4.05,f'fold {j+1}',ha='center',fontsize=9)
    box(ax,10.2,2.9,3.6,1.15,'Training bundle in fold 3\nuses SVD 3 for its 2-hop\nand label relations',LEARN,10)
    box(ax,10.2,1.2,3.6,1.0,'Evaluation and test\nuse the SVD of all training bundles',OUTPUT,10)
    arrow(ax,(8.25,2.36),(8.4,2.75),color=BLUE)
    ax.text(.2,.25,'Folds are set by a hash of the bundle, so duplicate bundles share a fold. An SVD that saw a bundle would encode who its items were bought with.',fontsize=9.5)
    fig.savefig(OUT/'cross_fitting.png',dpi=170,bbox_inches='tight');plt.close(fig)


def forward():
    """Shapes for one batch of b queries over n catalogue items (labels: L levels, sizes G_l)."""
    fig,ax=plt.subplots(figsize=(14,10.2));ax.set_xlim(0,14);ax.set_ylim(0,10.2);ax.axis('off')
    ax.text(.2,9.95,'Once per batch (or once for all of evaluation): catalogue-side tensors',fontsize=11,weight='bold')
    box(ax,2.2,9.2,3.9,.8,'$X$ ($n\\times128$) features\n$z=\\mathrm{norm}(X+f(X))$  ($n\\times128$)',LEARN,9.6)
    box(ax,6.9,9.2,4.0,.8,'$H$ ($n\\times128$) partner means\n$h_i=\\sum_j n_{ij}x_j/d_i$',FIXED,9.6)
    box(ax,11.5,9.2,4.0,.8,'keys $\\Psi=zK^{\\top}+HE^{\\top}$  ($n\\times64$)',LEARN,9.6)
    arrow(ax,(4.15,9.2),(4.9,9.2));arrow(ax,(8.9,9.2),(9.5,9.2))
    ax.text(.2,8.3,'Per query $S$ (seeds $y\\in\\{0,1\\}^n$); every row is computed for a batch of $b$ queries at once',fontsize=11,weight='bold')
    rows=[('Content',[('$\\varphi=\\frac{1}{|S|}\\sum_s (Az_s+Bh_s)$','$b\\times64$',LEARN),('$C=\\varphi\\Psi^{\\top}$','$b\\times n$',FIXED)]),
          ('Label $\\ell$',[('tally $t_\\ell$: seeds per label','$b\\times G_\\ell$',FIXED),('diagonal $\\delta_\\ell(\\log(1{+}t_\\ell[g(c)]),\\log|S|,\\log(1{+}|g(c)|))$','$b\\times n$',LEARN),('low rank $(t_\\ell/|S|)\\,U_\\ell M_\\ell V_\\ell[g(c)]^{\\top}$','$b\\times n$',LEARN)]),
          ('Graph',[('1 hop $a_1=yD^{-\\alpha}ND^{-\\beta}$','$b\\times n$',FIXED),('2 hop $a_2=((a_1\\odot d^{-\\gamma})L)R^{\\top}\\odot d^{-\\beta}$','$b\\times n$',FIXED),('$\\theta_1\\log(1{+}a_1)+\\theta_2\\log(1{+}a_2)+\\rho(\\log(1{+}f))$','$b\\times n$',LEARN)]),
          ('Gate',[('$u=(\\log|S|,\\ \\overline{\\log(1{+}f_s)},\\ \\mathrm{share}\\ f_s{=}0)$','$b\\times3$',FIXED),('$w=\\mathrm{softplus}(\\mathrm{MLP}(u))$','$b\\times(2{+}L)$',LEARN)])]
    y=7.5
    for name,steps in rows:
        ax.text(.25,y,name,fontsize=10.5,weight='bold',va='center')
        x=2.0
        for k,(text,shape,color) in enumerate(steps):
            width=3.7 if len(steps)==3 else (5.4 if len(steps)==2 else 3.7)
            box(ax,x+width/2,y,width,.72,text+'\n'+shape,color,8.6)
            if k:arrow(ax,(x-.25,y),(x-.02,y))
            x+=width+.27
        y-=1.15
    box(ax,7.0,2.3,8.6,.85,'$s=w_C\\,C+\\sum_\\ell w_\\ell\\,L_\\ell+w_G\\,G$   ($b\\times n$)',FIXED,11)
    box(ax,3.6,.95,5.2,.9,'Evaluation: drop seeds, top 20 per query',OUTPUT,10)
    box(ax,10.3,.95,5.4,.9,'Training: softmax of each hidden target against\nevery item outside its bundle; mean over targets, batch',OUTPUT,9.4)
    arrow(ax,(7.0,3.1),(7.0,2.75));arrow(ax,(5.5,1.87),(4.4,1.4));arrow(ax,(8.5,1.87),(9.5,1.4))
    fig.savefig(OUT/'forward.png',dpi=170,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--extract',action='store_true');args=p.parse_args()
    if args.extract:extract()
    learned=json.loads(LEARNED.read_text())
    overview();forward();label_relations(learned);ablation(learned);evolution();cross_fitting()
