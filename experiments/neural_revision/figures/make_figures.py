"""Draw the architecture figures in ARCHITECTURE.md: python experiments/neural_revision/figures/make_figures.py"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,FancyBboxPatch,Rectangle
import numpy as np

OUT=Path(__file__).parent
INPUT='#e8eef7';LEARNED='#fde8c8';FIXED='#eeeeee';OUTPUT='#dcefdc';INK='#222222';BLUE='#2f6db5';ORANGE='#c8701a'
plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans'})


def box(ax,x,y,w,h,text,color=FIXED,size=10.5,edge=INK):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.02,rounding_size=0.08',facecolor=color,edgecolor=edge,linewidth=1.1))
    ax.text(x,y,text,ha='center',va='center',fontsize=size,color=INK,linespacing=1.35)


def arrow(ax,a,b,color=INK,rad=0.,style='-|>',ls='-'):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=13,color=color,linewidth=1.3,linestyle=ls,connectionstyle=f'arc3,rad={rad}',shrinkA=0,shrinkB=0))


def canvas(w,h,xmax,ymax):
    fig,ax=plt.subplots(figsize=(w,h));ax.set_xlim(0,xmax);ax.set_ylim(0,ymax);ax.axis('off');return fig,ax


def legend(ax,x,y,dx=2.3):
    for i,(c,t) in enumerate(((INPUT,'input'),(LEARNED,'trainable'),(FIXED,'fixed operation'),(OUTPUT,'output'))):
        ax.add_patch(Rectangle((x+i*dx,y-.11),.3,.22,facecolor=c,edgecolor=INK,linewidth=.8))
        ax.text(x+i*dx+.4,y,t,va='center',fontsize=9.5)


def overview():
    fig,ax=canvas(13,6.6,13,6.6)
    ax.text(.25,6.3,'Item-vector path: runs over the whole catalogue',fontsize=11,weight='bold')
    y=5.45
    box(ax,1.05,y,1.7,1.0,'Item features\n$X$  ($n\\times128$)\ncatalogue only',INPUT)
    box(ax,3.65,y,2.1,1.0,'Residual encoder\n$z_i=x_i+f(x_i)$\n128 → 32 → 128',LEARNED)
    box(ax,6.5,y,2.2,1.0,'Graph layer\n$z_i+P\\,\\mathrm{mean}_{j\\sim i}\\,z_j$\n$P$: $128\\times128$',LEARNED)
    box(ax,9.2,y,1.8,1.0,'L2 normalize\nrows',FIXED)
    box(ax,11.75,y,1.9,1.0,'Item vectors\n$V$  ($n\\times128$)',FIXED)
    for a,b in ((1.9,2.6),(4.7,5.4),(7.6,8.3),(10.1,10.8)):arrow(ax,(a,y),(b,y))
    box(ax,6.5,3.95,2.2,.75,'Observed pairs $E$\nfrom training bundles',INPUT)
    arrow(ax,(6.5,4.33),(6.5,4.95))
    ax.text(.25,3.25,'Scoring path: runs once per query',fontsize=11,weight='bold')
    box(ax,1.05,1.85,1.7,1.05,'Seed items $S$\nas indicator\n$y\\in\\{0,1\\}^n$',INPUT)
    box(ax,5.6,2.5,3.3,.95,'Pair term  $p=W^{\\top}y$\n$W$: one weight per pair in $E$, $w\\geq0$',LEARNED,size=10)
    box(ax,5.6,1.05,3.3,.95,'Text term  $t=\\frac{\\tau}{|S|}\\,VV^{\\top}y$\n$\\tau$: temperature, starts at 10',LEARNED,size=10)
    arrow(ax,(1.9,2.05),(3.93,2.45));arrow(ax,(1.9,1.65),(3.93,1.1))
    arrow(ax,(6.5,3.57),(6.5,2.99),color=ORANGE,ls='--')
    ax.text(6.65,3.28,'$E$ fixes where $W$ may be nonzero',color=ORANGE,fontsize=9,va='center',ha='left')
    arrow(ax,(11.75,4.95),(7.27,1.3),color=BLUE)
    ax.text(10.15,3.25,'$V$',color=BLUE,fontsize=12)
    box(ax,10.5,1.8,1.15,.7,'$s=t+p$',FIXED)
    box(ax,12.2,1.8,1.25,1.0,'Rank, drop\nseeds,\ntop 20',OUTPUT,size=9.5)
    arrow(ax,(7.27,2.5),(9.9,1.95));arrow(ax,(7.27,.95),(9.9,1.6));arrow(ax,(11.1,1.8),(11.55,1.8))
    legend(ax,.25,.22)
    fig.savefig(OUT/'overview.png',dpi=170,bbox_inches='tight');plt.close(fig)


def toy():
    items='ABCDE';n=5
    bundles=[('A','B','C'),('B','D'),('C','D','E')]
    W=np.full((n,n),np.nan)
    values={('A','B'):.9,('A','C'):.2,('B','A'):.7,('B','C'):1.1,('C','A'):.3,('C','B'):.8,('B','D'):.4,('D','B'):.6,('C','D'):.5,('C','E'):1.3,('D','C'):.2,('D','E'):.7,('E','C'):1.0,('E','D'):.1}
    for (a,b),v in values.items():W[items.index(a),items.index(b)]=v
    y=np.array([0,1,1,0,0.]);p=np.nansum(W*y[:,None],axis=0)
    fig,axes=plt.subplots(1,3,figsize=(12,3.9),gridspec_kw=dict(width_ratios=[1.15,.3,.3],wspace=.45))
    ax=axes[0];ax.set_xlim(-.5,4.5);ax.set_ylim(4.5,-.5);ax.set_aspect('equal')
    for r in range(n):
        for c in range(n):
            observed=not np.isnan(W[r,c]);active=observed and y[r]==1
            ax.add_patch(Rectangle((c-.5,r-.5),1,1,facecolor='#f7c98b' if active else ('#fde8c8' if observed else 'white'),edgecolor='#999999'))
            ax.text(c,r,f'{W[r,c]:.1f}' if observed else ('·' if r!=c else '×'),ha='center',va='center',color=INK if observed else '#aaaaaa')
    ax.set_xticks(range(n),items);ax.set_yticks(range(n),items);ax.xaxis.tick_top();ax.tick_params(length=0)
    ax.set_title('$W$: seed (row) to candidate (column)',pad=26,fontsize=11)
    for name,ax,col,label in (('y',axes[1],y,'$y$: seeds {B, C}'),('p',axes[2],p,'$p=W^{\\top}y$')):
        ax.set_xlim(-.5,.5);ax.set_ylim(4.5,-.5);ax.set_aspect('equal')
        for r in range(n):
            ax.add_patch(Rectangle((-.5,r-.5),1,1,facecolor='#e8eef7' if name=='y' and col[r] else ('#dcefdc' if name=='p' and col[r] else 'white'),edgecolor='#999999'))
            ax.text(0,r,f'{col[r]:.0f}' if name=='y' else f'{col[r]:.1f}',ha='center',va='center')
        ax.set_xticks([]);ax.set_yticks(range(n),items);ax.tick_params(length=0);ax.set_title(label,pad=26,fontsize=11)
    fig.text(.5,-.06,'Toy bundles {A,B,C}, {B,D}, {C,D,E}. Weights are illustrative. Dots are pairs never seen in training: they have no parameter and stay exactly 0. Crosses are the unused diagonal.\nDarker rows are the seeds, so $p_E=w_{BE}+w_{CE}=0+1.3$. Seeds are removed from the ranking, whatever score they receive.',ha='center',va='top',fontsize=10)
    for ax in axes:
        for s in ax.spines.values():s.set_visible(False)
    fig.savefig(OUT/'pair_matrix.png',dpi=170,bbox_inches='tight');plt.close(fig)


def graph():
    fig,ax=canvas(12.5,4.7,12.5,4.5)
    box(ax,1.2,3.1,1.8,.8,'$z_j$\nneighbour $j$',INPUT);box(ax,1.2,2.1,1.8,.8,'$z_k$\nneighbour $k$',INPUT);box(ax,1.2,1.1,1.8,.8,'$z_l$\nneighbour $l$',INPUT)
    box(ax,4.1,2.1,2.0,1.0,'Unweighted mean\n$\\frac{1}{d_i}\\sum_{j\\sim i}z_j$',FIXED)
    box(ax,6.85,2.1,1.9,1.0,'Projection $P$\n$128\\times128$, no bias\nstarts at 0',LEARNED,size=9.5)
    box(ax,6.85,3.85,1.9,.7,'$z_i$  (item itself)',INPUT)
    box(ax,9.25,2.1,1.1,.7,'$+$',FIXED,size=14)
    box(ax,11.3,2.1,1.9,1.0,'$v_i$\nL2 normalized\nitem vector',OUTPUT)
    for y,yy in ((3.1,2.35),(2.1,2.1),(1.1,1.85)):arrow(ax,(2.1,y),(3.1,yy))
    arrow(ax,(5.1,2.1),(5.9,2.1));arrow(ax,(7.8,2.1),(8.7,2.1));arrow(ax,(7.8,3.85),(9.25,2.45));arrow(ax,(9.8,2.1),(10.35,2.1))
    ax.text(6.25,.18,'One layer only. Links count once however often a pair was seen, so there is no frequency input.\nAt the start $P=0$, so $v_i$ is just the normalized item vector, and history affects $V$ only as training moves $P$.',ha='center',va='center',fontsize=10)
    fig.savefig(OUT/'graph_layer.png',dpi=170,bbox_inches='tight');plt.close(fig)


def training():
    fig,ax=canvas(13,5.4,13,5.4)
    box(ax,1.25,4.5,2.0,.95,'Training bundle\n{a, b, c, d, e}',INPUT)
    box(ax,4.35,4.5,2.7,.95,'Random split\nseeds {a, c}\nhidden targets {b, d, e}',FIXED)
    box(ax,4.35,2.95,2.7,1.0,'Cold mask, per batch\nhide all history of a random\n25% of catalogue items',FIXED,size=9.8)
    box(ax,7.6,3.75,2.2,1.0,'Network\n$s=t+p$ with hidden\npaths removed',LEARNED)
    box(ax,10.95,3.75,3.3,1.15,'Per target $g$: softmax over\n$g$ and every item outside the\nbundle (no sampled negatives)',FIXED,size=9.8)
    box(ax,10.95,1.95,3.3,.95,'AdamW step, gradient clip 5\nthen clamp $w\\geq0$',LEARNED)
    box(ax,7.6,1.95,2.2,.95,'Every 2 epochs:\nvalidation NDCG@20',FIXED)
    box(ax,4.35,1.2,2.7,.95,'Keep the best epoch\nstop after 6 evaluations\nwithout improvement',OUTPUT,size=9.8)
    arrow(ax,(2.25,4.5),(3.0,4.5));arrow(ax,(5.7,4.5),(6.5,3.95));arrow(ax,(5.7,2.95),(6.5,3.55));arrow(ax,(8.7,3.75),(9.3,3.75));arrow(ax,(10.95,3.17),(10.95,2.43))
    arrow(ax,(9.3,1.95),(8.7,1.95));arrow(ax,(6.5,1.75),(5.7,1.4))
    ax.add_patch(FancyArrowPatch((9.75,2.43),(8.3,3.25),arrowstyle='-|>',mutation_scale=13,color=BLUE,linewidth=1.2,linestyle='--',connectionstyle='arc3,rad=.1'))
    ax.text(9.75,3.0,'next batch',color=BLUE,fontsize=9.5,ha='left')
    ax.text(.25,.25,'BundleFuse scores, weights and PPMI are not used anywhere in this loop. Test data is scored only after every seed has been frozen.',fontsize=10)
    fig.savefig(OUT/'training.png',dpi=170,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    overview();toy();graph();training()
