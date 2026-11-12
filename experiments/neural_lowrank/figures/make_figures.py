"""Draw the figures in ../ARCHITECTURE.md: python experiments/neural_lowrank/figures/make_figures.py"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,FancyBboxPatch,Rectangle
import numpy as np

OUT=Path(__file__).parent
INPUT='#e8eef7';LEARNED='#fde8c8';FIXED='#eeeeee';OUTPUT='#dcefdc';INK='#222222';BLUE='#2f6db5';ORANGE='#c8701a';RED='#b8322a'
plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans'})


def box(ax,x,y,w,h,text,color=FIXED,size=10.5,edge=INK):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.02,rounding_size=0.08',facecolor=color,edgecolor=edge,linewidth=1.1))
    ax.text(x,y,text,ha='center',va='center',fontsize=size,color=INK,linespacing=1.35)


def arrow(ax,a,b,color=INK,rad=0.,ls='-'):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color=color,linewidth=1.3,linestyle=ls,connectionstyle=f'arc3,rad={rad}',shrinkA=0,shrinkB=0))


def canvas(w,h):
    fig,ax=plt.subplots(figsize=(w,h));ax.set_xlim(0,w);ax.set_ylim(0,h);ax.axis('off');return fig,ax


def legend(ax,x,y,dx=2.3):
    for i,(c,t) in enumerate(((INPUT,'input'),(LEARNED,'trainable'),(FIXED,'fixed operation'),(OUTPUT,'output'))):
        ax.add_patch(Rectangle((x+i*dx,y-.11),.3,.22,facecolor=c,edgecolor=INK,linewidth=.8))
        ax.text(x+i*dx+.4,y,t,va='center',fontsize=9.5)


def overview():
    fig,ax=canvas(14,8.2)
    ax.set_xlim(-.4,14)
    ax.text(.25,7.9,'Three views of every item, computed once for the whole catalogue',fontsize=11,weight='bold')
    box(ax,1.3,6.9,2.2,.85,'Content features $x_i$\n128-d, catalogue only',INPUT,size=10)
    box(ax,1.3,5.6,2.2,.85,'Training bundles\ncounts $n_{ij}$, $f_i$',INPUT,size=10)
    box(ax,1.3,4.3,2.2,.85,'Group $g(i)$\ncategory or artist',INPUT,size=10)
    box(ax,4.5,6.9,2.6,.85,'Content $z_i$\n$\\mathrm{norm}(x_i+f(x_i))$',LEARNED,size=10)
    box(ax,4.5,5.6,2.6,.85,'History $h_i$\n$\\sum_j n_{ij}x_j\\,/\\sum_j n_{ij}$',FIXED,size=10)
    box(ax,4.5,4.3,2.6,.85,'Group embedding\n$e[g(i)]$, 16-d',LEARNED,size=10)
    arrow(ax,(2.4,6.9),(3.2,6.9));arrow(ax,(2.4,5.6),(3.2,5.6));arrow(ax,(2.4,4.3),(3.2,4.3))
    arrow(ax,(1.3,6.47),(3.2,5.75),color='#888888')
    # query and key projections
    box(ax,8.3,6.55,3.2,1.45,'Query side, per seed $s$\n$A z_s + B h_s + U_q e_q[g(s)]$\n$A,B$: $64\\times128$  $U_q$: $64\\times16$',LEARNED,size=10)
    box(ax,8.3,4.6,3.2,1.45,'Key side, per candidate $c$\n$K z_c + D h_c + U_k e_k[g(c)]$\n$K,D$: $64\\times128$  $U_k$: $64\\times16$',LEARNED,size=10)
    for y in (6.9,5.6,4.3):
        arrow(ax,(5.8,y),(6.7,6.4 if y>5 else 6.2),color=BLUE)
        arrow(ax,(5.8,y),(6.7,4.75 if y<6 else 4.95),color=ORANGE)
    box(ax,12.2,6.55,2.6,.95,'$\\varphi(S)=$ mean over seeds\n64-d query vector',FIXED,size=10)
    box(ax,12.2,4.6,2.6,.95,'$\\psi$: one 64-d key\nper catalogue item',FIXED,size=10)
    arrow(ax,(9.9,6.55),(10.9,6.55));arrow(ax,(9.9,4.6),(10.9,4.6))
    ax.text(1.55,3.35,'Scoring one query',fontsize=11,weight='bold')
    box(ax,3.0,2.3,3.8,1.1,'Pair term, one shared network\n$\\sum_{s\\in S}\\omega(\\log(1{+}n_{sc}),\\log(1{+}f_s),\\log(1{+}f_c))$\nzero where $n_{sc}=0$; 81 parameters',LEARNED,size=9.8)
    box(ax,8.3,2.3,3.2,1.1,'Relation term\n$\\varphi(S)^{\\top}\\psi(c)$\nrank 64, asymmetric',FIXED,size=10)
    arrow(ax,(10.9,6.3),(9.4,2.85),color=BLUE);arrow(ax,(12.2,4.12),(9.9,2.75),color=ORANGE)
    box(ax,11.2,1.0,1.6,.75,'$+$',FIXED,size=14)
    box(ax,13.1,1.0,1.4,1.0,'Rank,\ndrop seeds,\ntop 20',OUTPUT,size=9.5)
    arrow(ax,(4.9,1.75),(10.4,1.05));arrow(ax,(9.9,1.75),(10.4,1.15));arrow(ax,(12.0,1.0),(12.4,1.0))
    arrow(ax,(.2,5.6),(.9,2.85),color='#888888',rad=.45)
    legend(ax,.25,.25)
    fig.savefig(OUT/'overview.png',dpi=170,bbox_inches='tight');plt.close(fig)


def blocks():
    """The relation term is one bilinear form; its 3x3 blocks are the interactions it can express."""
    rows=['content $z_s$','history $h_s$','group $e_q[g(s)]$'];cols=['content $z_c$','history $h_c$','group $e_k[g(c)]$']
    text=[['$z_s^{\\top}A^{\\top}K z_c$\nlooks like / goes with\n(asymmetric, works cold)','$z_s^{\\top}A^{\\top}D h_c$\nseed resembles the\ncandidate\'s partners','$z_s^{\\top}A^{\\top}U_k e_k$\nseed content predicts\ncandidate group'],
          ['$h_s^{\\top}B^{\\top}K z_c$\ncandidate resembles the\nseed\'s partners (style)','$h_s^{\\top}B^{\\top}D h_c$\nshared partners\n(second order)','$h_s^{\\top}B^{\\top}U_k e_k$\nseed partners predict\ncandidate group'],
          ['$e_q^{\\top}U_q^{\\top}K z_c$\nseed group predicts\ncandidate content','$e_q^{\\top}U_q^{\\top}D h_c$\nseed group predicts\ncandidate partners','$e_q^{\\top}U_q^{\\top}U_k e_k$\ngroup-to-group table,\nrank 16']]
    cold=[[1,0,1],[1,0,1],[1,0,1]]
    fig,ax=plt.subplots(figsize=(12.5,6.4));ax.set_xlim(-1.6,3);ax.set_ylim(3.2,-.9);ax.axis('off')
    for r in range(3):
        ax.text(-.08,r+.5,rows[r],ha='right',va='center',fontsize=11)
        for c in range(3):
            color=OUTPUT if cold[r][c] else '#f6f6f6'
            if r==2 and c==2:color=LEARNED
            ax.add_patch(Rectangle((c,r),1,1,facecolor=color,edgecolor='#888888'))
            ax.text(c+.5,r+.5,text[r][c],ha='center',va='center',fontsize=9.6,linespacing=1.4)
    for c in range(3):ax.text(c+.5,-.12,cols[c],ha='center',va='bottom',fontsize=11)
    ax.text(1.5,-.62,'candidate $c$ (key side)',ha='center',fontsize=11,weight='bold')
    ax.text(-1.5,1.5,'seed $s$\n(query side)',ha='center',va='center',fontsize=11,weight='bold')
    ax.text(1.5,3.12,'Green: available when the candidate has never been sold, because $z_c$ and $g(c)$ need no history.\n'
            'Grey: needs candidate history, so it is zero for cold candidates. Orange: replaces the full category table.',ha='center',va='top',fontsize=10)
    fig.savefig(OUT/'blocks.png',dpi=170,bbox_inches='tight');plt.close(fig)


def leave_one_out():
    items='ABCDEF';n=6
    bundles=[('A','B','C'),('A','B'),('B','D'),('C','D','E'),('C','E','F')]
    current=('C','E','F')
    N=np.zeros((n,n),int)
    for b in bundles:
        for a in b:
            for c in b:
                if a!=c:N[items.index(a),items.index(c)]+=1
    own=np.zeros((n,n),int)
    for a in current:
        for c in current:
            if a!=c:own[items.index(a),items.index(c)]=1
    fig,axes=plt.subplots(1,3,figsize=(13,4.6),gridspec_kw=dict(width_ratios=[1,1,.55],wspace=.35))
    for ax,M,title in ((axes[0],N,'All training counts $n_{ij}$'),(axes[1],N-own,'Counts while scoring {C, E, F}')):
        ax.set_xlim(-.5,n-.5);ax.set_ylim(n-.5,-.5);ax.set_aspect('equal')
        for r in range(n):
            for c in range(n):
                hit=own[r,c] and M is not N
                ax.add_patch(Rectangle((c-.5,r-.5),1,1,facecolor='#f4c2bd' if hit else ('#fde8c8' if M[r,c] else 'white'),edgecolor='#aaaaaa'))
                ax.text(c,r,'×' if r==c else (str(M[r,c]) if M[r,c] or hit else '·'),ha='center',va='center',color=RED if hit else (INK if M[r,c] else '#aaaaaa'))
        ax.set_xticks(range(n),items);ax.set_yticks(range(n),items);ax.xaxis.tick_top();ax.tick_params(length=0);ax.set_title(title,pad=24,fontsize=11)
        for s in ax.spines.values():s.set_visible(False)
    f=np.array([sum(i in b for b in bundles) for i in items]);floo=f-np.array([i in current for i in items])
    ax=axes[2];ax.set_xlim(-.5,1.5);ax.set_ylim(n-.5,-.5);ax.set_aspect('equal')
    for r in range(n):
        for c,v in enumerate((f[r],floo[r])):
            ax.add_patch(Rectangle((c-.5,r-.5),1,1,facecolor=('#f4c2bd' if c and v==0 and f[r] else (OUTPUT if v else 'white')),edgecolor='#aaaaaa'))
            ax.text(c,r,str(v),ha='center',va='center')
    ax.set_xticks([0,1],['$f_i$','$f_i$ left\nout']);ax.set_yticks(range(n),items);ax.xaxis.tick_top();ax.tick_params(length=0);ax.set_title('Frequencies',pad=24,fontsize=11)
    for s in ax.spines.values():s.set_visible(False)
    fig.text(.5,-.04,'Training bundles {A,B,C}, {A,B}, {B,D}, {C,D,E}, {C,E,F}. When {C,E,F} is the training query, its own pairs lose one count (red).\n'
             'C–E falls from 2 to 1 because another bundle also contains it. C–F and E–F fall to 0, as for a pair never seen in training.\n'
             'F appears in no other bundle, so it has no history left and trains the never-sold path. '
             'The history views $h_C$, $h_E$, $h_F$ get the same subtraction. At test time nothing is removed.',ha='center',va='top',fontsize=10)
    fig.savefig(OUT/'leave_one_out.png',dpi=170,bbox_inches='tight');plt.close(fig)


def parameters():
    data=[('POG_DENSE',177811,182300,45393),('POG',163011,168195,45553),('Spotify_sparse',27344217,27344217+43039**2,1420497),('Spotify',67188723,67188723+49164**2,1616497)]
    fig,ax=plt.subplots(figsize=(10,4.4));x=np.arange(len(data));w=.26
    for i,(label,color) in enumerate((('Original (pair table)','#bbbbbb'),('Original + full category table','#e3a35c'),('Low-rank model','#4f8fd6'))):
        v=[d[i+1] for d in data];bars=ax.bar(x+(i-1)*w,v,w,label=label,color=color)
        for b,val in zip(bars,v):ax.text(b.get_x()+b.get_width()/2,val*1.15,f'{val/1e9:.2f}B' if val>=1e9 else (f'{val/1e6:.2f}M' if val>=1e6 else f'{val/1e3:.0f}k'),ha='center',fontsize=8.5)
    ax.set_yscale('log');ax.set_xticks(x,[d[0] for d in data]);ax.set_ylabel('Trainable parameters (log scale)');ax.set_ylim(1e4,1e11)
    ax.legend(frameon=False,fontsize=9.5,loc='upper left');ax.spines[['top','right']].set_visible(False)
    ax.set_title('Older models on Spotify: computed from pair and artist counts, never run',fontsize=9.5)
    fig.savefig(OUT/'parameters.png',dpi=170,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    overview();blocks();leave_one_out();parameters()
