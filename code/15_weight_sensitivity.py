import pandas as pd,numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import hypergeom
d=pd.read_csv('../results/surface_proteome_annotation_support_v6.csv')
C=['focus_selectivity','all_selectivity','coverage_mean','malignant_specificity_all_surfaceome']
Z=(d[C]-d[C].mean())/d[C].std()
y=d.known_adc_target.astype(bool).values
w0=np.array([.35,.2,.2,.25])
s=Z.values@w0
print('corr with published score',np.corrcoef(s,d.surfaceome_score)[0,1])
def ev(s):
    r=(-s).argsort().argsort()
    top=r<int(round(0.1*len(s)))
    return roc_auc_score(y,s), int((top&y).sum())
print('base',ev(s),'published',ev(d.surfaceome_score.values))
res={}
res['equal']=ev(Z.values@np.ones(4)/4)
for i,c in enumerate(C):
    w=w0.copy();w[i]=0;w/=w.sum();res['drop_'+c]=ev(Z.values@w)
rng=np.random.default_rng(20261103)
W=rng.dirichlet(np.ones(4),5000)
A=[];T=[];topc=np.zeros(len(d))
for w in W:
    sc=Z.values@w;a,t=ev(sc);A.append(a);T.append(t)
for k,v in res.items():print(k,v)
A=np.array(A);T=np.array(T)
print('dirichlet AUROC median %.3f IQR %.3f-%.3f min %.3f; top10 median %d range %d-%d; frac AUROC>0.75 %.3f'%(np.median(A),*np.percentile(A,[25,75]),A.min(),np.median(T),T.min(),T.max(),(A>0.75).mean()))
# hypergeom vs matched null
N=len(d);K=18;n=313
print('hypergeom P top10 12/18', hypergeom.sf(11,N,K,n))
# leading candidates stability: top 8 non-panel in each dirichlet draw
cands=['CLDN4','MAL2','TMPRSS4','CLDN3','MUC1','CLDN7','RAB25','LSR']
idx=[d.index[d.gene==g][0] for g in cands]
cnt=np.zeros(len(cands));ranks={g:[] for g in cands}
for w in W[:2000]:
    r=(-(Z.values@w)).argsort().argsort()
    for j,i in enumerate(idx): ranks[cands[j]].append(r[i]+1)
for g in cands:
    a=np.array(ranks[g]);print(g,'median rank',np.median(a),'top5pct frac %.3f'%((a<=157).mean()))
