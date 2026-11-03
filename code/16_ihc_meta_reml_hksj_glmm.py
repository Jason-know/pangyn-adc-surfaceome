import pandas as pd,numpy as np
from scipy import optimize,stats
from scipy.special import expit
d=pd.read_csv('../results/ihc_meta_primary_inputs_v6.csv')
P=pd.read_csv('../results/ihc_meta_primary_v6.csv')
gh_x,gh_w=np.polynomial.hermite.hermgauss(40)
out=[]
for _,row in P.iterrows():
    s=d[(d.target==row.target)&((d.threshold_class==row.analysis_threshold)|(row.analysis_threshold=='Mixed thresholds'))]
    e=s.n_positive.values.astype(float);n=s.n_total.values.astype(float)
    y=np.log((e+.5)/(n-e+.5));v=1/(e+.5)+1/(n-e+.5);k=len(y)
    assert k==row.k,(row.target,k)
    def nll(t2):
        w=1/(v+t2);mu=(w*y).sum()/w.sum()
        return .5*(np.log(v+t2).sum()+np.log(w.sum())+(w*(y-mu)**2).sum())
    t2=optimize.minimize_scalar(nll,bounds=(0,50),method='bounded').x
    if nll(0)<=nll(t2): t2=0.0
    w=1/(v+t2);mu=(w*y).sum()/w.sum()
    q=(w*(y-mu)**2).sum()/(k-1);se=np.sqrt(max(q,1)*1/w.sum())  # HKSJ with q>=1 modification... record both
    se_h=np.sqrt(q/w.sum())
    tc=stats.t.ppf(.975,k-1)
    ci=expit([mu-tc*se_h,mu+tc*se_h])
    pi=expit([mu-tc*np.sqrt(t2+1/w.sum()),mu+tc*np.sqrt(t2+1/w.sum())]) if k>=3 else [np.nan]*2
    # GLMM binomial-normal
    def gl(p):
        m,lt=p;tau=np.exp(lt);ll=0
        for ei,ni in zip(e,n):
            th=m+np.sqrt(2)*tau*gh_x
            ll+=np.log((gh_w/np.sqrt(np.pi)*stats.binom.pmf(ei,ni,expit(th))).sum()+1e-300)
        return -ll
    r=optimize.minimize(gl,[mu,np.log(np.sqrt(t2)+.1)],method='Nelder-Mead',options={'xatol':1e-6,'fatol':1e-8,'maxiter':4000})
    out.append(dict(target=row.target,threshold=row.analysis_threshold,k=k,DL=round(row.pooled_prevalence*100,1),DL_ci=f"{row.ci_low*100:.1f}-{row.ci_high*100:.1f}",REML_HKSJ=round(expit(mu)*100,1),REML_ci=f"{ci[0]*100:.1f}-{ci[1]*100:.1f}",PI=f"{pi[0]*100:.1f}-{pi[1]*100:.1f}",tau2=round(t2,3),GLMM=round(expit(r.x[0])*100,1)))
o=pd.DataFrame(out);print(o.to_string());o.to_csv('../results/ihc_meta_REML_HKSJ_GLMM.csv',index=False)
