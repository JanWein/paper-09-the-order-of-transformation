"""R1 exact rational-learning model. All examples are synthetic, not calibrated.
Run: python code/rational_learning.py. No network or operational data are used.
The legacy analysis.py is a separate behavioral/design benchmark.
"""
from pathlib import Path
import itertools,json,platform
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'output'
OUT.mkdir(exist_ok=True)

def posterior(p,H,L,s):
    lh=H if s else 1-H; ll=L if s else 1-L
    prob=p*lh+(1-p)*ll
    return (prob,p*lh/prob if prob else p)

def mean(p,H,L,c): return p*H+(1-p)*L-c

def v2(p,H=.9,L=.1,c=.5,delta=.9):
    m=mean(p,H,L,c)
    ev=sum(pr*max(mean(pp,H,L,c),0.) for pr,pp in
           (posterior(p,H,L,s) for s in [0,1]))
    return max(0.,m+delta*ev)

def cutoff(H=.9,L=.1,c=.5,delta=.9):
    h=H-c;l=L-c
    return (-l)*(1+delta*L)/((H-L)+delta*(H*h-L*l))

def brute_v2(p,H,L,c,delta):
    # Independently enumerate all root actions and terminal action mappings.
    values=[0.,delta*max(mean(p,H,L,c),0.)]
    for actions in itertools.product([0,1],repeat=2):
        value=mean(p,H,L,c)
        for s in [0,1]:
            pr,pp=posterior(p,H,L,s)
            value+=delta*pr*actions[s]*mean(pp,H,L,c)
        values.append(value)
    return max(values)

def W(p,k,H=.9,L=.1,c=.5,delta=.9):
    B=(1+delta)*(H-c)
    return max(v2(p,H,L,c,delta),p*B-k)

def w_action(p,k):
    vv=v2(p);cert=p*.76-k
    if max(vv,cert)<=1e-12:return 'abstain'
    return 'validate' if cert>vv+1e-12 else 'experiment'

def policies(p,K,k,zeta=0.,dT=.01,dF=.01,H=.9,L=.1,aH=.6,aL=.08,c=.5,delta=.9):
    abar=p*aH+(1-p)*aL
    A_H=(1-zeta)*aH+zeta*abar
    A_L=(1-zeta)*aL+zeta*abar
    ea=0.; stop_option=0.; wt=-dT; eb=0.
    for s in [0,1]:
        pr,pp=posterior(p,A_H,A_L,s)
        w=W(pp,k,H,L,c,delta)
        ea+=pr*w;wt+=pr*max(w-K,0.);stop_option+=pr*max(K-w,0.)
        pr,pp=posterior(p,H,L,s);eb+=pr*W(pp,k,H,L,c,delta)
    wf=-K-dF+eb
    direct=max(W(p,k,H,L,c,delta)-K,0.)
    values={'no_repair':0.,'direct_repair':direct,
            'tools_first':wt,'foundations_first':wf}
    # Tie at zero is deliberately assigned to no repair.
    choice=max(values,key=values.get)
    return dict(prior=p,K=K,k=k,zeta=zeta,**values,choice=choice,
                screening_option=stop_option,diagnostic_advantage=eb-ea,
                decomposition_error=(wt-wf)-(dF-dT+stop_option-(eb-ea)))

checks={};rng=np.random.default_rng(9172026)
max_brute_error=0.;max_decomp_error=0.;max_posterior_error=0.
for _ in range(3000):
    L=rng.uniform(.01,.4);H=rng.uniform(.6,.99);c=rng.uniform(L+.01,H-.01)
    delta=rng.uniform(.01,.99);p=rng.uniform();pstar=cutoff(H,L,c,delta)
    assert v2(pstar,H,L,c,delta)<1e-12
    assert v2(min(1,pstar+1e-6),H,L,c,delta)>0
    err=abs(v2(p,H,L,c,delta)-brute_v2(p,H,L,c,delta))
    max_brute_error=max(max_brute_error,err)
    assert err<1e-12
for _ in range(3000):
    p=rng.uniform(.001,.999);K=rng.uniform(0,1);k=rng.uniform(0,.5);z=rng.uniform()
    row=policies(p,K,k,z)
    max_decomp_error=max(max_decomp_error,abs(row['decomposition_error']))
    assert abs(row['decomposition_error'])<1e-12
    # This parametrization has a garbling of b: the diagnostic advantage is >=0.
    assert row['diagnostic_advantage']>=-1e-12
    assert row['direct_repair']>=0
    # Check a^z likelihood against full old-type/final-type enumeration.
    abar=p*.6+(1-p)*.08
    for s in [0,1]:
        pr_old,q=posterior(p,.6,.08,s)
        pr_final,pp=posterior(p,(1-z)*.6+z*abar,(1-z)*.08+z*abar,s)
        error=max(abs(pr_old-pr_final),abs(pp-(z*p+(1-z)*q)))
        max_posterior_error=max(max_posterior_error,error);assert error<1e-12

# General value-of-extra-information inequality and reset null at fixed repair.
for p in np.linspace(.01,.99,99):
    for k in [0.,.02,.15,.35]:
        extra=sum(pr*W(pp,k) for pr,pp in (posterior(p,.6,.08,s) for s in [0,1]))
        assert extra>=W(p,k)-1e-12
        reset=policies(p,.2,k,zeta=1.)
        assert abs(reset['screening_option']-max(.2-W(p,k),0))<1e-12

# Independently solve finite-horizon use/certificate trees approaching the
# infinite-horizon sufficient stopping example. A certificate is allowed at
# every date in this check, unlike the main two-task entry-only specification.
def finite_repeated_certificate(p,k,horizon,delta=.9,H=.9,L=.1,c=.5):
    @lru_cache(None)
    def solve(remaining,succ,fail):
        if remaining==0:return 0.
        odds=(p/(1-p))*(H/L)**succ*((1-H)/(1-L))**fail
        pp=odds/(1+odds);pred=pp*H+(1-pp)*L
        use=mean(pp,H,L,c)+delta*(pred*solve(remaining-1,succ+1,fail)
                              +(1-pred)*solve(remaining-1,succ,fail+1))
        cert=pp*(H-c)*(1-delta**remaining)/(1-delta)-k
        return max(0.,use,cert)
    return solve(horizon,0,0)
stop_p=(.4/.92)**3/(1+(.4/.92)**3)
for horizon in [1,2,5,10,30,60]:
    assert finite_repeated_certificate(stop_p,.35,horizon)==0.

# Convexity and exact certificate-region endpoint checks on a dense grid.
grid_check=np.linspace(0,1,10001)
vcheck=np.array([v2(p) for p in grid_check])
assert np.diff(vcheck,n=2).min()>-1e-12
assert np.min(.76*grid_check-vcheck)>-1e-12

examples=[policies(.5,K,k,zeta=z) for K,k,z in
          [(.1,.15,0),(.2,.15,0),(.2,.02,0),(.8,.15,0),(.2,.15,1)]]
pd.DataFrame(examples).to_csv(OUT/'rational_policy_examples.csv',index=False)

histories=[]
for nfail in [0,1,2,3,4]:
    odds=(.4/.92)**nfail;p=odds/(1+odds)
    inf_bound=mean(p,.9,.1,.5)+.9*p*(.4/.1)
    for z in [0.,.5,1.]:
        pp=z*.5+(1-z)*p
        histories.append(dict(failures=nfail,zeta=z,old_posterior=p,
           final_posterior=pp,v2=v2(pp),action_k035=w_action(pp,.35),
           infinite_first_use_upper=mean(pp,.9,.1,.5)+.9*pp*4.,
           infinite_certificate_net=pp*4.-.35))
pd.DataFrame(histories).to_csv(OUT/'rational_histories.csv',index=False)

# Exact entry-validation regions: W=max(V2,pB-k), no fitted latent trust state.
pgrid=np.linspace(0,1,1001)
validation=pd.DataFrame({'p':pgrid,'v2':[v2(p) for p in pgrid],
                         'full_information':.76*pgrid})
validation['information_value']=validation.full_information-validation.v2
validation.to_csv(OUT/'rational_validation_frontier.csv',index=False)

Kgrid=np.linspace(0,.65,101);kgrid=np.linspace(0,.4,101)
policy_rows=[];ids={'no_repair':0,'direct_repair':1,'tools_first':2,'foundations_first':3}
matrix=np.zeros((len(kgrid),len(Kgrid)),dtype=int)
for j,k in enumerate(kgrid):
    for i,K in enumerate(Kgrid):
        row=policies(.5,K,k);policy_rows.append(row);matrix[j,i]=ids[row['choice']]
pd.DataFrame(policy_rows).to_csv(OUT/'rational_policy_grid.csv',index=False)

fig,ax=plt.subplots(1,2,figsize=(10.2,3.9))
ax[0].plot(pgrid,validation.v2,label='Endogenous use / experimentation',color='#176a7a')
ax[0].plot(pgrid,.76*pgrid-.15,label='Perfect certificate less fee k=.15',color='#ac4d31')
ax[0].axhline(0,color='grey',lw=.8);ax[0].axvline(cutoff(),color='grey',ls=':')
ax[0].set(xlabel='Posterior of effective persistent type',ylabel='Two-task expected net surplus',ylim=(-.17,.8))
ax[0].legend(fontsize=7,loc='upper left')
im=ax[1].imshow(matrix,origin='lower',aspect='auto',extent=[0,.65,0,.4],
                cmap=ListedColormap(['#dddddd','#e6b657','#176a7a','#ac4d31']),vmin=-.5,vmax=3.5)
ax[1].set(xlabel='Physical remediation cost K',ylabel='Credible validation fee k')
cb=fig.colorbar(im,ax=ax[1],ticks=[0,1,2,3],fraction=.05)
cb.ax.set_yticklabels(['No repair','Direct repair','Tools first','Foundations first'],fontsize=7)
fig.tight_layout();fig.savefig(ROOT/'figures/rational_frontier.pdf');plt.close(fig)

table=['\\begin{table}[ht]\\centering\\small',
       '\\caption{Solved synthetic policy values, common prior .5 and equal pilot cost .01.}',
       '\\begin{tabular}{rrrrrrl}\\toprule',
       '$K$ & $k$ & $\\zeta$ & Direct & Tools first & Foundations first & Optimum\\\\\\midrule']
for r in examples:
    label=r['choice'].replace('_',' ')
    table.append(f"{r['K']:.2f} & {r['k']:.2f} & {r['zeta']:.0f} & {r['direct_repair']:.4f} & {r['tools_first']:.4f} & {r['foundations_first']:.4f} & {label}\\\\")
table+=['\\bottomrule\\end{tabular}',
        '\\par\\smallskip\\footnotesize No repair always has value zero. Direct denotes the ex-ante optimized direct-repair/no-repair menu. Every value is synthetic, in normalized surplus units.',
        '\\end{table}']
(OUT/'rational_policy_table.tex').write_text('\n'.join(table))
meta=dict(seed=9172026,random_bellman_checks=3000,random_policy_checks=3000,
          extra_information_checks=396,posterior_mapping_max_error=max_posterior_error,
          bellman_max_error=max_brute_error,decomposition_max_error=max_decomp_error,
          pstar=cutoff(),finite_horizon_tasks=2,real_observations=0,
          entry_certificate_only=True,infinite_bound_finite_checks=[1,2,5,10,30,60],
          convexity_grid_points=10001,python=platform.python_version())
(OUT/'rational_verification.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2));print(pd.DataFrame(examples).round(6).to_string(index=False))
