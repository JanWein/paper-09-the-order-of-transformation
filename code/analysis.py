"""Synthetic analysis only. Run from any directory: python code/analysis.py.
Python replaces unavailable R; the manuscript is authored in LaTeX.
All parameters are illustrative and have not been estimated from firms.
"""
from pathlib import Path
import json, platform
import numpy as np
import pandas as pd
from scipy.stats import t as student_t, norm
from scipy.optimize import brentq, minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output'
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(90172026)

def state(x, q0, q1, alpha):
    early = (1-alpha)*x + alpha*q0
    return (1-alpha)*early + alpha*q1, (1-alpha)*x + alpha*q1

def probes(x, q, c, lam):
    if x >= c: return 0
    n = int(np.ceil(np.log((q-c)/(q-x))/np.log(1-lam)))
    # Guard a possible floating-point boundary error.
    while q-(q-x)*(1-lam)**n < c-1e-12: n += 1
    return n

# Nontrivial algebra and boundary validation, independent of the Monte Carlo.
max_gap_error = 0.
for _ in range(10000):
    x,q0,q1,a = rng.uniform(size=4)
    xt,xf = state(x,q0,q1,a)
    max_gap_error=max(max_gap_error,abs((xt-xf)-a*(1-a)*(q0-x)))
    equal_trial_F=(1-a)*((1-a)*x+a*q1)+a*q1
    assert abs((xt-equal_trial_F)-a*(1-a)*(q0-q1))<1e-12
assert max_gap_error < 1e-12
for x in np.linspace(0,.59,60):
    n=probes(x,.8,.6,.2)
    assert .8-(.8-x)*.8**n >= .6-1e-12
    if n>0: assert .8-(.8-x)*.8**(n-1) < .6+1e-12

# Exact examples include a tools-first activation advantage, a scar, and no gap.
examples=[]
beta=.95; v=1.; H=24; e=.15; b=.05; kf=kt=1.
for label,x,q0,q1,a,c in [('scar',.65,.1,.8,.4,.6),
                        ('engagement',.2,.65,.8,.4,.46),
                        ('both_active',.7,.7,.8,.4,.6),
                        ('both_inactive',.1,.1,.8,.2,.6)]:
    xt,xf=state(x,q0,q1,a)
    at,af=int(xt>=c),int(xf>=c)
    annuity=(1-beta**H)/(1-beta)
    delta=e-b+(1-beta)*(kf-kt)+beta**2*v*annuity*(at-af)
    examples.append(dict(case=label,x0=x,q0=q0,q1=q1,alpha=a,c=c,
                         x_tools=xt,x_foundations=xf,adopt_tools=at,adopt_foundations=af,
                         npv_tools_minus_foundations=delta))
pd.DataFrame(examples).to_csv(OUT/'model_examples.csv',index=False)

# Recovery at independent probability epsilon per period, before production.
xs=examples[0]['x_tools']; n=probes(xs,.8,.6,.2)
eps_values=[0,.01,.05,.1,.25,.5,1.]
recovery=[]
for eps in eps_values:
    r=eps*beta/(1-(1-eps)*beta)
    recovery.append(dict(epsilon=eps,required_probes=n,
                         expected_wait=n/eps if eps else np.inf,
                         discounted_activation=r**n,
                         lost_continuation_value=v*(1-r**n)/(1-beta),
                         probe_cost_pv=.02*sum(r**j for j in range(1,n+1))))
pd.DataFrame(recovery).to_csv(OUT/'recovery.csv',index=False)

# Check the one-trial capacity optimum against independent scalar minimization.
capacity=[]
for K in [.1,.5,1.,5.,20.,100.]:
    d=.02; A=beta; B=1-beta
    def loss(eps):
        return d+(v-d*B)/(B+A*eps)+K*eps**2
    threshold=A*(v-d*B)/2
    optimum=1. if K<=threshold else brentq(lambda eps:2*K*eps*(B+A*eps)**2-A*(v-d*B),0.,1.)
    numeric=minimize_scalar(loss,bounds=(0,1),method='bounded')
    assert abs(loss(optimum)-numeric.fun)<1e-4
    capacity.append(dict(capacity_cost_K=K,optimal_epsilon=optimum,
                         expected_wait=1/optimum,optimized_loss=loss(optimum)))
pd.DataFrame(capacity).to_csv(OUT/'optimal_capacity.csv',index=False)
# Independent stochastic check of E[beta^tau_n], including n>1.
recovery_checks=[]
for n_check in [1,3,7]:
    eps=.2
    tau=rng.geometric(eps,size=(200000,n_check)).sum(1)
    draws=beta**tau
    analytic=(eps*beta/(1-(1-eps)*beta))**n_check
    error=draws.mean()-analytic
    assert abs(error)<5*draws.std(ddof=1)/np.sqrt(len(draws))
    recovery_checks.append(dict(n=n_check,analytic=analytic,simulated=draws.mean(),
                                mcse=draws.std(ddof=1)/np.sqrt(len(draws))))
pd.DataFrame(recovery_checks).to_csv(OUT/'recovery_validation.csv',index=False)

# Design simulation. Unit is a randomized team, never a team-week row.
# Z=1 foundations-first. Both bundles complete before week 9.
# Tool availability is weeks 1-16 for T, 5-16 for F.
R=2000
times=np.arange(1,17)
window=np.arange(12,16) # weeks 13-16, declared before simulation
scenarios={'null':(0.,0.,1.),'duration_only':(.6,0.,1.),
           'transient_carryover':(.6,.25,.65),'persistent_carryover':(.6,.25,1.)}
rows=[]
for N in [48,96,192]:
    for scenario,(gamma,theta,rho) in scenarios.items():
        z=np.zeros(N); z[N//2:]=1
        exposure=times[None,:]-4*z[:,None]
        exposure=np.maximum(exposure,0)
        skill=gamma*(1-np.exp(-exposure/4))
        scar=theta*z[:,None]*rho**np.maximum(times[None,:]-9,0)
        mean=.03*times[None,:]+skill+scar
        true=(mean[z==1][:,window].mean()-mean[z==0][:,window].mean())
        # stationary AR(1) innovations; team heterogeneity variance .64
        u=rng.normal(0,.8,size=(R,N,1))
        noise=np.empty((R,N,16)); noise[:,:,0]=rng.normal(size=(R,N))
        for j in range(1,16):
            noise[:,:,j]=.6*noise[:,:,j-1]+rng.normal(0,.8,size=(R,N))
        y=mean[None,:,:]+u+noise
        w=y[:,:,window].mean(axis=2)
        a,bv=w[:,N//2:],w[:,:N//2]
        estimate=a.mean(1)-bv.mean(1)
        va=a.var(1,ddof=1)/(N//2); vb=bv.var(1,ddof=1)/(N//2)
        se=np.sqrt(va+vb)
        df=(va+vb)**2/(va**2/(N//2-1)+vb**2/(N//2-1))
        crit=student_t.ppf(.975,df)
        # Deliberately invalid independence calculation on team-week rows.
        fa=y[:,N//2:][:,:,window].reshape(R,-1)
        fb=y[:,:N//2][:,:,window].reshape(R,-1)
        naive_se=np.sqrt(fa.var(1,ddof=1)/fa.shape[1]+fb.var(1,ddof=1)/fb.shape[1])
        # Full-case estimator under assignment-dependent outcome-related loss.
        pkeep=1/(1+np.exp(-(.9+.5*w-.6*z[None,:])))
        keep=rng.uniform(size=w.shape)<pkeep
        ca=(a*keep[:,N//2:]).sum(1)/keep[:,N//2:].sum(1)
        cb=(bv*keep[:,:N//2]).sum(1)/keep[:,:N//2].sum(1)
        rows.append(dict(N=N,scenario=scenario,true_policy_effect=true,
                         mean_estimate=estimate.mean(),bias=estimate.mean()-true,
                         empirical_sd=estimate.std(ddof=1),mean_se=se.mean(),
                         coverage=np.mean(abs(estimate-true)<=crit*se),
                         rejection_zero=np.mean(abs(estimate)>crit*se),
                         naive_coverage=np.mean(abs(estimate-true)<=1.96*naive_se),
                         naive_rejection_zero=np.mean(abs(estimate)>1.96*naive_se),
                         complete_case_bias=(ca-cb).mean()-true,
                         mcse_coverage=np.sqrt(.95*.05/R)))
res=pd.DataFrame(rows);res.to_csv(OUT/'design_simulation.csv',index=False)
table=['\\begin{table}[ht]\\centering\\small',
       '\\caption{Synthetic experiments: selected results at 96 independent teams.}',
       '\\begin{tabular}{lrrrr}\\toprule',
       'Scenario & Truth & Coverage & Power / size & Naive coverage\\\\\\midrule']
for _,row in res[res.N==96].iterrows():
    label=row.scenario.replace('_',' ')
    table.append(f'{label} & {row.true_policy_effect:.3f} & {row.coverage:.3f} & {row.rejection_zero:.3f} & {row.naive_coverage:.3f}\\\\')
table += ['\\bottomrule\\end{tabular}',
          '\\par\\smallskip\\parbox{.92\\linewidth}{\\footnotesize Coverage refers to the known policy truth. Power / size is rejection of zero; only the null row measures size. Naive intervals ignore within-team dependence. Each cell uses 2,000 synthetic experiments.}',
          '\\end{table}']
(OUT/'simulation_table.tex').write_text('\n'.join(table))

# Large-sample planning sensitivity, no pilot-based justification is implied.
precision=[]
for N in [48,96,192]:
    for baseline_R2 in [0,.4]:
        # Variance of mean of four stationary AR1 observations plus team effect.
        variance=.64+(4+2*sum((4-k)*.6**k for k in range(1,4)))/16
        se=np.sqrt(4*variance*(1-baseline_R2)/N)
        precision.append(dict(N=N,baseline_R2=baseline_R2,se=se,
                              mde_80=(norm.ppf(.975)+norm.ppf(.8))*se,
                              halfwidth_95=1.96*se))
pd.DataFrame(precision).to_csv(OUT/'precision.csv',index=False)

# Visual mechanism comparison. Identical final resources do not ensure equal use.
fig,ax=plt.subplots(1,2,figsize=(10,3.7))
for j,case in enumerate(examples[:2]):
    for label,xstart,colour in [('Tools first',case['x_tools'],'#ac4d31'),
                                 ('Foundations first',case['x_foundations'],'#176a7a')]:
        vals=[xstart]
        for t in range(20):
            x=vals[-1]; vals.append(.8*x+.2*case['q1'] if x>=case['c'] else x)
        ax[j].plot(range(21),vals,label=label,color=colour,lw=2)
    ax[j].axhline(case['c'],color='grey',ls=':',label='Use threshold')
    ax[j].set(title=case['case'].capitalize(),xlabel='Common post-completion periods',ylabel='Willingness state',ylim=(0,1))
ax[0].legend(fontsize=8);fig.tight_layout();fig.savefig(ROOT/'figures/mechanism.pdf');plt.close(fig)

meta={'seed':90172026,'replications_per_cell':R,'python':platform.python_version(),
      'numpy':np.__version__,'pandas':pd.__version__,
      'max_algebra_error':max_gap_error,'probe_boundary_checks':60,
      'real_observations':0,'interpretation':'All numerical outputs are synthetic.'}
(OUT/'run_metadata.json').write_text(json.dumps(meta,indent=2))
print(res.round(4).to_string(index=False))
print(pd.DataFrame(precision).round(4).to_string(index=False))
print(pd.DataFrame(examples).round(4).to_string(index=False))
print(pd.DataFrame(recovery).round(4).to_string(index=False))
