"""R2 exact Fraction arithmetic checks. Synthetic, no data, no network.
The analytic proof is in cost_regions.tex. Finite boundary/interior checks below
are verification, not a numerical substitute for that proof.
"""
from fractions import Fraction as F
from pathlib import Path
from itertools import combinations, product
import csv,json,platform
P=Path(__file__).resolve().parents[1]; O=P/'output';O.mkdir(exist_ok=True)
f=lambda x:F(str(x))
p0=F(1,2);aH=F(3,5);aL=F(2,25);H=F(9,10);L0=F(1,10);c=F(1,2);delta=F(9,10)
t=F(17,50);r=1-t;B=F(19,25);u=F(2497,4250);v=F(57,85);w=F(38,165);n=F(38,125)
kb=F(19,250);kp=F(353,4250);km=w;k0=F(59,250);kd=F(916,4125)
def post(p,h,l,s):
    lh=h if s else 1-h;ll=l if s else 1-l
    pr=p*lh+(1-p)*ll
    return pr,p*lh/pr

def V(p,h=H,l=L0,cc=c,de=delta):
    m=lambda q:l+(h-l)*q-cc
    return max(F(0),m(p)+de*sum(pr*max(F(0),m(q)) for pr,q in [post(p,h,l,s) for s in (0,1)]))
def W(p,k,h=H,l=L0,cc=c,de=delta):return max(V(p,h,l,cc,de),p*(1+de)*(h-cc)-k)
def vals(k,z=F(0),prior=p0,ah=aH,al=aL,h=H,l=L0,cc=c,de=delta):
    tt=prior*ah+(1-prior)*al;rr=1-tt
    qp=prior*ah/tt;qm=prior*(1-ah)/rr
    lo=W(prior+(1-z)*(qm-prior),k,h,l,cc,de)
    up=W(prior+(1-z)*(qp-prior),k,h,l,cc,de)
    mid=W(prior,k,h,l,cc,de)
    late=sum(pr*W(q,k,h,l,cc,de) for pr,q in [post(prior,h,l,s) for s in (0,1)])
    return tt,lo,mid,up,late

def check(K,d,vs):
    tt,lo,mid,up,late=vs;rr=1-tt;A=tt*up+rr*lo
    jt=-d+tt*max(up-K,0)+rr*max(lo-K,0)
    competitors=(F(0),max(mid-K,0),late-K-d)
    weak=all(jt>=x for x in competitors);strict=all(jt>x for x in competitors)
    fc=(late-tt*up)/rr;gc=(mid-tt*up+d)/rr;qc=up-d/tt
    partition=((K<=lo and A>=late and d<=A-mid and K<=A-d) or
               (lo<=K<=mid and K>=fc and K>=gc and K<=qc) or
               (mid<=K<=up and K>=fc and K<=qc) or
               (K>=up and d==0 and K>=late))
    assert partition==weak,(K,d,vs)
    if late>=A:
        assert strict==(max(fc,gc)<K<qc),(K,d,vs,'strict')
        db=tt*min(up-mid,(up-late)/rr)
        assert (max(fc,gc)<qc)==(d<db)
    return strict,weak

def reps(points):
    points=sorted(set(x for x in points if x>=0))
    return sorted(set(points+[(a+b)/2 for a,b in zip(points,points[1:])]+[points[-1]+1]))

def arrangement(vs):
    tt,lo,mid,up,late=vs;rr=1-tt;A=tt*up+rr*lo
    # Every boundary used in exact menu and weak partition, K=intercept+slope*d.
    lines=[(lo,0),(mid,0),(up,0),(late,0),(A,-1),((late-tt*up)/rr,0),((mid-tt*up)/rr,1/rr),(up,-1/tt),(F(0),0)]
    ds={F(0),max(F(0),A-mid),F(1)}
    for (a,b),(aa,bb) in combinations(lines,2):
        if b!=bb:
            root=(aa-a)/(b-bb)
            if root>=0:ds.add(root)
    count=st=we=0
    for d in reps(ds):
        ks=reps([F(0)]+[a+b*d for a,b in lines])
        for K in ks:
            strict,weak=check(K,d,vs);count+=1;st+=strict;we+=weak
    return count,st,we

assert [p*B-V(p) for p in [F(1,10),F(9,10),F(15,17),F(10,33),p0]]==[kb,kb,kp,km,k0]
fees=reps([F(0),kb,kp,kd,km,k0,F(1,2)])
count=st=we=0
for k in fees:
    for z in [F(0),F(1,4),F(1,2),F(3,4),F(1)]:
        vs=vals(k,z)
        aa,bb,cc=arrangement(vs);count+=aa;st+=bb;we+=cc
        if z==0:
            tt,lo,mid,up,late=vs
            assert up==max(u,v-k) and lo==max(F(0),w-k)
            assert mid==max(F(18,125),F(19,50)-k)
            assert late==max(n,F(19,50)-k)
            db=tt*min(up-mid,(up-late)/(1-tt))
            db_formula=F(247,2500) if k<=kp else F(441,6250)+t*k if k<=kd else F(241,1650)
            assert db==db_formula
# Analytic fee exclusion corollary, including both pilot-ceiling endpoints.
for fee,dcost in product(fees,[F(0),F(247,2500),F(3,25),F(241,1650),F(1,5)]):
    tt,lo,mid,up,late=vals(fee)
    exists=dcost<tt*min(up-mid,(up-late)/(1-tt))
    if dcost<F(247,2500):pred=True
    elif dcost>=F(241,1650):pred=False
    else:pred=fee>F(50,17)*(dcost-F(441,6250))
    assert exists==pred
# Additional primitive cases, including an uninformative early diagnostic and
# a non-dominated experiment: general weak table must still hold there.
primitive_checks=0
for prior,ah,al,h,l,cc,de in [
    (F(1,3),F(1,2),F(1,5),F(4,5),F(1,5),F(2,5),F(3,4)),
    (p0,F(1,2),F(1,2),H,L0,c,delta),
    (p0,F(19,20),F(1,20),H,L0,c,delta)]:
    for k,z in product([F(0),F(1,10),F(1,2)],[F(0),F(1,2),F(1)]):
        aa,bb,ccount=arrangement(vals(k,z,prior,ah,al,h,l,cc,de));primitive_checks+=aa
# Open-region witness: analytically proved in text; all corners checked exactly.
for k,d,K in product([f('.14'),f('.16')],[f('.009'),f('.011')],[f('.19'),f('.21')]):
    assert check(K,d,vals(k))[0]
k=f('.15');d=f('.01');R=max(W(p0,k)+d/t,n+r*d/t)
zc=max((u-R)/F(377,850),(v-k-R)/F(247,850))
assert zc==F(838,1235)
for z in [F(0),zc/2,zc, (zc+1)/2,F(1)]:
    tt,lo,mid,up,late=vs=vals(k,z)
    db=tt*min(up-mid,(up-late)/(1-tt))
    assert (d<db)==(z<zc)
    assert up==max(u-F(377,850)*z,v-k-F(247,850)*z)
# Reset contraction checked at every sampled menu boundary/interior of this slice.
contraction_checks=0
for K,d in product(reps([0,F(1,5),u,n]),reps([0,F(1,100),F(241,1650)])):
    prev=None
    for z in [F(0),F(1,4),F(1,2),F(3,4),F(1)]:
        tt,lo,mid,up,late=vals(k,z);jt=-d+tt*max(up-K,0)+(1-tt)*max(lo-K,0)
        if prev is not None:assert jt<=prev
        prev=jt;contraction_checks+=1
# Exhaustive finite fee table exported with exact rational strings.
with (O/'certificate_fee_partition.csv').open('w') as ff:
    wr=csv.writer(ff);wr.writerow(['k_cell','L','M','U','N'])
    for row in [('0:k_b','38/165-k','19/50-k','57/85-k','19/50-k'),('k_b:k_plus','38/165-k','19/50-k','57/85-k','38/125'),('k_plus:k_minus','38/165-k','19/50-k','2497/4250','38/125'),('k_minus:k_0','0','19/50-k','2497/4250','38/125'),('k_0:infinity','0','18/125','2497/4250','38/125')]:wr.writerow(row)
summary=dict(revision='CENTRAL_R2_20261017_P09',real_observations=0,arithmetic='Python fractions.Fraction, exact',baseline_cost_boundary_and_interior_checks=count,baseline_strict_cases=st,baseline_weak_cases=we,additional_primitive_checks=primitive_checks,open_box_corner_checks=8,reset_contraction_checks=contraction_checks,reset_critical_exact=str(zc),reset_critical_decimal=float(zc),pilot_ceiling_low=str(F(247,2500)),pilot_ceiling_high=str(F(241,1650)),python=platform.python_version(),scope='Finite exact boundary/interior verification of analytic proof; no empirical data and no proof of global novelty.')
(O/'cost_region_verification.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
