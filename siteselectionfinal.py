#!/bin/env python

import random
import matplotlib.pyplot as plt
import numpy as np
#Author: Aadeshnpn
##Nest site selection process start when a cluster is formed
##by the colony splitting itself when the queen & about half the old colony depart

##Important Notation
#R -> fraction of bees resting
#O -> fraction of bees observing
#E -> fraction of bees exploring
#A -> fraction of bees assessing
#D -> fraction of bees dancing
#a -> rates at which bees cease resting
#b -> rates at which bees cease observing
#c -> rates at which bees cease exploring
#m -> rates at which bees cease assessing
#n -> rates at which bees cease dancing
#p(D) -> fraction of newly employed bees which become assessors
#q(D) -> fraction of newly employed bees which become explorers
#v    -> fraction of bees that retire after dancing
#w    -> fraction of bees that return to site after dancing

##Assumptions
#R    -> O(a) <> a is the positive rate at which Resters(R) become Observers(O)
#O    -> E,A  <> b is the positive rate at which Observers (O) become employed
#E    -> O    <> c is the positive rate at which Explolers (E) become Observers (O)
#A    -> D    <> m is the positive rate at which Assessors (A) become dancers (D)
#D    -> R,A  <> n is the positive rate at which Dancers (D) cease to dance
#p(D) -> D/(D+e) be the fraction of newly employed bees which are successfully recruited to assess the site (O->A) when there are D bees dancing for it
#q(D) -> 1-p(D) represent the remaining fraction (O->E)
#e    -> Half-saturation constant (positive)
#0<v<=1 bees that cease to dance immediately retire (D->R)
#w=1-v -> return to the site to reassess it (D -> A)
#All bess are initiall resters, observers, and explorers except for a single assessor bee that has independently made a one-time discovery

##Equations
#X. = dX/d1
#R. = -aR                   + vnD, R(0)=R0,
#O. =  aR-     bO + cE,            O(0)=O0,
#E. =      q(D)b0 - cE,            E(0)=E0,
#A. =      p(D)b0       -mA + wnD, A(0)=A0,
#D. =                    mA -  nD, D(0)=0,
#where R0+O0+E0+A0=1 && 0< A0 <<1

##Site Quality
#Q     = Site Quality
#σ     = time spent on each waggle run
#M     = number of waggle runs performed by bee dancing for perfect site
#n(Q)  = rate at which bees cease dancing
#v(Q)  = fraction of bees that retire after dancing
#w(Q)  = fraction of bees that return to site after dancing

##Algorithm
#If the site is viable , check (Q>=Q** as A*>=Aq)
#But the site should be able to attract quorum in finite time, A(t)=Aq for t>0 and quorum time tq=min(t>=0 : A(t) =Aq)
#A site in viable in t hours if tq <=t
##Numerical Studies

a=0.100 #10 min resting
b=0.125 #08 min observing
c=0.033 #30 min exploring
m=0.050 #20 min assessing
σ=0.1   #consecutive waggle runs start 6 secs apart
M=150   #150 waggle runs for a perfect site
e=0.323 #chose so taht Q* =0.450
A_0=0.005     #200 scount bees
A_q=0.1       #20 accessing bees produce quorum [20,22]
O_bar=0.211
E_bar=0.789
Q_ast=0.450  #(A_ast=0)
Q_ast2=0.500 #(A_ast=A_q)
t_o=list(range(1,3000)) #varied between simulations

def single_site_model(R_0,O_0,E_0,A_0,D_0,a,b,c,v,w,m,n,Aq,stopval=3000):
    tq=[]
    ##Initial Setting
    R=[]
    O=[]
    E=[]
    A=[]
    D=[]
    for i in t_o:
        R_dot=-a*R_0+v*n*D_0
        O_dot=a*R_0-b*O_0+c*E_0
        p_d=D_0/(D_0+e)
        q_d=1-p_d
        E_dot=q_d*b*O_0-c*E_0
        A_dot=p_d*b*O_0-m*A_0+w*n*D_0
        D_dot=m*A_0-n*D_0
        R_0=R_0+R_dot;R.append(R_0)
        O_0=O_0+O_dot;O.append(O_0)
        E_0=E_0+E_dot;E.append(E_0)
        A_0=A_0+A_dot;A.append(A_0)
        D_0=D_0+D_dot;D.append(D_0)
        if A_0>=Aq:
            tq.append(i)
        if i>=stopval:
            break
    print ("Final values of : R,O,E,A,D",i,R_0,O_0,E_0,A_0,D_0)
    return R_0,O_0,E_0,A_0,D_0,tq,R,O,E,A,D

def two_site_model(R_0,O_0,E_0,A_1,D_1,A_2,D_2,a,b,c,v1,v2,w1,w2,m,n1,n2,Aq,stopval=3000):
    #R,O,E,A1,D1,A2,D2,tq=two_site_model(R0,O0,E0-A_0,A0,D0,A_0,0,a,b,c,v1,v2,w1,w2,m,n1,n2,A_q)
    tq1=[]
    tq2=[]
    R=[];O=[];E=[];A1=[];A2=[];D1=[];D2=[]
    for i in t_o:
        R_dot=-a*R_0+v1*n1*D_1+v2*n2*D_2
        O_dot=a*R_0-b*O_0+c*E_0
        p_d=(D_1+D_2)/(D_1+D_2+e)
        q_d=1-p_d
        E_dot=q_d*b*O_0-c*E_0
        A_1dot=(D_1/(D_1+D_2+e))*b*O_0-m*A_1+w1*n1*D_1
        A_2dot=(D_2/(D_1+D_2+e))*b*O_0-m*A_2+w2*n2*D_2
        D_1dot=m*A_1-n1*D_1
        D_2dot=m*A_2-n2*D_2
        R_0=R_0+R_dot;R.append(R_0)
        O_0=O_0+O_dot;O.append(O_0)
        E_0=E_0+E_dot;E.append(E_0)
        A_1=A_1+A_1dot;A1.append(A_1)
        A_2=A_2+A_2dot;A2.append(A_2)
        D_1=D_1+D_1dot;D1.append(D_1)
        D_2=D_2+D_2dot;D2.append(D_2)
        if A_2>=Aq:
            tq2.append(i)
        if A_1>=Aq:
            tq1.append(i)
        if i>=stopval:
            break
    print ("Final values of : R,O,E,A1,D1,A2,D2",i,R_0,O_0,E_0,A_1,D_1,A_2,D_2)
    return R_0,O_0,E_0,A_1,D_1,A_2,D_2,tq1,tq2,R,O,E,A1,A2,D1,D2
    
def existence_of_solution():
    R_ast=R_0
    O_ast=O_0
    E_ast=E_0
    A_ast=A_0
    D_ast=D_0
    #Each components always sum to one
    #A_ast=0 #Equlibrium is disinterested #D_ast=0
    #A_ast>0 #Equlibrium is interested #D_ast>0
    #A_ast<A_q  #PIE
    #A_ast>+A_q #FIE
    O_bar=c/(b+c)
    E_bar=b/(b+c)
    rec_num=(1/(e*n*v)*(1/(((1/b)+(1/c)))))
    abs_rec_num=b/(e*n*v)
    print ("O_bar value is:",O_bar)
    print ("E_bar value is:",E_bar)
    print ("rec_num value is:",rec_num)
    print ("abs_rec_num values is:",abs_rec_num)
    print ("All components sum to:",round(R_ast+O_ast+E_ast+A_ast+D_ast,2))
    print ("Since O_bar < 1, it follows basic recruitment number is less than absolute recruitment number")

def disinterested_equilibrium(e,n,v,b,c):
    rec_num=(1/(e*n*v)*(1/(((1/b)+(1/c)))))
    abs_rec_num=b/(e*n*v)
    print ("Since rec_num >1, DE is unstable. Basic recruitment number is : ",rec_num)

def interested_equilibrium(e,n,v,b,c,m,Q):
    rec_num=round((1/(e*n*v)*(1/(((1/b)+(1/c))))),1)
    #print (rec_num)
    if Q<=Q_ast:
        print ("Since Q is less than Q*, IE doesn't exists")
        return 0
    if rec_num>1.0:
        print ("Since basic recruitment number is greater than 1, Interested equilibrium is feasible. R0 is : ",rec_num)
    else:
        print ("Since basic recruitment number is greater than 1, Interested equilibrium is not feasible. R0 is : ",rec_num)
        return 0
    D_ast=round(((1/(n*v))-e*((1/b)+(1/c)))/(((1/m)+(1/n))*(1/v)+(1/a)+(1/b)),3)
    R_ast=round(((n*v)/a)*D_ast,3)
    O_ast=round(((n*v)/b)*(D_ast+e),3)
    E_ast=round((e*n*v)/c,3)
    A_ast=round((n/m)*D_ast,3)
    print ("All components sum to:",round(R_ast+O_ast+E_ast+A_ast+D_ast,2))
    print ("D_Ast",D_ast)
    if A_ast < A_q:
        print ("Interested Equlibrium constitutes a PIE with A* and Aq as:",A_ast,A_q)
    else:
        print ("Interested Equlibrium constitutes a FIE with A* and Aq as:",A_ast,A_q)
    
#existence_of_solution()    
#disinterested_equilibrium(e,n,v,b,c)
#interested_equilibrium(e,n,v,b,c,m)

def plotGraph(rlist,olist,elist,alist,dlist,Q,Aq,Q_ast,Q_ast2,rang=3000):
    rlist=np.array(rlist)
    olist=np.array(olist)
    elist=np.array(elist)
    alist=np.array(alist)
    dlist=np.array(dlist)
    plt.plot(np.array(list(range(1,rang))),rlist,'b*',label="R")
    plt.plot(np.array(list(range(1,rang))),olist,'b--',label="O")
    plt.plot(np.array(list(range(1,rang))),elist,'g^',label="E")
    plt.plot(np.array(list(range(1,rang))),alist,'r--',label="A")
    plt.plot(np.array(list(range(1,rang))),dlist,'bs',label="D")
    plt.xlabel('Time (Mins)')
    plt.title('Assessment (Q = %0.2f)' % Q)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.ylabel('Time (Mins)')
    plt.show()

def plotGraph2(rlist,olist,elist,a1list,a2list,d1list,d2list,Q1,Q2,t0,rang=3000):
    rlist=np.array(rlist)
    olist=np.array(olist)
    elist=np.array(elist)
    a1list=np.array(a1list)
    a2list=np.array(a2list)
    d1list=np.array(d1list)
    d2list=np.array(d2list)
    #plt.plot(np.array(list(range(1,rang))),rlist,'b*',label="R")
    plt.plot(np.array(list(range(1,rang))),olist,'b--',label="O")
    plt.plot(np.array(list(range(1,rang))),elist,'g^',label="E")
    plt.plot(np.array(list(range(1,rang))),a1list,'r--',label="A1")
    plt.plot(np.array(list(range(1,rang))),a2list,'bs',label="A2")
    plt.xlabel('Time (Mins)')
    plt.title('Discrimination (Q1 = %0.2f, Q2 = %0.2f,t(o) = %d mins)' % (Q1,Q2,t0))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.ylabel('Time (Mins)')
    plt.show()
            
def run_single_model():
    for Q in [0.45,0.48,0.50,0.55]:
        #print (Q)
        print ("---------------------------------------------")
        n=0.066/Q
        v=1-Q
        w=Q
        R,O,E,A,D,tq,rlist,olist,elist,alist,dlist=single_site_model(0,O_bar,E_bar-A_0,A_0,0,a,b,c,v,w,m,n,A_q)
        #rlist=np.array(rlist);olist=np.array(olist);np.array(elist);np.array(alist);np.array(dlist)
        plotGraph(rlist,olist,elist,alist,dlist,Q,A_q,Q_ast,Q_ast2)
        interested_equilibrium(e,n,v,b,c,m,Q)
        if len(tq)!=0:
            print ("Quorum time achieved at %f hrs:",tq[0]/(60.0))
        print ("---------------------------------------------")

def run_double_model():
    #for Q in [0.45,0.48,0.50,0.55]:
        #print (Q)
    print ("---------------------------------------------")
    Q1=0.6
    Q2=0.7
    n1=0.066/Q1
    n2=0.066/Q2
    v1=1-Q1
    v2=1-Q2
    w1=Q1
    w2=Q2
    t0=120
    R0,O0,E0,A0,D0,tq,rlist,rlist,rlist,rlist,rlist=single_site_model(0,O_bar,E_bar-A_0,A_0,0,a,b,c,v1,w1,m,n1,A_q,t0)
    R,O,E,A1,D1,A2,D2,tq1,tq2,rlist,olist,elist,a1list,a2list,d1list,d2list=two_site_model(R0,O0,E0-A_0,A0,D0,A_0,0,a,b,c,v1,v2,w1,w2,m,n1,n2,A_q)
    plotGraph2(rlist,olist,elist,a1list,a2list,d1list,d2list,Q1,Q2,t0)
    #interested_equilibrium(e,n,v,b,c,m,Q)
    if len(tq1)!=0:
        print ("Quorum time achieved for site1 at  hrs:",tq1[0]/(60.0))
    print ("---------------------------------------------")
    if len(tq2)!=0:
        print ("Quorum time achieved for site2 at  hrs:",tq2[0]/(60.0))
    print ("---------------------------------------------")
                    
#run_single_model()
run_double_model()
