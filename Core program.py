

import numpy as np
import random





""" The payoff_calculation function calculates the payoff of player I and player II when I plays the strategy P and II plays Q using transition matrix """




# NOTATION: Q is the MUTANT strategy, P is the resident strategy

# We expect vectors of size 5 in here, last element is the probability of initial cooperation

# A1, A2,... each represents a row of the transition matrix

# We should have a 5x5 matrix here for the 5 states CC,CD,DC,DD,A 

# P = [pcc,pcd,pdc,pdd,p0] where p0 is the probability of initial cooperation

def payoff_calculation(P,Q):  
    A1 = np.array([P[0]*Q[0]*(P_trans_CC),P[0]*(1-Q[0])*(P_trans_CC),(1-P[0])*Q[0]*(P_trans_CC),(1-P[0])*(1-Q[0])*P_trans_CC,1-P_trans_CC])
    A2 = np.array([P[1]*Q[2]*P_trans_CD,P[1]*(1-Q[2])*P_trans_CD,(1-P[1])*Q[2]*P_trans_CD,(1-P[1])*(1-Q[2])*P_trans_CD,1-P_trans_CD])
    A3 = np.array([P_trans_CD*P[2]*Q[1],P_trans_CD*P[2]*(1-Q[1]),P_trans_CD*(1-P[2])*Q[1],P_trans_CD*(1-P[2])*(1-Q[1]),1-P_trans_CD])
    A4 = np.array([P_trans_DD*P[3]*Q[3],P_trans_DD*P[3]*(1-Q[3]),P_trans_DD*(1-P[3])*Q[3],P_trans_DD*(1-P[3])*(1-Q[3]),1-P_trans_DD])
    A5 = np.array([0,0,0,0,1])
    matrix = np.array([A1,A2,A3,A4,A5])
    v0 = [P[4]*Q[4],P[4]*(1-Q[4]),(1-P[4])*Q[4],(1-P[4])*(1-Q[4]),0]
    V= (1-delta)*np.dot(v0,np.linalg.inv((np.eye(5)-delta*matrix)))  #Stationary state vector/ mean distribution                                                #Normalizing
    pA = np.dot(np.array([R_,S_,T_,P_,Q_]),V)                        #Payoff of first player
    pB = np.dot(np.array([R_,T_,S_,P_,Q_]),V)
    return list([pA,pB,V])                                           #returns the mean distribution and the payoffs of the players




""" Setting up the parameters of simulation (global) """ 




delta = 0.9   # Continuation probability / payoff discounting parameter
b = 2           # benefit term in donation game
c = 1            # Cost term, hence b/c = b
#alpha = 0.3      # alpha is used to parameterize absorbing state payoff 

R_ = b-c         # Reward R
S_ = -c          # Sucker's payoff S
T_ = b           # Temptation T
P_ = 0           # Mutual Defection punishment P
Q_ = 0.1         # Absorbing state payoff

P_trans_CC=1                                                #  Probability of being in state 1 in next round when in CC (z1)
P_trans_CD = 0.8                                            # "                                                     " CD (z2)
P_trans_DD = 0.5                                            # "                                                     " DD (z4)
init_strat = [0.01,0.01,0.01,0.01,0.01]                     # Initial strategy the population will be full off 
init_state = payoff_calculation(init_strat,init_strat)      # Calculating the payoffs of the two strategies 




# Calculates the net payoff in a population of N with j mutants
# Notations P: resident, Q: mutant
# Notations PP: payoff of P against P, PQ: payoff of P against Q, and so on

def net_payoff(PP,PQ,QP,QQ,N,j):
    Pnet = ((N-j-1)*PP + j*PQ)/(N-1)
    Qnet = ((j-1)*QQ + (N-j)*QP)/(N-1)
    return [Pnet,Qnet]




"""  The function fixation prob calculates the fixation probability of a Q type mutant in a population of N players where residents apply the strategy P """




# beta is the selection strength

def fixation_prob(P,Q,N,beta):
    P_case = payoff_calculation(P,P)
    P_vec = P_case[2]                       # mean distributuion when P plays against P
    PP = P_case[0]                          #Payoff of P against itself
    mixed= payoff_calculation(P,Q)
    PQ = mixed[0]                           #Payoff of P against Q
    QP = mixed[1]                           #Payoff of Q against P
    Q_case = payoff_calculation(Q,Q)
    Q_vec = Q_case[2]                       # mean distribution when Q vs Q
    QQ = Q_case[0] 

    #Calculating the fixation prob using the formula
    S = 1
    factor = 1
    for i in range(1,N):
        pi = net_payoff(PP,PQ,QP,QQ,N,i)
        alpha = np.exp(-beta*(pi[1]-pi[0]))
        factor*=alpha
        S+=factor
    fixprob = 1/S
    stats = [PP,QQ,P_vec,Q_vec]
    return fixprob,stats    

# pi contains [avg payoff of P, avg payoff of Q,state vector of all P population, state vector of all Q population]




""" The functions below calculates various outcomes of the evolutionary process such 
as cooperation rate, time spent in healthy state, probability of the resident
population being composed of partner strategy platers and mean payoff of the population"""




# Calculates time spent in state-1 from mean distribution
def t_state1(vector):
    return 1-vector[4]

# Calculates cooperation rate from mean distribution
def c_rate(vector):
    return (vector[0]+(vector[1]+vector[2])/2)/(1-vector[4])


#Checks if a strategy is partner 
def partner_check(arr,ep=0.1):
    z2 = P_trans_CD
    z4 = P_trans_DD
    p1 = arr[0]
    p2 = arr[1]
    p3 = arr[2]
    p4 = arr[3]
    p0 = arr[4]
    cond0 = p0-(1-ep)                # p_{0} should be in neighbourhood of 1 or p0>(1-ep) 
    cond1 = p1 -(1-ep)               # p_{1} should be in neighbourhood of 1 or p1>(1-ep) 
    flag = 0
    #Next conditions are from Akin's lemma
    cond2 = (T_ - R_)*(1 - delta)*(1 - (1 - p3)*delta*z2) + (R_ - Q_)*(1 - z2)*(delta*z2*(p2 - p3) - 1) - (R_ - S_)*(1 - delta)*(1 - p2)*delta*z2 
    cond3 = T_ - R_ + delta * (Q_ - T_ + (-1 + delta) * P_ * (-1 + p2) * z2 - Q_ * z2 + delta * Q_ * z2 - delta * p2 * Q_ * z2 + p2 * R_ * z2 + (-1 + p4) * (-R_ + delta * (Q_ - T_) + T_) * z4 + delta * (p2 - p4) * (Q_ - R_) * z2 * z4)
    if cond0>0 and cond1>0 and cond2<0 and cond3<0:
        flag=1        
    return flag


# Main function
# Takes mutants in every iteration and checks if it can replace the resident strategy

""" The function selection runs the evolutionary process for a number of mutants indicated by the parameter time,
beta specifies the strength of the selection  """

#parameters: popN is population size, time is number of generations, Beta is selection strength
def selection(popN,time,Beta):
    #Initial paramters for the run 

    strat = []                                       # Always consists of two elements, [resident,mutant] for each iteration 
    Avg_payoff = [init_state[0]]   
    strat.append(init_strat)                        # initial resident population of random population
    mutant = np.random.uniform(0, 1, 5)             # Drawing five values from 0 to 1 for initial mutant

    coop_rate = [c_rate(init_state[2])]
    T_state1 = [t_state1(init_state[2])]
    partners = [0]
    strat.append(mutant)                           
    #Above code initiates a population of ALL-D with a random mutant to check against for the first time


    for i in range(time):
        prob,pi = fixation_prob(P = strat[0],Q = strat[1],N=popN,beta = Beta)  #Should give the fixation probability of strat Q in resident P population


        #Scenario when mutant gets fixed
        if random.random()<prob: 
            del strat[0]                          #Eliminate resident strategy
            Avg_payoff.append(pi[1])
            coop_rate.append(c_rate(pi[3]))
            T_state1.append(t_state1(pi[3]))
            partners.append(partner_check(strat[0]))
        else:
        #Mutant doesn't get selected 
            del strat[1]                          #Eliminate previous mutant
            Avg_payoff.append(pi[0])
            coop_rate.append(coop_rate[-1])       # Copying the the last element as nothing new is selected
            T_state1.append(T_state1[-1])
            partners.append(partners[-1])
        mutant = np.random.uniform(0, 1, 5)   #New mutant introduced
        strat.append(mutant)

    return np.array(coop_rate),np.array(T_state1),np.array(partners),np.array(Avg_payoff) 




# Array of selection strength

iterations = 100
# Number of generations
t = 1000000
beta = 1                            # Selection strength
partner = np.zeros(t+1)             # string has 0 or 1 depending on whether a partner strategy gets fixed at each time step
coop_rate = np.zeros(t+1)           # string stores cooperation rate for each time step
time_1 = np.zeros(t+1)              # string has time spent in state-1 for each time step
avg_payoff = np.zeros(t+1)          # string has average payoff ""                       ""


coop_traj = []
time_1_traj = []
partner_traj = []

#Running for 100 ensembles
for i in range(iterations):
    Results = selection(popN = 100,time = t,Beta = beta)
    coop_traj.append(Results[0])
    time_1_traj.append(Results[1])
    partner_traj.append(Results[2])

#Taking the average of 100 ensembles
coop_rate = np.mean(coop_traj,axis = 0)  
time_1 = np.mean(time_1_traj,axis = 0)
partner = np.mean(partner_traj,axis = 0)
avg_payoff = np.mean(partner_traj,axis = 0)




# Setting transition probabilities to 1 for single state (s) prisoner's dilemma

P_trans_CC = 1                                             
P_trans_CD = 1
P_trans_DD = 1   

beta = 1

iterations = 100
# Number of generations
t = 1000000
partner_s = np.zeros(t+1)             
coop_rate_s = np.zeros(t+1)
time_1_s = np.zeros(t+1)
avg_payoff_s = np.zeros(t+1)
coop_traj_s = []
partner_traj_s = []

for i in range(iterations):
    Results = selection(popN = 100,time = t,Beta = beta)
    coop_traj_s.append(Results[0])
    partner_traj_s.append(Results[2])
    print(i)

# Taking the average of 100 ensembles/trajectories
coop_rate_s = np.mean(coop_traj_s,axis = 0)
partner_s = np.mean(partner_traj_s,axis = 0)




""" Saving results of the evolutionary process"""


np.save("coop_rate_stc_Q01_z11.npy",coop_rate)
np.save("coop_rate_reg_Q01_z11.npy",coop_rate_s)
np.save("prtnr_stc_Q01_z11.npy", partner)
np.save("prtnr_reg_Q01_z11.npy", partner_s)
np.save("time_state1_Q01_z11.npy",time_1)


np.save("cooperation_trajectory.npy",coop_traj)
np.save("time_trajectory.npy", time_1_traj)
np.save("partner_trajectory.npy",partner_traj)
np.save("cooperation_trajectory_s.npy",coop_traj_s)
np.save("partner_trajectory_s.npy",partner_traj_s)

