#!/usr/bin/python
import gym
import numpy as np
import random
import pylab as pl
import matplotlib


def eps_greedy_action(Q, s, epsilon=0.0001):
    return random.choice(np.arange(Q.shape[1])) if (random.uniform(0, 1) <= epsilon) else Q[s, :].argmax()
    #return Q[s,:].argmax()

def fill_model(env,Q,model_nextS,model_nextR):
    for i in range(10000):
        sa = env.observation_space.sample()
        aa = eps_greedy_action(Q,sa)

        s1,r,d,_ = env.step(aa)
        model_nextS[sa,aa] = s1
        model_nextR[sa,aa] = r


def plan(Q,lr,y,model_nextS,model_nextR,plan_steps=15):
    
    vis_before = np.nonzero(model_nextS)
    pstates = list(set(vis_before[0]))

    psteps=0        
    
    while psteps<=plan_steps-1:

        #select a state at random from previous states
        if pstates:
            rstate = random.sample(pstates,1)
            rstate = rstate[0]
        else:
            rstate=0
        #find previously sampled actions
        pacts = np.nonzero(model_nextS[rstate,:])
        if pacts[0].any():
            ract = random.sample(pacts[0],1)
            ract = ract[0]
        else:
            ract=0

        #simulate experience from model
        r_hat = (model_nextR[rstate,ract]).astype('int32')
        s_hat = (model_nextS[rstate,ract]).astype('int32')
    
        #use Q-learning for planning
        Q[rstate,ract] = Q[rstate,ract] + lr*(r_hat + y*np.max(Q[s_hat,:]) - Q[rstate,ract])
        psteps+=1

    return Q   

def train(env,lr=0.1,y=0.99,num_episodes=10000,planning=False):
    
    Q = np.ones([env.observation_space.n,env.action_space.n])
    #initialize model
    model_nextS=np.zeros([env.observation_space.n,env.action_space.n])
    model_nextR=np.zeros([env.observation_space.n,env.action_space.n])
    #fill_model(env,Q,model_nextS,model_nextR)
    #create lists to contain total rewards and steps per episode
    #jList = []
    rList = []
    score=[]
    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        #d = False
        j = 0
        #The Q-Table learning algorithm
        while j < 2500:
            j+=1
            #a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
            a=eps_greedy_action(Q,s)
            #Get new state and reward from environment
            s1,r,done,_ = env.step(a)
            
            if done:
                Q[s,a] = Q[s,a]+lr*(r - Q[s,a])
            else:
                Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
            
            if planning:
                #update model
                model_nextS[s,a] = s1
                model_nextR[s,a] = r
                Q=plan(Q,lr,y,model_nextS,model_nextR)
            
            rAll += r
            s = s1
            if done:
                if len(score) < 100:
                    score.append((r))
                else:
                    score[i%100] = r
            
                print("Episode {} finished after {} timesteps with r={}. Running score: {}".format(i, j, r, np.mean(score)))

                break
    return Q

def evaluate(env,Q):
    num_episodes=1000
    donecount=0 #counter for how many times the goal is reached
    tsteps=[]
    rsum=0
    score=[]
    for i in range(1000):
        s=env.reset()
        
        prev_observation = None
        prev_action = None

        t = 0
        
        for t in range(2500):
            action = eps_greedy_action(Q,s)
            s1, reward, done, _ = env.step(action)
            rsum+=reward

            s=s1
            if s1==15:
                donecount+=1
                tsteps.append(t)
            if done:
                if len(score) < 100:
                    score.append((reward))
                else:
                    score[i%100] = reward
                
                break
    
    #eps=np.arange(len(rsum))
    #pl.plot(eps,rsum)
    #pl.show()
    #print donecount
    tsteps = np.asarray(tsteps,dtype='float32')
    mts = tsteps.mean()
    avg_reward = rsum/float(1000)
    #print mts,avg_reward
    
    print("{} episodes finished in an average of {}. Running score: {}".format(num_episodes, mts, np.mean(score)))

if __name__=='__main__':
    env = gym.make("FrozenLake-v0")
    Q=train(env,planning=False)
    evaluate(env,Q)
