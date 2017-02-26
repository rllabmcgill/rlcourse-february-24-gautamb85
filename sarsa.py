import gym
import numpy as np
import random

eps=0.01
def eps_greedy_action(Q, s, epsilon=0.001):
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
        #Q[rstate,ract] = Q[rstate,ract] + lr*(r_hat + y*np.max(Q[rstate,:]) - Q[rstate,ract])
        Q[rstate,ract] = Q[rstate,ract] + lr*(r_hat + y*Q[s_hat,ract] - Q[rstate,ract])
        psteps+=1

    return Q

def train(env,planning=True):
    
    alpha = 0.1
    gamma = 0.99

    Q = np.ones([env.observation_space.n,env.action_space.n])
    model_nextS=np.zeros([env.observation_space.n,env.action_space.n])
    model_nextR=np.zeros([env.observation_space.n,env.action_space.n])
    fill_model(env,Q,model_nextS,model_nextR)
 
    score = []

    for i in range(10000):
        s = env.reset()
        action = eps_greedy_action(Q,s,eps)

        prev_observation = None
        prev_action = None

        t = 0

        for t in range(2500):
            s1, reward, done, _ = env.step(action)

            action = eps_greedy_action(Q,s1,eps)

            if not prev_observation is None:
                if done:
                    Q[prev_observation,prev_action] += alpha * (reward - Q[prev_observation,prev_action])
                else:
                    Q[prev_observation, prev_action] += alpha * (reward + gamma * Q[s1,action] - Q[prev_observation, prev_action])
                    #Q[prev_observation, prev_action] += alpha * (reward + gamma *np.sum(0.25*Q[s1,:]) - Q[prev_observation, prev_action])
                if planning:
                    model_nextS[prev_observation,prev_action] = s1
                    model_nextR[prev_observation,prev_action] = reward
                    Q=plan(Q,alpha,gamma,model_nextS,model_nextR)
 
            prev_observation = s1
            prev_action = action
            
            if done:
                if len(score) < 100:
                    score.append(reward)
                else:
                    score[i % 100] = reward

                print("Episode {} finished after {} timesteps with r={}. Running score: {}".format(i, t, reward, np.mean(score)))
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
    env = gym.make("FrozenLake8x8-v0")
    Q=train(env,planning=False)
    evaluate(env,Q)
