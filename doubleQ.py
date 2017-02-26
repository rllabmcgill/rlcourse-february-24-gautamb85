import gym
import numpy as np
import random

def eps_greedy_action(Q, s, epsilon=0.0001):
    return random.choice(np.arange(Q.shape[1])) if (random.uniform(0, 1) <= epsilon) else Q[s, :].argmax()
    #return Q[s,:].argmax()

def train(env,lr=0.1,y=0.99,num_episodes=10000):
    
    Qa = np.ones([env.observation_space.n,env.action_space.n])
    Qb = np.ones([env.observation_space.n,env.action_space.n])

    #initialize model
    model_nextS=np.zeros([env.observation_space.n,env.action_space.n])
    model_nextR=np.zeros([env.observation_space.n,env.action_space.n])

    updates=['updateA','updateB']
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
            #Choose an action by greedily (with noise) picking from Q table
            if np.max(Qa[s,:])>np.max(Qb[s,:]):
                a = eps_greedy_action(Qa,s) 
            else:
                a = eps_greedy_action(Qb,s) 
       
            #Get new state and reward from environment
            s1,r,done,_ = env.step(a)
            choice = random.sample(updates,1)
            choice = choice[0]
            
            if choice=='updateA':
                a_star = np.argmax(Qa[s1,:])
                if done:
                    Qa[s,a] = Qa[s,a] + lr*(r - Qa[s,a])
                else:
                    Qa[s,a] = Qa[s,a] + lr*(r + y*Qb[s1,a_star] - Qa[s,a])
            
            if choice=='updateB':
                b_star = np.argmax(Qb[s1,:])
                if done:
                    Qb[s,a] = Qb[s,a] + lr*(r - Qb[s,a])
                else:
                    Qb[s,a] = Qb[s,a] + lr*(r + y*(Qa[s1,b_star] - Qb[s,a]))
            
            rAll += r
            s = s1
            if done:
                if len(score) < 100:
                    score.append(r)
                else:
                    score[i%100] = r
            
                print("Episode {} finished after {} timesteps with r={}. Running score: {}".format(i, j, r, np.mean(score)))

                break
    Q=Qa
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
    Q=train(env)
    evaluate(env,Q)
