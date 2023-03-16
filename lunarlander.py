#pip install tensorflow==1.15.0, tensorflow-gpu==1.15.0, stable_baselines, gym box2d-py

import gym #lunar lander environment, lunar lander model 적용
from stable_baselines import ACER #강화학습 reinforcement agent 훈련을 위해 acer policy(rl algorithm like dqn) 불러오기
from stable_baselines.common.vec_env import DummyVecEnv #create dummy vectorized environment for stable baseline
from stable_baselines.common.evaluation import evaluate_policy #make it easy to test our model and it's performance

environment_name = 'LunarLander-v2' #효율성
#test random environment->take random steps
env = gym.make(environment_name)#lunar lander environment 형성
episodes = 10 #착륙 10번 시도
for episode in range(1, episodes+1): #1부터 11까지 정수 범위를 반환
    state = env.reset() #resetting variables
    done = False
    score = 0

    while not done: #done이 True
        env.render() #시각화
        action = env.action_space.sample() #take random steps(퍼포먼스가 좋지는 않음)
        n_state, reward, done, info = env.step(action)#?
        score+=reward #score에 reward 합산
    print('Episode:{} Score:{}'.format(episode, score)) #format 함수로 문자열 포맷팅
env.close()

#build and train the model->goals: high explained variance, high mean_episode reward
env = gym.make(environment_name)
env = DummyVecEnv([lambda: env]) #rap the environment in dummy vec environment, generate env each time
model = ACER('MlpPolicy', env, verbose = 1) #u can use other algorithm, 3key variable(2 argument, 1 key argument) mlppolicy=우리가 사용하는 정책, verbose=1 기본,0 출력X, 2 출력 간소화
model.learn(total_timesteps=100000) #train 100000 different steps, run callback(automatically end training at optimal level)

#save and test the model
evaluate_policy(model, env, n_eval_episodes=10, render=True) #two argument, 2 keyword argument n_eval_Episode how many chances we are giving our model to perform render=true to visualize
env.close() #close 함수

model.save("ACER_model") #모델 저장
del model #모델 삭제
model = ACER.load("ACER_model", env=env) #reload model, also need to pass throughout env while reloading/load() algorithm
obs = env.reset() #reset environment back to base state, capture those observations in obs
while True: #계속 러닝함
    action, _states = model.predict(obs) #use predict method to generate action based on observation, model recieves inputs from lunar env and predict the best action/get actopn and current state
    obs, rewards, done, info = env.step(action) #apply action to env using step, get new observation, rewards, whether it;s done or not, and additional info
    env.render()

env.close() #러닝 멈추기 위해
