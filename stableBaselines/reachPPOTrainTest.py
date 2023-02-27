import gym
import panda_gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

def train_save_PPO_CartPole(env, modelName):
    
    model = PPO("MultiInputPolicy", env, verbose=1,learning_rate = 1e-3)
    model.learn(total_timesteps=50000)
    model.save(modelName)

    del model

def load_model(env, modelName):

    model = PPO.load(modelName, env=env)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render()
        # VecEnv resets automatically
        # if done:
        #   obs = env.reset()

    env.close()

def main():
    env = gym.make("PandaReach-v2", render=False)
    modelName = "PPO_PandaReach-v2"
    train_save_PPO_CartPole(env, modelName)
    #load_model(env, modelName)


main()
