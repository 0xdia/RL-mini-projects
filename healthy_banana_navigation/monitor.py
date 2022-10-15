from agent import Agent
import time

seed = 2

def run(env, num_episodes, load_model, path, save_model, train_mode):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=train_mode)[brain_name]
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)
    agent = Agent(state_size, action_size, seed, load_model, path, train_mode)
    
    last_episode, scores = agent.episode_num, agent.scores
    accuracy = []
    for i in range(1, num_episodes+1):
        print(f"[*] Episode: {i+last_episode}")
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        healthy_bananas = total_bananas = 0        
        while True:
            action = agent.act(state, 0.1)
            if not train_mode:
                time.sleep(0.05)
            env_info = env.step(action)[brain_name]    
            next_state = env_info.vector_observations[0]   
            reward = env_info.rewards[0]       
            done = env_info.local_done[0]                 
            agent.step(state, action, reward, next_state, done)
            score += reward                             
            state = next_state
            if reward != 0:
                total_bananas += 1
                if reward == 1.0:
                    healthy_bananas += 1
            if done:                                     
                break
        scores.append(score)
        accuracy.append((healthy_bananas / total_bananas) if (total_bananas != 0) else .0)
    if train_mode and save_model:
        agent.save_model('./qnetwork', scores)
    return scores, accuracy