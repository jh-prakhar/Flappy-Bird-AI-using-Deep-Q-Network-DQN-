import gymnasium as gym
import flappy_bird_gymnasium
import pygame

# Creating Our Env
env = gym.make("FlappyBird-v0", render_mode = "human")
state, info = env.reset()
done = False
11
# Initiallizing PyGame keyword
pygame.init()
screen = pygame.display.get_surface() # Gym has already created a window

while not done:
    action = 0 # defalut -> 0 is no flap & 1 is flap

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                action = 1 # flap
    

    state, reward, done, truncated, info = env.step(action)
    env.render()
env.close()
pygame.quit()