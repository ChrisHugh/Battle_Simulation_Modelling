# arena.py
import pygame
import sys
from entities import Boss, Agent  # import your classes

WIDTH, HEIGHT = 800, 600
FPS = 5

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Battle Arena Test")
clock = pygame.time.Clock()

# Create boss
boss = Boss(x=WIDTH // 2, y=100, health=100)

# Create agents dynamically
agent_roles = ["warrior", "archer", "healer"]
agents = []
for i, role in enumerate(agent_roles):
    x = WIDTH // 2 + i * 30  # offset so agents don’t spawn stacked
    y = HEIGHT - 100
    agents.append(Agent(role=role, x=x, y=y))

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Agents move toward boss
    agents = [a for a in agents if a.is_alive]
    for agent in agents:
        agent.move(boss)

    # Agents attack or heal
    for agent in agents:
        agent.attack_or_heal(boss, agents)

    # Boss step (Tank Buster ability only for now)
    boss.step(agents)

    # Draw everything
    screen.fill((24, 24, 28))
    boss.draw(screen)
    for agent in agents:
        agent.draw(screen)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()
