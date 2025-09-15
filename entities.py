import pygame
import random
import math
from abilities import Ability

# Window settings
WIDTH, HEIGHT = 800, 600
FPS = 60

RED = (200, 60, 60) # Boss
GREEN = (80, 180, 80)   # warriors
BLUE = (80, 140, 220)   # archers
PINK = (255, 105, 180) # healers
BLACK = (24, 24, 28)
BROWN = (102, 51, 0)

# ------------------- BOSS -------------------
class Boss:
    def __init__(self, x=WIDTH//2, y=100, health=1000, color=RED):
        self.x = x
        self.y = y
        self.name = "Boss"
        self.health = health
        self.max_health = health
        self.color = color
        self.radius = 30
        self.ticks = 0
        self.abilities = {
            "Frontal Cone": Ability("Frontal Cone", damage=125, range=1000, aoe_shape='cone', aoe_radius=200, cooldown=30),
            "Tank Buster": Ability("Tank Buster", damage=125, range=20, aoe_shape='single-target', aoe_radius=0, cooldown=10),
            "Fireball": Ability("Fireball", damage=60, range=1000, aoe_shape='aoe-ground-effect', aoe_radius=5, cooldown=40)}
        self.active_ability = None
        self.active_ability_target = None
        self.affected_agents = None
        self.fireball_zones = []
        self.fireball_radius = 8
        self.ability_ticks_remaining = 0
        


    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)
        if self.active_ability is not None and self.active_ability_target is not None:
            points = self.get_cone_vertices(self.active_ability, self.active_ability_target)
            pygame.draw.polygon(screen, (255, 100, 100), points)
        if getattr(self, "fireball_zones", None):
            for (fx, fy) in self.fireball_zones:
                pygame.draw.circle(screen, (255, 0, 0), (int(fx), int(fy)), self.fireball_radius * 2, 2)

    def move(self, agents):
        # move towards the agent with the lowest hp
        pass

    def step(self, agents):
        alive_agents = [a for a in agents if a.health > 0]
        nearest_agent = min(alive_agents,key=lambda a:math.hypot(a.x - self.x, a.y - self.y))
        dx = nearest_agent.x - self.x
        dy = nearest_agent.y - self.y
        if self.active_ability is None:
            for ability in self.abilities.values():
                if ability.is_ready(self.ticks):
                    if ability.name == "Tank Buster" and math.hypot(dx,dy) <= ability.range:
                            nearest_agent.health -= ability.damage
                            print(f"{self.name} casts {ability.name} on {nearest_agent.role} for {ability.damage} damage")
                            ability.last_used_tick = self.ticks
                            continue
                    elif ability.name == "Frontal Cone" and math.hypot(dx,dy) <= ability.range:
                        self.active_ability_target = nearest_agent
                        self.active_ability = ability
                        self.ability_ticks_remaining = 15
                        ability.last_used_tick = self.ticks
                    elif ability.name == 'Fireball' and math.hypot(dx,dy) <= ability.range:
                        self.fireball_zones = [(a.x,a.y) for a in alive_agents]
                        self.active_ability = ability
                        self.ability_ticks_remaining = 10
                        ability.last_used_tick = self.ticks

        else:
            if self.ability_ticks_remaining > 0:
                self.ability_ticks_remaining -= 1
            else:
                if self.active_ability.name == 'Frontal Cone':
                    for agent in alive_agents:
                        if self.agent_in_cone(agent, self.active_ability, self.active_ability_target):
                            agent.health -= self.active_ability.damage
                            print(f"{self.name} hits {agent.role} with {self.active_ability.name} for {self.active_ability.damage} damage")
                    self.active_ability = None
                    self.active_ability_target = None
                elif self.active_ability.name == 'Fireball':
                    for (fx,fy) in self.fireball_zones:
                        for agent in alive_agents:
                            if self.agent_in_fireball(agent, (fx,fy), self.fireball_radius):
                                agent.health -= self.active_ability.damage
                                print(f"{self.name} hits {agent.role} with {self.active_ability.name} for {self.active_ability.damage} damage")
                    self.active_ability = None
                    self.fireball_zones = []
        self.ticks += 1

    def get_cone_vertices(self, ability, target_agent):
        # 1. Compute direction vector toward target
        dx = target_agent.x - self.x
        dy = target_agent.y - self.y
        dist = math.hypot(dx, dy)
        if dist == 0:
            dist = 1  # prevent division by zero
        dir_x = dx / dist
        dir_y = dy / dist

        # 2. Half-angle of the cone in radians
        half_angle = math.radians(20) / 2 

        # 3. Compute left and right directions using rotation formula
        left_dx = dir_x * math.cos(half_angle) - dir_y * math.sin(half_angle)
        left_dy = dir_x * math.sin(half_angle) + dir_y * math.cos(half_angle)
        right_dx = dir_x * math.cos(-half_angle) - dir_y * math.sin(-half_angle)
        right_dy = dir_x * math.sin(-half_angle) + dir_y * math.cos(-half_angle)

        # 4. Compute triangle points
        tip = (self.x, self.y)
        left_point = (self.x + left_dx * ability.aoe_radius, self.y + left_dy * ability.aoe_radius)
        right_point = (self.x + right_dx * ability.aoe_radius, self.y + right_dy * ability.aoe_radius)

        return [tip, left_point, right_point]

    def agent_in_cone(self, agent, ability, target):
        vec_to_agent = (agent.x - self.x, agent.y - self.y)
        dx = target.x - self.x
        dy = target.y - self.y
        dist = math.hypot(vec_to_agent[0], vec_to_agent[1])
        if dist == 0:
            return True
        else:
            cos_theta = (dx * vec_to_agent[0] + dy * vec_to_agent[1]) / (math.hypot(dx, dy) * dist)
            cos_theta = max(-1, min(1, cos_theta))
            angle_to_agent = math.acos(cos_theta)
        return dist <= ability.aoe_radius and abs(angle_to_agent) < math.radians(20) / 2
    
    def agent_in_fireball(self, agent, center, radius):
        dist = math.hypot(agent.x - center[0], agent.y-center[1])
        return dist <= radius



# ------------------- AGENTS -------------------
class Agent:
    role_colors = {"warrior": BROWN, "archer": GREEN, "healer": PINK} # add classes later
    role_damage = {"warrior": 10, "archer": 7.5, "healer": 5} # add classes later
    role_heal = {"warrior": 0, "archer": 0, "healer": 10}
    role_health = {"warrior": 125, "archer": 135, "healer": 80} 
    role_speed = {"warrior": 1.5, "archer": 3, "healer": 1} 

    def __init__(self, x,y,role):
        self.x = x
        self.y = y
        self.radius = 10
        self.role = role
        self.color = self.role_colors[role]
        self.health = self.role_health[role]
        self.max_health = self.role_health[role]
        self.damage_done = 0
        self.healing_done = 0
        self.speed = self.role_speed[role]

    @property
    def is_alive(self):
        return self.health > 0

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)

    def attack_or_heal(self, target , allies):
        MELEE_RANGE = 40
        PROJECTILE_RANGE = 120
        MAX_PROJECTILE_RANGE = 240
        HEAL_RANGE = 200
        miss_prob = 0.0
        dist = math.sqrt((self.x - target.x)**2 + (self.y - target.y)**2)
        # TODO: implement attack or heal for each role
        if self.role == "warrior":
            if dist <= MELEE_RANGE:
                target.health -= self.role_damage[self.role]
                self.damage_done += self.role_damage[self.role]
                print(f"{self.role} at ({self.x:.1f},{self.y:.1f}) hit the boss! Boss HP: {target.health}")

        elif self.role == "archer":
            if dist > MAX_PROJECTILE_RANGE:
                miss_prob = 1.0
            elif dist <= PROJECTILE_RANGE:
                miss_prob = 0.0
            else:
                miss_prob = 1 - dist/PROJECTILE_RANGE
            if random.random() < miss_prob:
                #print(f"{self.role} missed their attack against {target}, due to a range of {dist} resulting in a miss probability of {miss_prob}")
                pass
            else:
                target.health -= self.role_damage[self.role]
                self.damage_done += self.role_damage[self.role]
                #print(f"{self.role} at ({self.x:.1f},{self.y:.1f}) hit the boss! Boss HP: {target.health}")

        elif self.role == "healer":
            nearest_ally = None
            nearest_dist = float("inf")

            for ally in allies:
                if ally is self:
                    continue
                if ally.health < ally.max_health:
                    dist = math.sqrt((self.x - ally.x)**2 + (self.y - ally.y)**2)
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_ally = ally
            if nearest_ally != None and nearest_dist < HEAL_RANGE:
                nearest_ally.health += self.role_heal[self.role]
                self.healing_done += self.role_heal[self.role]
                nearest_ally.health = min(nearest_ally.max_health, nearest_ally.health + self.role_heal[self.role])
                print(f"{self.role} healed {nearest_ally.role} for {self.role_heal[self.role]}! {nearest_ally.role} HP: {nearest_ally.health}")


    
    def move(self, goal_x = None, goal_y = None):
        if goal_x is None or goal_y is None:
            # Random wandering
            angle = random.uniform(0, 2*math.pi)
            dx, dy = math.cos(angle), math.sin(angle)
        else:
            dx = goal_x - self.x
            dy = goal_y - self.y
            dist = math.hypot(dx, dy)
            if dist > 0:
                dx, dy = dx / dist, dy / dist
            else:
                dx, dy = 0, 0

        # Move with role-specific speed
        self.x += dx * self.speed
        self.y += dy * self.speed

        # Chance for random mutation
        if random.random() < 0.05: 
            angle = random.uniform(0, 2*math.pi)
            self.x += math.cos(angle) * self.speed
            self.y += math.sin(angle) * self.speed

        # Ensure arena limits are kept
        self.x = max(self.radius, min(WIDTH - self.radius, self.x))
        self.y = max(self.radius, min(HEIGHT - self.radius, self.y))


