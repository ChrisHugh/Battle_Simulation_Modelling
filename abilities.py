class Ability:
    def __init__(self, name, damage, range, aoe_shape, aoe_radius, cooldown):
        self.name = name
        self.damage = damage
        self.range = range
        self.aoe_shape = aoe_shape
        self.aoe_radius = aoe_radius
        self.cooldown = cooldown
        self.last_used_tick = -cooldown

    def is_ready(self, current_tick):
        return current_tick - self.last_used_tick >= self.cooldown
