# prey.py
import random
import math
from creature import Creature
import constants

class Prey(Creature):
    def __init__(self, x, y, size):
        super().__init__(x, y, size, (0, 255, 0), constants.PREY_INITIAL_HEALTH, constants.PREY_MAX_HEALTH, constants.PREY_HEALTH_DECAY, constants.PREY_HEARING_RANGE)
        self.sight_range = constants.PREY_SIGHT_RANGE  # 使用新的视觉范围
        self.turn_counter = 0  # 用于记录逃跑时的计时器

    def draw(self, screen):
        self.reset_color()  # 重置颜色
        super().draw(screen)
        if self.selected:  # 如果被选中，显示视觉和听觉范围
            self.draw_sight_range(screen)
            self.draw_hearing_range(screen)
            observed_predator, observed_prey, observed_food, observed_obstacle, _ = self.observe_info(self.env_predators, self.env_prey, self.env_food, self.env_obstacles)
            self.highlight_targets(screen, observed_predator, observed_prey, observed_food, observed_obstacle)

    def move_strategy(self):
        Creature.reset_all_colors(self.env_predators + self.env_prey)
        observed_predator, observed_prey, observed_food, observed_obstacle, heard_sounds = self.observe_info(self.env_predators, self.env_prey, self.env_food, self.env_obstacles)
        move_vector = self.get_target(observed_predator, observed_prey, observed_food, observed_obstacle, heard_sounds)

        # 更新速度部分
        self.previous_velocity = self.velocity[:]
        self.velocity[0] += move_vector[0]  # 更新 x 方向的速度
        self.velocity[1] += move_vector[1]  # 更新 y 方向的速度

        # 限制速度
        speed = math.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2)
        if speed > constants.PREY_MAX_SPEED:
            self.velocity[0] = (self.velocity[0] / speed) * constants.PREY_MAX_SPEED
            self.velocity[1] = (self.velocity[1] / speed) * constants.PREY_MAX_SPEED

        # 避免速度归零
        if self.velocity[0] == 0 and self.velocity[1] == 0:
            self.velocity = [random.choice([-1, 1]), random.choice([-1, 1])]

        # 移动 Prey
        self.rect.x += self.velocity[0]
        self.rect.y += self.velocity[1]

    def get_target(self, observed_predator, observed_prey, observed_food, observed_obstacle, heard_sounds):
        move_vector = [0, 0]
        avoid_vector = [0, 0]

        # 远离捕食者
        if observed_predator:
            dx = self.rect.x - observed_predator.rect.x
            dy = self.rect.y - observed_predator.rect.y
            dist = self.distance_to(observed_predator)
            if dist > 0:  # 检查距离是否为零
                avoid_vector[0] += (dx / dist) * constants.PREY_EVASION_FACTOR
                avoid_vector[1] += (dy / dist) * constants.PREY_EVASION_FACTOR

            # 定期回头观察
            self.turn_counter += 1
            if self.turn_counter >= constants.PREY_TURN_INTERVAL:
                self.turn_counter = 0  # 重置计时器
                # 模拟回头观察：调整方向
                avoid_vector[0] += random.uniform(-0.5, 0.5)
                avoid_vector[1] += random.uniform(-0.5, 0.5)

        # 靠近食物
        if observed_food:
            dx = observed_food.rect.x - self.rect.x
            dy = observed_food.rect.y - self.rect.y
            dist = self.distance_to(observed_food)
            if dist > 0:  # 检查距离是否为零
                move_vector[0] += dx / dist
                move_vector[1] += dy / dist

        # 避免障碍物
        if observed_obstacle:
            dx = self.rect.x - observed_obstacle.rect.x
            dy = self.rect.y - observed_obstacle.rect.y
            dist = self.distance_to(observed_obstacle)
            if dist > 0:  # 检查距离是否为零
                avoid_vector[0] += dx / dist
                avoid_vector[1] += dy / dist

        # 利用听觉信息
        for sound_intensity, sound_direction in heard_sounds:
            move_vector[0] += sound_intensity * math.cos(sound_direction)
            move_vector[1] += sound_intensity * math.sin(sound_direction)

        # 随机移动
        if not observed_predator and not observed_food:
            if random.random() < constants.PREY_RANDOM_MOVE_CHANCE:
                angle = random.uniform(-math.pi, math.pi)
                move_vector[0] += math.cos(angle) * constants.PREY_RANDOM_MOVE_SPEED
                move_vector[1] += math.sin(angle) * constants.PREY_RANDOM_MOVE_SPEED

        # 将避让捕食者和靠近食物的向量相结合
        final_vector = [
            move_vector[0] + avoid_vector[0],
            move_vector[1] + avoid_vector[1]
        ]

        return final_vector

    def eat_food(self, foods):
        for food in foods:
            if self.rect.colliderect(food.rect):
                self.health += constants.FOOD_HEALTH_GAIN
                if self.health > self.max_health:
                    self.health = self.max_health
                foods.remove(food)
                return

    def crossbreed(self, other):
        child_x = (self.rect.x + other.rect.x) // 2
        child_y = (self.rect.y + other.rect.y) // 2
        child_size = constants.BLOCK_SIZE
        child = Prey(child_x, child_y, child_size)
        return child

    def mutate(self):
        self.velocity[0] = random.choice([-1, 1])
        self.velocity[1] = random.choice([-1, 1])

    def update_health(self):
        # 基础的健康值减少
        health_decay = self.health_decay

        # 根据速度变化计算加速度
        accel_x = self.velocity[0] - self.previous_velocity[0]
        accel_y = self.velocity[1] - self.previous_velocity[1]
        acceleration = math.sqrt(accel_x ** 2 + accel_y ** 2)
        
        # 根据加速度计算额外的健康值减少
        health_decay += acceleration * constants.PREY_ACCELERATION_HEALTH_DECAY_FACTOR

        self.health -= health_decay

        if self.health <= 0:
            self.health = 0
        elif self.health > self.max_health:
            self.health = self.max_health
