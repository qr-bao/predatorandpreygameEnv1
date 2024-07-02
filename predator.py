import math
from creature import Creature
import constants
import random
class Predator(Creature):
    def __init__(self, x, y, size):
        super().__init__(x, y, size, (255, 0, 0), constants.PREDATOR_INITIAL_HEALTH, constants.PREDATOR_MAX_HEALTH, constants.PREDATOR_HEALTH_DECAY, constants.PREDATOR_HEARING_RANGE)
        self.sight_range = constants.PREDATOR_SIGHT_RANGE  # 使用新的视觉范围
        self.prey_list = []

    def draw(self, screen):
        self.reset_color()  # 重置颜色
        super().draw(screen)
        if self.selected:  # 如果被选中，显示视觉和听觉范围
            self.draw_sight_range(screen)
            self.draw_hearing_range(screen)
            observed_predator, observed_prey, observed_food, observed_obstacle, _ = self.observe_info(self.env_predators, self.env_prey, self.env_food, self.env_obstacles)
            self.highlight_targets(screen, observed_predator, observed_prey, observed_food, observed_obstacle)

    def set_prey_list(self, prey_list):
        self.prey_list = prey_list

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
        if speed > constants.PREDATOR_MAX_SPEED:
            self.velocity[0] = (self.velocity[0] / speed) * constants.PREDATOR_MAX_SPEED
            self.velocity[1] = (self.velocity[1] / speed) * constants.PREDATOR_MAX_SPEED

        # 避免速度归零
        if self.velocity[0] == 0 and self.velocity[1] == 0:
            self.velocity = [random.choice([-1, 1]), random.choice([-1, 1])]

        # 移动 Predator
        self.rect.x += self.velocity[0]
        self.rect.y += self.velocity[1]

    def get_target(self, observed_predator, observed_prey, observed_food, observed_obstacle, heard_sounds):
        move_vector = [0, 0]

        if observed_prey:
            # 靠近猎物并加速
            dx = observed_prey.rect.x - self.rect.x
            dy = observed_prey.rect.y - self.rect.y
            dist = self.distance_to(observed_prey)
            
            if dist > 0:  # 检查距离是否为零
                move_vector[0] += (dx / dist) * constants.PREDATOR_ACCELERATION_FACTOR
                move_vector[1] += (dy / dist) * constants.PREDATOR_ACCELERATION_FACTOR

        elif observed_food:
            # 靠近食物
            dx = observed_food.rect.x - self.rect.x
            dy = observed_food.rect.y - self.rect.y
            dist = self.distance_to(observed_food)
            if dist > 0:  # 检查距离是否为零
                move_vector[0] += dx / dist
                move_vector[1] += dy / dist
        
        else:
            # 停下来并旋转观察周围
            if random.random() < constants.PREDATOR_ROTATION_CHANCE:
                angle = random.uniform(-math.pi, math.pi)
                self.velocity[0] = math.cos(angle) * constants.PREDATOR_ROTATION_SPEED
                self.velocity[1] = math.sin(angle) * constants.PREDATOR_ROTATION_SPEED
            else:
                move_vector[0] = 0
                move_vector[1] = 0

        # 利用听觉信息来影响移动策略
        for sound_intensity, sound_direction in heard_sounds:
            move_vector[0] += sound_intensity * math.cos(sound_direction)
            move_vector[1] += sound_intensity * math.sin(sound_direction)

        return move_vector

    def hunt_prey(self, prey_list):
        for prey in prey_list:
            if self.rect.colliderect(prey.rect):
                self.health += prey.health * constants.PREDATOR_HEALTH_GAIN_FACTOR
                if self.health > self.max_health:
                    self.health = self.max_health
                prey.health = 0  # 猎物死亡
                prey_list.remove(prey)
                return

    def crossbreed(self, other):
        child_x = (self.rect.x + other.rect.x) // 2
        child_y = (self.rect.y + other.rect.y) // 2
        child_size = constants.BLOCK_SIZE
        child = Predator(child_x, child_y, child_size)
        return child

    def mutate(self):
        self.velocity[0] = random.choice([-1, 1])
        self.velocity[1] = random.choice([-1, 1])

    def update_health(self):
        # 基础的健康值减少
        health_decay = self.health_decay * 2  # 将捕食者的生命值减少速度设置为猎物的两倍

        # 根据速度变化计算加速度
        accel_x = self.velocity[0] - self.previous_velocity[0]
        accel_y = self.velocity[1] - self.previous_velocity[1]
        acceleration = math.sqrt(accel_x ** 2 + accel_y ** 2)
        
        # 根据加速度计算额外的健康值减少
        health_decay += acceleration * constants.PREDATOR_ACCELERATION_HEALTH_DECAY_FACTOR

        self.health -= health_decay

        if self.health <= 0:
            self.health = 0
        elif self.health > self.max_health:
            self.health = self.max_health
