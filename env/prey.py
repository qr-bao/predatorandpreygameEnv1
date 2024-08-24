# prey.py
import random
import math
from env.creature import Creature
import env.constants as constants
import numpy as np
class Prey(Creature):
    def __init__(self, x, y, size,name="prey",algorithm = "initalrandom"):
        super().__init__(x, y, size, (255, 165, 0), constants.PREY_INITIAL_HEALTH, constants.PREY_MAX_HEALTH, constants.PREY_HEALTH_DECAY, constants.PREY_HEARING_RANGE)
        self.sight_range = constants.PREY_SIGHT_RANGE  # 使用新的视觉范围
        self.turn_counter = 0  # 用于记录逃跑时的计时器
        self.name = name
        self.type = 'prey'
        self.algorithm = algorithm
    #     self.log()
    # def log(self):
    #     print(self.name,end="    ")
    #     print(self.algorithm)
    def draw(self, screen):
        self.reset_color()  # 重置颜色
        super().draw(screen)
        if self.selected:  # 如果被选中，显示视觉和听觉范围
            self.draw_sight_range(screen)
            self.draw_hearing_range(screen)

            # 获取感知信息矩阵
            other_data = self.observe_info(self.env_predators, self.env_prey, self.env_food, self.env_obstacles)
            
            # # 解析矩阵，分别处理视距内的捕食者、猎物、食物和障碍物
            # observed_predator = [item for item in other_data if item[0] == 1]  # 捕食者
            # observed_prey = [item for item in other_data if item[0] == 2]  # 猎物
            # observed_food = [item for item in other_data if item[0] == 3]  # 食物
            # observed_obstacle = [item for item in other_data if item[0] == 4]  # 障碍物
            
            # # 选中时高亮目标
            # self.highlight_targets(screen, observed_predator, observed_prey, observed_food, observed_obstacle)

    def get_observe_info(self):
        ob_env = self.observe_info(self.env_predators, self.env_prey, self.env_food, self.env_obstacles)
        # print(ob_env)
        return ob_env
    def move_strategy(self,move_vector):
        Creature.reset_all_colors(self.env_predators + self.env_prey)
        # ob_env = self.observe_info(self.env_predators, self.env_prey, self.env_food, self.env_obstacles)
        
        
        # move_vector = self.get_target(ob_env)

        # 更新速度部分
        self.previous_velocity = self.velocity[:]
        self.velocity[0] = move_vector[0]  # 更新 x 方向的速度
        self.velocity[1] = move_vector[1]  # 更新 y 方向的速度

        # 限制速度
        speed = math.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2)
        if speed > constants.PREY_MAX_SPEED:
            self.velocity[0] = (self.velocity[0] / speed) * constants.PREY_MAX_SPEED
            self.velocity[1] = (self.velocity[1] / speed) * constants.PREY_MAX_SPEED

        # 避免速度归零
        if self.velocity[0] == 0 and self.velocity[1] == 0:
            self.velocity = [random.choice([-1, 1]), random.choice([-1, 1])]

        # 细化移动步骤并检测碰撞
        step_size = 5  # 可以调整步长大小
        total_steps = max(abs(self.velocity[0]), abs(self.velocity[1])) // step_size

        for step in range(int(total_steps)):
            self.rect.x += self.velocity[0] / total_steps
            self.rect.y += self.velocity[1] / total_steps
            self.eat_food(self.env_food)  # 在每一步检查碰撞

        # 移动 Prey
        self.rect.x += self.velocity[0]
        self.rect.y += self.velocity[1]

    def get_target(self, ob_env):
        # 解析矩阵，分别处理视距内的捕食者、猎物、食物和障碍物
        observed_predator = [item for item in ob_env if item[0] == 1]  # 捕食者
        observed_prey = [item for item in ob_env if item[0] == 2]  # 猎物
        observed_food = [item for item in ob_env if item[0] == 3]  # 食物
        observed_obstacle = [item for item in ob_env if item[0] == 4]  # 障碍物
        sounds = [item for item in ob_env if item[0] == 9]
        move_vector = [0, 0]
        avoid_vector = [0, 0]

        # 远离捕食者
        if observed_predator:
            closest_predator = min(observed_predator, key=lambda predator: math.sqrt(predator[1]**2 + predator[2]**2))
            
            dx = closest_predator[1]
            dy = closest_predator[2]
            dist = math.sqrt(dx**2 + dy**2)
            if dist > 0:  # 检查距离是否为零
                avoid_vector[0] -= (dx / dist) * constants.PREY_EVASION_FACTOR
                avoid_vector[1] -= (dy / dist) * constants.PREY_EVASION_FACTOR

            # 定期回头观察
            self.turn_counter += 1
            if self.turn_counter >= constants.PREY_TURN_INTERVAL:
                self.turn_counter = 0  # 重置计时器
                # 模拟回头观察：调整方向
                avoid_vector[0] += random.uniform(-0.5, 0.5)
                avoid_vector[1] += random.uniform(-0.5, 0.5)

        # 靠近食物
        if observed_food:
            closest_food = min(observed_food, key=lambda food: math.sqrt(food[1]**2 + food[2]**2))

            dx = closest_food[1]
            dy = closest_food[2]
            dist = math.sqrt(dx**2 + dy**2)
            if dist > 0:  # 检查距离是否为零
                move_vector[0] += dx / dist
                move_vector[1] += dy / dist

        # 避免障碍物
        if observed_obstacle:
            closest_obstacle = min(observed_obstacle, key=lambda obstacle: math.sqrt(obstacle[1]**2 + obstacle[2]**2))

            dx = closest_obstacle[1]
            dy = closest_obstacle[2]
            dist = math.sqrt(dx**2 + dy**2)
            if dist > 0:  # 检查距离是否为零
                avoid_vector[0] -= dx / dist
                avoid_vector[1] -= dy / dist

        # 利用听觉信息
        for sound in sounds:
            sound_intensity = sound[1]
            sound_direction = sound[2]
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
                foods.remove(food)  # 从foods列表中移除被吃掉的食物
                return


    def crossbreed(self, other,name):
        child_x = (self.rect.x + other.rect.x) // 2
        child_y = (self.rect.y + other.rect.y) // 2
        # name = f"Prey{self.algorithm}_{next_prey_id}"  # 使用全局唯一ID生成名称
        child = Prey(child_x, child_y, constants.BLOCK_SIZE,name=name, algorithm=self.algorithm)

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
