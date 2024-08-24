import math
from env.creature import Creature
import env.constants as constants
import random
class Predator(Creature):
    def __init__(self, x, y, size,name="pred",algorithm = "initalrandom"):
        super().__init__(x, y, size, (128,128,128), constants.PREDATOR_INITIAL_HEALTH, constants.PREDATOR_MAX_HEALTH, constants.PREDATOR_HEALTH_DECAY, constants.PREDATOR_HEARING_RANGE)
        self.sight_range = constants.PREDATOR_SIGHT_RANGE  # 使用新的视觉范围
        self.prey_list = []
        self.name = name
        self.type = 'predator'
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
            other_data = self.observe_info(self.env_predators, self.env_prey, self.env_food, self.env_obstacles)

            # # 解析矩阵，分别处理视距内的捕食者、猎物、食物和障碍物
            # observed_predator = [item for item in other_data if item[0] == 1]  # 捕食者
            # observed_prey = [item for item in other_data if item[0] == 2]  # 猎物
            # observed_food = [item for item in other_data if item[0] == 3]  # 食物
            # observed_obstacle = [item for item in other_data if item[0] == 4]  # 障碍物
            # self.highlight_targets(screen, observed_predator, observed_prey, observed_food, observed_obstacle)

    def set_prey_list(self, prey_list):
        self.prey_list = prey_list
    def get_observe_info(self):
        ob_env = self.observe_info(self.env_predators, self.env_prey, self.env_food, self.env_obstacles)
        return ob_env

    def move_strategy(self,move_vector):
        Creature.reset_all_colors(self.env_predators + self.env_prey)
        # ob_env = self.get_observe_info()
        # ob_env = self.observe_info(self.env_predators, self.env_prey, self.env_food, self.env_obstacles)
        move_vector = move_vector

        # 更新速度部分
        self.previous_velocity = self.velocity[:]
        self.velocity[0] = move_vector[0]  # 更新 x 方向的速度
        self.velocity[1] = move_vector[1]  # 更新 y 方向的速度

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

    def get_target(self, ob_env):
        move_vector = [0, 0]

        observed_prey = [item for item in ob_env if item[0] == 2]
        observed_food = [item for item in ob_env if item[0] == 3]
        sounds = [item for item in ob_env if item[0] == 9]
        if observed_prey:
            # 靠近猎物并加速
            closest_prey = min(observed_prey, key=lambda prey: math.sqrt(prey[1]**2 + prey[2]**2))
            dx = closest_prey[1]
            dy = closest_prey[2]
            dist = math.sqrt(dx**2 + dy**2)
            
            if dist > 0:  # 检查距离是否为零
                move_vector[0] += (dx / dist) * constants.PREDATOR_ACCELERATION_FACTOR
                move_vector[1] += (dy / dist) * constants.PREDATOR_ACCELERATION_FACTOR

        elif observed_food:
            # 靠近食物
            closest_food = min(observed_food, key=lambda food: math.sqrt(food[1]**2 + food[2]**2))
            dx = closest_food[1]
            dy = closest_food[2]
            dist = math.sqrt(dx**2 + dy**2)
            if dist > 0:  # 检查距离是否为零
                move_vector[0] += dx / dist
                move_vector[1] += dy / dist

        else:
            # 停下来并旋转观察周围
            if random.random() < constants.PREDATOR_ROTATION_CHANCE:
                angle = random.uniform(-math.pi, math.pi)
                move_vector[0] = math.cos(angle) * constants.PREDATOR_ROTATION_SPEED
                move_vector[1] = math.sin(angle) * constants.PREDATOR_ROTATION_SPEED
            else:
                move_vector[0] = 0
                move_vector[1] = 0

        # 利用听觉信息来影响移动策略
        for sound in sounds:
            sound_intensity = sound[1]
            sound_direction = sound[2]
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

    def crossbreed(self, other,name):
        child_x = (self.rect.x + other.rect.x) // 2
        child_y = (self.rect.y + other.rect.y) // 2
        # name = f"Pred{self.algorithm}_{next_pred_id}"  # 使用全局唯一ID生成名称
        child = Predator(child_x, child_y, constants.BLOCK_SIZE,name=name, algorithm=self.algorithm)
        
        return child

    def mutate(self):
        self.velocity[0] = random.choice([-1, 1])
        self.velocity[1] = random.choice([-1, 1])

    def update_health(self):
        # 基础的健康值减少
        health_decay = self.health_decay   # 将捕食者的生命值减少速度设置为猎物的两倍

        # 根据速度变化计算加速度
        accel_x = self.velocity[0] - self.previous_velocity[0]
        accel_y = self.velocity[1] - self.previous_velocity[1]
        acceleration = math.sqrt(accel_x ** 2 + accel_y ** 2)
        
        # 根据加速度计算额外的健康值减少
        health_decay += acceleration * constants.PREDATOR_ACCELERATION_HEALTH_DECAY_FACTOR
        self.health -= health_decay

        if self.health <= 0:
            self.health = 0
            self.is_alive = False
        elif self.health > self.max_health:
            self.health = self.max_health
