import numpy as np
import math
import random
import env.constants as constants


def random_predator_algorithm(observation_info):
    angle = np.random.uniform(0, 2 * np.pi)

    # Generate random length less than A
    length = np.random.uniform(0, constants.PREY_MAX_SPEED)

    # Calculate x and y based on angle and length
    a = np.random.uniform(0, 1)
    x = length * np.cos(angle)
    y = length * np.sin(angle)

    velocity = np.array([a,x, y], dtype=np.float32)
    return velocity

def math_predator_algorithm(ob_env):
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
    born_factor = np.random.uniform(0, 1)
    move_vector = [born_factor,move_vector[0],move_vector[1]]

    return move_vector
def math_prey_algorithm(ob_env):
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
        if random.random() <= 0.2:
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
    born_factor = np.random.uniform(0, 1)

    # 将避让捕食者和靠近食物的向量相结合
    final_vector = [
        born_factor,
        move_vector[0] + avoid_vector[0],
        move_vector[1] + avoid_vector[1]
    ]
    
    return final_vector