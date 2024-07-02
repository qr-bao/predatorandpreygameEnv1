# creature.py
import pygame
import random
import math
import constants
from abc import ABC, abstractmethod

class Creature(ABC):
    def __init__(self, x, y, size, color, initial_health, max_health, health_decay, hearing_range):
        self.rect = pygame.Rect(x, y, size, size)
        self.color = color
        self.original_color = color  # 保存原始颜色
        angle = random.uniform(0, 2 * math.pi)
        self.velocity = [math.cos(angle), math.sin(angle)]
        self.previous_velocity = self.velocity[:]  # 新增的
        self.health = initial_health
        self.max_health = max_health
        self.health_decay = health_decay
        self.sight_range = 200  # 初始值
        self.sight_angle = math.pi / 3  # 可视角度
        self.hearing_range = hearing_range  # 听觉范围
        self.selected = False  # 新增的
        self.iteration_counter = 0  # 新增的迭代计数器
    
    def increment_iteration(self):
        self.iteration_counter += 1
    
    def draw_sight_range(self, screen):
        end_x = self.rect.centerx + self.sight_range * math.cos(math.atan2(self.velocity[1], self.velocity[0]))
        end_y = self.rect.centery + self.sight_range * math.sin(math.atan2(self.velocity[1], self.velocity[0]))
        pygame.draw.line(screen, (255, 255, 0), self.rect.center, (end_x, end_y), 1)  # 画出视觉范围线
        pygame.draw.arc(screen, (255, 255, 0), (self.rect.centerx - self.sight_range, self.rect.centery - self.sight_range, 2 * self.sight_range, 2 * self.sight_range), 
                        math.atan2(self.velocity[1], self.velocity[0]) - self.sight_angle / 2, 
                        math.atan2(self.velocity[1], self.velocity[0]) + self.sight_angle / 2, 1)  # 画出视觉范围弧线

    def draw_hearing_range(self, screen):
        pygame.draw.circle(screen, (128, 128, 128), self.rect.center, self.hearing_range, 1)  # 画出听觉范围

    def highlight_targets(self, screen, observed_predator, observed_prey, observed_food, observed_obstacle):
        if observed_predator:
            observed_predator.color = (255, 255, 0)  # 高亮捕食者
        if observed_prey:
            observed_prey.color = (0, 0, 255)  # 高亮猎物
        if observed_food:
            observed_food.color = (255, 0, 0)  # 高亮食物
        if observed_obstacle:
            pygame.draw.rect(screen, (255, 255, 255), observed_obstacle.rect, 2)  # 高亮障碍物框架

    def reset_color(self):
        self.color = self.original_color  # 重置颜色

    @classmethod
    def reset_all_colors(cls, creatures):
        for creature in creatures:
            creature.reset_color()

    def move(self, control_panel_width, screen_width, screen_height, obstacles):
        # 保存原始位置
        original_position = self.rect.topleft

        # 子类具体实现移动策略
        self.move_strategy()

        # 碰撞检测，防止小方块移出游戏空间
        if self.rect.left < control_panel_width or self.rect.right > screen_width:
            self.velocity[0] = -self.velocity[0]
        if self.rect.top < 0 or self.rect.bottom > screen_height:
            self.velocity[1] = -self.velocity[1]

        # 检查是否与障碍物碰撞
        if any(self.rect.colliderect(obs.rect) for obs in obstacles):
            # 如果碰撞，恢复到原始位置并反转速度
            self.rect.topleft = original_position
            self.velocity[0] = -self.velocity[0]
            self.velocity[1] = -self.velocity[1]

    def observe_info(self, env_predators, env_prey, env_food, env_obstacles):
        def is_in_sight(target):
            dx = target.rect.centerx - self.rect.centerx
            dy = target.rect.centery - self.rect.centery
            distance = math.sqrt(dx**2 + dy**2)
            angle_to_target = math.atan2(dy, dx)
            relative_angle = angle_to_target - math.atan2(self.velocity[1], self.velocity[0])
            relative_angle = (relative_angle + math.pi) % (2 * math.pi) - math.pi  # 将角度限制在 [-pi, pi]
            return (distance <= self.sight_range) and (abs(relative_angle) <= self.sight_angle / 2)

        def is_occluded(target, obstacles):
            for obstacle in obstacles:
                if self.rect.colliderect(obstacle.rect):
                    continue
                if self.line_intersects_rect((self.rect.centerx, self.rect.centery), (target.rect.centerx, target.rect.centery), obstacle.rect):
                    return True
            return False

        def find_first_visible(targets, obstacles):
            visible_targets = sorted(targets, key=lambda t: self.distance_to(t))
            for target in visible_targets:
                if is_in_sight(target) and not is_occluded(target, obstacles):
                    return target
            return None

        def is_in_hearing_range(target):
            dx = target.rect.centerx - self.rect.centerx
            dy = target.rect.centery - self.rect.centery
            distance = math.sqrt(dx**2 + dy**2)
            return distance <= self.hearing_range

        def get_sound_intensity(target):
            dx = target.rect.centerx - self.rect.centerx
            dy = target.rect.centery - self.rect.centery
            distance = math.sqrt(dx**2 + dy**2)
            return max(0, (self.hearing_range - distance) / self.hearing_range)

        def get_sound_direction(target):
            dx = target.rect.centerx - self.rect.centerx
            dy = target.rect.centery - self.rect.centery
            angle_to_target = math.atan2(dy, dx)
            relative_angle = angle_to_target - math.atan2(self.velocity[1], self.velocity[0])
            relative_angle = (relative_angle + math.pi) % (2 * math.pi) - math.pi  # 将角度限制在 [-pi, pi]
            return relative_angle

        # 视觉感知
        observed_predator = find_first_visible(env_predators, env_obstacles)
        observed_prey = find_first_visible(env_prey, env_obstacles)
        observed_food = find_first_visible(env_food, env_obstacles)
        observed_obstacle = find_first_visible(env_obstacles, env_obstacles)

        # 听觉感知
        heard_entities = [entity for entity in env_predators + env_prey if is_in_hearing_range(entity)]
        heard_sounds = [(get_sound_intensity(entity), get_sound_direction(entity)) for entity in heard_entities]

        return observed_predator, observed_prey, observed_food, observed_obstacle, heard_sounds

    def distance_to(self, other):
        dx = other.rect.centerx - self.rect.centerx
        dy = other.rect.centery - self.rect.centery
        return (dx ** 2 + dy ** 2) ** 0.5

    def line_intersects_rect(self, p1, p2, rect):
        lines = [
            ((rect.left, rect.top), (rect.right, rect.top)),
            ((rect.right, rect.top), (rect.right, rect.bottom)),
            ((rect.right, rect.bottom), (rect.left, rect.bottom)),
            ((rect.left, rect.bottom), (rect.left, rect.top))
        ]
        for line in lines:
            if self.lines_intersect(p1, p2, line[0], line[1]):
                return True
        return False

    def lines_intersect(self, p1, p2, p3, p4):
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    @abstractmethod
    def move_strategy(self):
        pass

    @abstractmethod
    def get_target(self, observed_predator, observed_prey, observed_food, observed_obstacle):
        pass

    def update_health(self):
        self.health -= self.health_decay
        if self.health <= 0:
            self.health = 0
        elif self.health > self.max_health:
            self.health = self.max_health

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)

    def is_dead(self):
        return self.health <= 0
