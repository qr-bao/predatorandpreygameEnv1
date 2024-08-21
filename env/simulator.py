# simulator.py
import random
import pygame
from env.predator import Predator
from env.prey import Prey
from env.food import Food
from env.obstacle import Obstacle
import env.constants as constants

class Simulator:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.predators = []
        self.preys = []
        self.foods = []
        self.obstacles = []
        self.selected_agent = None
        self.dead_predator_count = 0  # 新增变量记录死亡的捕食者数量
        self.food_generation_timer = 0  # 初始化食物生成计时器
        self.food_iteration_count = 0
        self.next_predator_id = 1  # 用于生成唯一的捕食者名称
        self.next_prey_id = 1  # 用于生成唯一的猎物名称
        self.agent_status = {}  # Dictionary to track alive status
    def initialize(self,all_pred_algorithms,all_prey_algorithms):
        self.initialize_obstacles()
        self.initialize_agents(predAlgorithms =all_pred_algorithms,preyAlgorithms = all_prey_algorithms)
        self.initialize_food()
        self.food_generation_timer = 0  # 重置计时器
        self.dead_predator_count = 0  # 新增变量记录死亡的捕食者数量
        self.food_iteration_count = 0
        self.next_predator_id = 1  # 用于生成唯一的捕食者名称
        self.next_prey_id = 1  # 用于生成唯一的猎物名称



    def initialize_obstacles(self):
        self.obstacles = []
        num_obstacles = random.randint(5, 10)
        for _ in range(num_obstacles):
            while True:
                width = random.randint(50, 200)
                height = random.randint(50, 200)
                x = random.randint(constants.CONTROL_PANEL_WIDTH, self.screen_width - width)
                y = random.randint(0, self.screen_height - height)
                new_obstacle = Obstacle(x, y, width, height)

                if not any(new_obstacle.rect.colliderect(obs.rect) for obs in self.obstacles):
                    self.obstacles.append(new_obstacle)
                    break

    def initialize_agents(self,predAlgorithms = ["random" for _ in range(constants.NUM_PREDATORS)],preyAlgorithms = ["random" for _ in range(constants.NUM_PREY)]):
        self.predators = []
        self.preys = []
        if len(predAlgorithms)!= constants.NUM_PREDATORS and len(preyAlgorithms)!= constants.NUM_PREY:
            print("algorithms lens not equal number of agent")
        for predalgorithm in predAlgorithms:
            self.generate_predator(algorithm=predalgorithm)

        for preyalgorithm in preyAlgorithms:
            self.generate_prey(algorithm=preyalgorithm)

    def initialize_food(self):
        self.foods = []

        for _ in range(constants.NUM_FOOD):
            self.generate_food()

    def generate_prey(self,algorithm="random"):
        while True:
            x = random.randint(constants.CONTROL_PANEL_WIDTH, self.screen_width - constants.BLOCK_SIZE)
            y = random.randint(0, self.screen_height - constants.BLOCK_SIZE)
            name = f"Prey{algorithm}_{self.next_prey_id}"  # 使用全局唯一ID生成名称
            new_prey = Prey(x, y, constants.BLOCK_SIZE,name=name,algorithm=algorithm)

            if not any(new_prey.rect.colliderect(obs.rect) for obs in self.obstacles):
                self.preys.append(new_prey)
                self.agent_status[new_prey.name] = True  # Mark as alive
                self.next_prey_id += 1  # 在成功生成猎物后增加ID
                break

    def generate_predator(self,algorithm="random"):
        while True:
            x = random.randint(constants.CONTROL_PANEL_WIDTH, self.screen_width - constants.BLOCK_SIZE)
            y = random.randint(0, self.screen_height - constants.BLOCK_SIZE)
            name = f"Pred{algorithm}_{self.next_predator_id}"  # 使用全局唯一ID生成名称
            new_predator = Predator(x, y, constants.BLOCK_SIZE,name=name,algorithm=algorithm)
            if not any(new_predator.rect.colliderect(obs.rect) for obs in self.obstacles):
                self.predators.append(new_predator)
                self.agent_status[new_predator.name] = True  # Mark as alive
                self.next_predator_id += 1  # 在成功生成猎物后增加ID
                break

    def generate_food(self):
        while True:
            x = random.randint(constants.CENTER_AREA_X_START, constants.CENTER_AREA_X_START + constants.CENTER_AREA_WIDTH - constants.FOOD_SIZE)
            y = random.randint(constants.CENTER_AREA_Y_START, constants.CENTER_AREA_Y_START + constants.CENTER_AREA_HEIGHT - constants.FOOD_SIZE)
            new_food = Food(x, y, constants.FOOD_SIZE)
            if not any(new_food.rect.colliderect(obs.rect) for obs in self.obstacles):
                self.foods.append(new_food)
                break

    def breedPrey(self, prey,otherPrey):
        if prey.iteration_counter < constants.PREY_REPRODUCTION_ITERATION_THRESHOLD:
            return
        if prey.health < constants.PREY_MIN_HEALTH_FOR_REPRODUCTION:
            return
        if random.random() > constants.PREY_REPRODUCTION_PROBABILITY:
            return

        other_prey = otherPrey
        if other_prey.health >= constants.PREY_MIN_HEALTH_FOR_REPRODUCTION:
            child = prey.crossbreed(other_prey,self.next_prey_id)
            if random.random() < constants.MUTATION_CHANCE:
                child.mutate()
            self.ensure_no_collision(child)
            self.preys.append(child)
            self.next_prey_id += 1  # 成功生成猎物后增加ID
        # # 生成新的猎物
        # child_x = (prey1.rect.x + prey2.rect.x) // 2
        # child_y = (prey1.rect.y + prey2.rect.y) // 2
        # name = f"Prey{prey1.algorithm}_{self.next_prey_id}"  # 使用全局唯一ID生成名称

        # child = Prey(child_x, child_y, constants.BLOCK_SIZE,name=name, algorithm=prey1.algorithm)
        # # 确保新生成的猎物不与其他障碍物或智能体发生碰撞
        #     # 确保新生成的猎物不会与任何障碍物重叠
        # if not any(child.rect.colliderect(obs.rect) for obs in self.obstacles):
        #     self.preys.append(child)
        #     self.next_prey_id += 1  # 成功生成猎物后增加ID

    def breedPredator(self, predator,otherpredator):
        if predator.iteration_counter < constants.PREDATOR_REPRODUCTION_ITERATION_THRESHOLD:
            return
        if predator.health < constants.PREDATOR_MIN_HEALTH_FOR_REPRODUCTION:
            return
        if random.random() > constants.PREDATOR_REPRODUCTION_PROBABILITY:
            return

        other_predator = otherpredator
        if other_predator.health >= constants.PREDATOR_MIN_HEALTH_FOR_REPRODUCTION:
            child = predator.crossbreed(other_predator,self.next_predator_id)
            if random.random() < constants.MUTATION_CHANCE:
                child.mutate()
            self.ensure_no_collision(child)
            self.predators.append(child)
            self.next_predator_id += 1  # 成功生成猎物后增加ID


    def applyGeneticAlgorithm(self):
        new_prey_born = 0
        new_predator_born = 0

        for prey in self.preys:
            initial_prey_count = len(self.preys)
            self.breedPrey(prey)
            if len(self.preys) > initial_prey_count:
                new_prey_born += 1

        for predator in self.predators:
            initial_predator_count = len(self.predators)
            self.breedPredator(predator)
            if len(self.predators) > initial_predator_count:
                new_predator_born += 1

        return new_prey_born, new_predator_born

    def generate_agent(self):
        self.applyGeneticAlgorithm()

    def ensure_no_collision(self, agent):
        while any(agent.rect.colliderect(obs.rect) for obs in self.obstacles):
            agent.rect.x = random.randint(constants.CONTROL_PANEL_WIDTH, self.screen_width - agent.rect.width)
            agent.rect.y = random.randint(0, self.screen_height - agent.rect.height)

    def add_food(self):
        self.food_iteration_count += 1
        if self.food_iteration_count % constants.FOOD_GENERATION_INTERVAL == 0:
            self.generate_random_food()
            self.generate_food_near_existing()

    def generate_random_food(self):
        num_random_foods = int(constants.MAX_FOOD_COUNT * constants.RANDOM_FOOD_PROPORTION)
        new_foods = []
        while len(new_foods) < num_random_foods:
            x = random.randint(constants.CENTER_AREA_X_START,constants.CENTER_AREA_X_START+constants.CENTER_AREA_WIDTH - constants.FOOD_SIZE)
            y = random.randint(constants.CENTER_AREA_Y_START, constants.CENTER_AREA_Y_START+constants.CENTER_AREA_HEIGHT - constants.FOOD_SIZE)
            new_food = Food(x, y, constants.FOOD_SIZE)
            if not any(new_food.rect.colliderect(obs.rect) for obs in self.obstacles) and \
            not any(new_food.rect.colliderect(f.rect) for f in self.foods):
                new_foods.append(new_food)
        self.foods.extend(new_foods)
        
    def generate_food_near_existing(self):
        directions = [(constants.FOOD_SPAWN_DISTANCE, 0), (-constants.FOOD_SPAWN_DISTANCE, 0),
                    (0, constants.FOOD_SPAWN_DISTANCE), (0, -constants.FOOD_SPAWN_DISTANCE)]
        
        if len(self.foods) >= constants.MAX_FOOD_COUNT:
            return

        new_foods = []
        for food in self.foods:
            for dx, dy in directions:
                x = food.rect.x + dx
                y = food.rect.y + dy
                if constants.CENTER_AREA_X_START <= x <= constants.CENTER_AREA_X_START + constants.CENTER_AREA_WIDTH - constants.FOOD_SIZE and \
                constants.CENTER_AREA_Y_START <= y <= constants.CENTER_AREA_Y_START + constants.CENTER_AREA_HEIGHT - constants.FOOD_SIZE:
                    new_food = Food(x, y, constants.FOOD_SIZE)
                    if not any(new_food.rect.colliderect(obs.rect) for obs in self.obstacles) and \
                    not any(new_food.rect.colliderect(f.rect) for f in self.foods):
                        new_foods.append(new_food)
                        if len(new_foods) + len(self.foods) >= constants.MAX_FOOD_COUNT:
                            break
            if len(new_foods) + len(self.foods) >= constants.MAX_FOOD_COUNT:
                break

        self.foods.extend(new_foods)

    def check_events(self):
        pass

    def remove_dead(self):
        self.predators = [p for p in self.predators if p.is_alive]
        self.preys = [p for p in self.preys if p.is_alive ]
        # Update the status of agents
        for agent in list(self.agent_status.keys()):
            if agent not in [p.name for p in self.predators + self.preys]:
                self.agent_status[agent] = False
    def is_agent_alive(self, name):
            return self.agent_status.get(name, False)  # Return False if agent is not found

    # def obs_envs(self):
    #     for predator in self.predators:
    #         predator.set_prey_list(self.preys)
    #         predator.env_predators = self.predators
    #         predator.env_prey = self.preys
    #         predator.env_food = self.foods
    #         predator.env_obstacles = self.obstacles
    #         # obs_env = self.observe_info_predator(predator)
    #         # move_vector = self.action_predator(obs_env)

    #         self.move_predator(predator)
    #         predator.increment_iteration()  # 增加迭代计数器

    #     for prey in self.preys:
    #         prey.env_predators = self.predators
    #         prey.env_prey = self.preys
    #         prey.env_food = self.foods
    #         prey.env_obstacles = self.obstacles
    #         self.move_prey(prey)
    #         prey.increment_iteration()  # 增加迭代计数器

    def check_collisions(self):
        # 捕食者和猎物之间的相遇检测
        for predator in self.predators:
            for prey in self.preys:
                if predator.rect.colliderect(prey.rect):
                    self.handle_predator_prey_collision(predator, prey)
        
        # 捕食者之间的相遇检测
        for i, predator1 in enumerate(self.predators):
            for predator2 in self.predators[i+1:]:
                if predator1.rect.colliderect(predator2.rect):
                    self.handle_predator_predator_collision(predator1, predator2)
        # 猎物之间的相遇检测
        for i, prey1 in enumerate(self.preys):
            for prey2 in self.preys[i+1:]:
                if prey1.rect.colliderect(prey2.rect):
                    self.handle_prey_prey_collision(prey1, prey2)
        
    def handle_predator_prey_collision(self, predator, prey):
        # 增加捕食者的健康值
        predator.health += prey.health * constants.PREDATOR_HEALTH_GAIN_FACTOR

        if predator.health > predator.max_health:
            predator.health = predator.max_health

        # 移除猎物
        prey.is_alive = False
        prey.health = 0
        # prey.prey_list.remove(prey)
    def handle_predator_predator_collision(self, predator1, predator2):
        # 检查是否使用相同的算法
        if predator1.algorithm == predator2.algorithm:
            self.breedPredator(predator1, predator2)

    # def breed_predator(self, predator1, predator2):
    #     # 生成新的捕食者
    #     child_x = (predator1.rect.x + predator2.rect.x) // 2
    #     child_y = (predator1.rect.y + predator2.rect.y) // 2
    #     name = f"Pred{predator1.algorithm}_{self.next_predator_id}"  # 使用全局唯一ID生成名称

    #     child = Predator(child_x, child_y, constants.BLOCK_SIZE, name = name,algorithm=predator1.algorithm)

    #         # 确保新生成的猎物不会与任何障碍物重叠
    #     if not any(child.rect.colliderect(obs.rect) for obs in self.obstacles):
    #         self.predators.append(child)
    #         self.next_predator_id += 1  # 成功生成猎物后增加ID


    def handle_prey_prey_collision(self, prey1, prey2):
        # 检查是否使用相同的算法
        if prey1.algorithm == prey2.algorithm:

            self.breedPrey(prey1, prey2)

    # def breed_prey(self, prey1, prey2):
    #     # 生成新的猎物
    #     child_x = (prey1.rect.x + prey2.rect.x) // 2
    #     child_y = (prey1.rect.y + prey2.rect.y) // 2
    #     name = f"Prey{prey1.algorithm}_{self.next_prey_id}"  # 使用全局唯一ID生成名称

    #     child = Prey(child_x, child_y, constants.BLOCK_SIZE,name=name, algorithm=prey1.algorithm)
    #     # 确保新生成的猎物不与其他障碍物或智能体发生碰撞
    #         # 确保新生成的猎物不会与任何障碍物重叠
    #     if not any(child.rect.colliderect(obs.rect) for obs in self.obstacles):
    #         self.preys.append(child)
    #         self.next_prey_id += 1  # 成功生成猎物后增加ID

    def move_models(self):
        for predator in self.predators:
            predator.set_prey_list(self.preys)
            predator.env_predators = self.predators
            predator.env_prey = self.preys
            predator.env_food = self.foods
            predator.env_obstacles = self.obstacles
            # self.move_predator(predator)

            predator_ob_env = predator.get_observe_info()
            predator_move_vector = predator.get_target(predator_ob_env)
            # print(move_vector)
            predator.move_strategy(predator_move_vector)
            predator.move(constants.CONTROL_PANEL_WIDTH, self.screen_width, self.screen_height, self.obstacles)

            predator.increment_iteration()  # 增加迭代计数器

        for prey in self.preys:
            prey.env_predators = self.predators
            prey.env_prey = self.preys
            prey.env_food = self.foods
            prey.env_obstacles = self.obstacles

            # self.move_prey(prey)
            prey_ob_env = prey.get_observe_info()
            prey_move_vector = prey.get_target(prey_ob_env)
            prey.move_strategy(prey_move_vector)
            prey.move(constants.CONTROL_PANEL_WIDTH, self.screen_width, self.screen_height, self.obstacles)
            
            prey.increment_iteration()  # 增加迭代计数器
    def obsreve_prey(self):
        pass
    def observe_info_predator(self):
        pass
    def move_prey(self, prey):
        prey_ob_env = prey.get_observe_info()
        move_vector = prey.get_target(prey_ob_env)
        prey.move_strategy(move_vector)
        prey.move(constants.CONTROL_PANEL_WIDTH, self.screen_width, self.screen_height, self.obstacles)

    def move_predator(self, predator):
        predator_ob_env = predator.get_observe_info()
        move_vector = predator.get_target(predator_ob_env)
        print(move_vector)
        predator.move_strategy(move_vector)
        predator.move(constants.CONTROL_PANEL_WIDTH, self.screen_width, self.screen_height, self.obstacles)

    def draw_models(self, screen):
        for obstacle in self.obstacles:
            obstacle.draw(screen)
        for predator in self.predators:
            predator.draw(screen)
        for prey_item in self.preys:
            prey_item.draw(screen)
        for food_item in self.foods:
            food_item.draw(screen)

        if self.selected_agent:
            agent_info = (
                f"{self.selected_agent.__class__.__name__}: "
                f"Position ({self.selected_agent.rect.x}, {self.selected_agent.rect.y}), "
                f"Velocity ({self.selected_agent.velocity[0]}, {self.selected_agent.velocity[1]}), "
                f"Health ({self.selected_agent.health})"
            )
            info_surface = pygame.font.Font(None, 24).render(agent_info, True, (255, 255, 255))
            screen.blit(info_surface, (50, self.screen_height - 100))

    def update_health(self):
        for predator in self.predators:
            predator.update_health()
        for prey_item in self.preys:
            prey_item.update_health()

    def prey_hunt(self):
        for prey_item in self.preys:
            prey_item.eat_food(self.foods)

    def predator_hunt(self):
        for predator in self.predators:
            predator.hunt_prey(self.preys)
            
    def remove_food(self, food):
        if food in self.foods:
            self.foods.remove(food)
                
    def decrease_health(self):
        self.update_health()
        self.remove_dead()

    def get_agent_info(self, pos):
        for agent in self.predators + self.preys:
            if agent.rect.collidepoint(pos):
                return agent
        return None

    def spawn_food(self, count):
        for _ in range(count):
            self.generate_food()

