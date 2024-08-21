# main.py
import pygame
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from env.simulator import Simulator
import env.constants as constants

pygame.init()

# 设置全屏窗口
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.display.set_caption("Control Panel and Game Space")

# 获取屏幕尺寸
screen_width, screen_height = screen.get_size()
#print(screen_width)
#print(screen_height)

# 字体设置
font = pygame.font.Font(None, constants.FONT_SIZE)

# 侧边栏宽度
sidebar_width = constants.CONTROL_PANEL_WIDTH

# 按钮配置
buttons = [
    pygame.Rect(50, 100, constants.BUTTON_WIDTH, constants.BUTTON_HEIGHT),
    pygame.Rect(50, 100 + (constants.BUTTON_HEIGHT + constants.BUTTON_MARGIN), constants.BUTTON_WIDTH, constants.BUTTON_HEIGHT),
    pygame.Rect(50, 100 + 2 * (constants.BUTTON_HEIGHT + constants.BUTTON_MARGIN), constants.BUTTON_WIDTH, constants.BUTTON_HEIGHT)
]

slow_button = pygame.Rect(50, 100 + 3 * (constants.BUTTON_HEIGHT + constants.BUTTON_MARGIN), constants.BUTTON_WIDTH, constants.BUTTON_HEIGHT)

# 定义游戏状态
game_state = constants.MAIN_MENU

# 初始化模拟器
sim = Simulator(screen_width, screen_height)

# 初始化帧率
fps = constants.DEFAULT_FPS
clock = pygame.time.Clock()

# 迭代计数器
iteration_count = 0

# 初始化数据记录结构
prey_counts = []
predator_counts = []
prey_born_count = 0
predator_born_count = 0

# 创建图表
fig, ax = plt.subplots(figsize=(3, 2))  # 设置图表尺寸适应侧边栏
canvas = FigureCanvas(fig)

def reset_algorithm():
    prey_algorithms = ["PPO","PPO","PPO","PPO","PPO","PPO","DDPG","DDPG","DDPG","DDPG","DDPG","DDPG","DDPG","DDPG","DDPG"]
    pred_algorithms = ["PPO","PPO","PPO","DDPG","DDPG","DDPG"]
    all_pred_algorithms = assign_algorithms_to_agents(constants.NUM_PREDATORS,pred_algorithms)
    all_prey_algorithms = assign_algorithms_to_agents(constants.NUM_PREY,prey_algorithms)
    return all_pred_algorithms,all_prey_algorithms
def assign_algorithms_to_agents(len_agents, algorithm_names):
    """
    分配算法给每个智能体。

    参数:
    - agents: 智能体列表。
    - algorithm_names: 预定义的算法名称列表。

    返回:
    - 包含算法名称的列表，长度与agents列表相同。如果算法名称不足，则用'random'补充。
    """
    assigned_algorithms = []
    for i in range(len_agents):
        if i < len(algorithm_names):
            assigned_algorithms.append(algorithm_names[i])
        else:
            assigned_algorithms.append('random')
    return assigned_algorithms
def update_plot(prey_counts, predator_counts):
    ax.clear()
    ax.plot(prey_counts, label="Prey", color='blue')
    ax.plot(predator_counts, label="Predator", color='red')
    ax.legend(loc='upper right')
    ax.set_title("Population Over Time")
    canvas.draw()

def blit_plot():
    raw_data = canvas.buffer_rgba()
    raw_data_bytes = raw_data.tobytes()
    size = canvas.get_width_height()
    plot_surface = pygame.image.fromstring(raw_data_bytes, size, "RGBA")
    return plot_surface
all_pred_algorithms,all_prey_algorithms = reset_algorithm()
# 游戏主循环
running = True
selected_agent = None

while iteration_count<10000000:
    delta_time = clock.get_time() / 1000.0  # 计算帧时间，单位为秒
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if game_state == constants.MAIN_MENU:
                for button in buttons:
                    if button.collidepoint(event.pos):
                        if buttons.index(button) == 0:
                            game_state = constants.IN_GAME
                            sim.initialize(all_pred_algorithms,all_prey_algorithms)
                            print("Game Started")
            elif game_state == constants.IN_GAME:
                if slow_button.collidepoint(event.pos):
                    fps = constants.SLOW_FPS if fps == constants.DEFAULT_FPS else constants.DEFAULT_FPS
                else:
                    selected_agent = sim.get_agent_info(event.pos)
                    if selected_agent is not None:
                        print(selected_agent.name)
                    else:
                        print("No agent selected.")
                    if selected_agent:
                        selected_agent.selected = True
                        for agent in sim.predators + sim.preys:
                            if agent != selected_agent:
                                agent.selected = False
                    else:
                        for agent in sim.predators + sim.preys:
                            agent.selected = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if game_state == constants.IN_GAME:
                    game_state = constants.MAIN_MENU
                    selected_agent = None
                    for agent in sim.predators + sim.preys:
                        agent.selected = False

    screen.fill((255,255, 255))

    # 绘制侧边栏背景
    pygame.draw.rect(screen, (50, 50, 50), (0, 0, sidebar_width, screen_height))

    if game_state == constants.MAIN_MENU:
        for button in buttons:
            color = constants.BUTTON_HOVER_COLOR if button.collidepoint(pygame.mouse.get_pos()) else constants.BUTTON_COLOR
            pygame.draw.rect(screen, color, button)
            text = font.render(constants.BUTTON_TEXTS[buttons.index(button)], True, (255, 255, 255))
            screen.blit(text, (button.x + (constants.BUTTON_WIDTH - text.get_width()) // 2, button.y + (constants.BUTTON_HEIGHT - text.get_height()) // 2))

    elif game_state == constants.IN_GAME:
        sim.check_events()

        
        sim.move_models()
        sim.add_food()  # 传递时间间隔
        sim.prey_hunt()
        sim.check_collisions()
        # sim.predator_hunt()
        # new_prey_born, new_predator_born = sim.applyGeneticAlgorithm()
        sim.decrease_health()  # 更新健康值
        sim.remove_dead()  # 清理死亡个体
        iteration_count += 1  # 增加迭代计数器
        sim.draw_models(screen)

        # 每100个回合输出日志
        # if iteration_count % 10 == 0:
        #     new_prey_born, new_predator_born = sim.applyGeneticAlgorithm()
            # prey_born_count += new_prey_born
            # predator_born_count += new_predator_born
            # print(f"Iteration {iteration_count}: Current Predators: {len(sim.predators)}, New Predators Born: {predator_born_count}, Predators Died: {sim.dead_predator_count}")
            # prey_born_count = 0
            # predator_born_count = 0
            # sim.dead_predator_count = 0  # 重置死亡捕食者计数
        # if iteration_count %100 ==0:
        #     print(sim.preys[1].name,end="    ")
        #     print(sim.preys[1].health,end="    ")
        # 更新数据记录结构
        prey_counts.append(len(sim.preys))
        predator_counts.append(len(sim.predators))

        # 更新并绘制图表
        update_plot(prey_counts, predator_counts)
        plot_surface = blit_plot()
        plot_rect = plot_surface.get_rect(center=(sidebar_width // 2, screen_height // 2))
        screen.blit(plot_surface, plot_rect.topleft)

        color = constants.BUTTON_HOVER_COLOR if slow_button.collidepoint(pygame.mouse.get_pos()) else constants.BUTTON_COLOR
        pygame.draw.rect(screen, color, slow_button)
        text = font.render("Slow Down", True, (255, 255, 255))
        screen.blit(text, (slow_button.x + (constants.BUTTON_WIDTH - text.get_width()) // 2, slow_button.y + (constants.BUTTON_HEIGHT - text.get_height()) // 2))

        # 显示迭代次数
        iteration_text = font.render(f"Iteration: {iteration_count}", True, (255, 255, 255))
        screen.blit(iteration_text, (50, screen_height - 150))

        if selected_agent:
            agent_info = (
                f"{selected_agent.__class__.__name__}: "
                f"Position ({selected_agent.rect.x}, {selected_agent.rect.y}), "
                f"Velocity ({selected_agent.velocity[0]}, {selected_agent.velocity[1]}), "
                f"Health ({selected_agent.health})"
            )
            info_surface = font.render(agent_info, True, (0, 0, 0))
            screen.blit(info_surface, (50, screen_height - 100))

    pygame.display.flip()
    clock.tick(fps)

pygame.quit()
