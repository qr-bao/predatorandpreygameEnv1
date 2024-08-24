# constants.py

# 窗口设置
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 900

SCREEN_WIDTH1 = 3840
SCREEN_HEIGHT1 = 2160
# 控制栏和游戏空间宽度
CONTROL_PANEL_WIDTH = 400

# 按钮属性
BUTTON_COLOR = (0, 255, 0)
BUTTON_HOVER_COLOR = (255, 0, 0)
BUTTON_WIDTH = 150
BUTTON_HEIGHT = 50
BUTTON_MARGIN = 20

# 字体设置
FONT_SIZE = 36
BUTTON_TEXTS = ["Start Game", "Button 2", "Button 3", "Slow Down"]

# 新增常量，定义食物生成的中心75%范围
CENTER_AREA_WIDTH = int((SCREEN_WIDTH1 -CONTROL_PANEL_WIDTH)*0.75 )
CENTER_AREA_HEIGHT = int(SCREEN_HEIGHT1 * 0.75)
CENTER_AREA_X_START = int(CONTROL_PANEL_WIDTH+(SCREEN_WIDTH1-CONTROL_PANEL_WIDTH-CENTER_AREA_WIDTH)//2) 
CENTER_AREA_Y_START = int((SCREEN_HEIGHT1-CENTER_AREA_HEIGHT)//2)
# 游戏状态
MAIN_MENU = 0
IN_GAME = 1

# 小方块属性
NUM_PREDATORS =10  # 捕食者初始数量
NUM_PREY = 20 # 猎物初始数量
BLOCK_SIZE = 30

# 捕食者和猎物的生命值和生命值衰减
PREDATOR_INITIAL_HEALTH = 45  # 增加初始健康值
PREY_INITIAL_HEALTH = 20  # 增加初始健康值
PREDATOR_HEALTH_DECAY = 0.1  # 减少健康值衰减速度
PREY_HEALTH_DECAY = 0.2  # 减少猎物的健康值衰减速度

# 捕食者和猎物的生命值上限
PREDATOR_MAX_HEALTH = 100  # 增加健康值上限
PREY_MAX_HEALTH = 50  # 增加健康值上限

# 食物属性
NUM_FOOD = 100  # 增加初始食物数量
FOOD_SIZE = 20
FOOD_COLOR = (0, 0, 255)
FOOD_HEALTH_GAIN = 4  # 增加食物提供的健康值
# FOOD_GENERATION_INTERVAL = 80  # 每5秒生成一次食物
# 新增的常量
RANDOM_FOOD_PROPORTION = 0.1  # 随机食物的比例
FOOD_GENERATION_INTERVAL = 45  # 食物生成的迭代间隔
FOOD_SPAWN_DISTANCE = 55  # 食物生成时的偏移距离
MAX_FOOD_COUNT = 400  # 食物生成的最大数量
# 捕食相关
PREDATOR_HEALTH_GAIN_FACTOR = 0.8  # 调整捕食获得的健康值

# 游戏速度
DEFAULT_FPS = 30
SLOW_FPS = 10

# 遗传算法相关
MUTATION_CHANCE = 0.01  # 增加突变几率
PREY_REPRODUCTION_PROBABILITY = 0.2
PREDATOR_REPRODUCTION_PROBABILITY = 0.05
PREDATOR_SPEED = 5

# 速度和加速度
PREY_MAX_SPEED = 10 
PREY_MAX_ACCELERATION = 2.5
PREY_MAX_TURNING_ANGLE = 0.5

PREDATOR_MAX_SPEED = 10 
PREDATOR_MAX_ACCELERATION = 4
PREDATOR_MAX_TURNING_ANGLE = 0.8

# 听觉范围
PREDATOR_HEARING_RANGE = 800
PREY_HEARING_RANGE = 800
max_observation_count = 5
max_hearing_count = 5

# 健康值减少相关常量
PREDATOR_ACCELERATION_HEALTH_DECAY_FACTOR = 0.05# 减少加速导致的健康值衰减
PREY_ACCELERATION_HEALTH_DECAY_FACTOR = 0.10  # 减少加速导致的健康值衰减

# 新增的常量
PREDATOR_MIN_DISTANCE = 10  # 捕食者最小接近距离
PREDATOR_ROTATION_CHANCE = 0.05  # 减少停下来旋转的几率
PREDATOR_ROTATION_SPEED = 1  # 旋转的速度
PREDATOR_ACCELERATION_FACTOR = 9.2  # 调整捕食者加速因子

PREY_EVASION_FACTOR = 2.5  # 增强远离捕食者的因子
PREY_RANDOM_MOVE_CHANCE = 0.05  # 减少随机移动的几率
PREY_RANDOM_MOVE_SPEED = 1.5  # 随机移动的速度
PREY_TURN_INTERVAL = 50  # 增加定期回头观察的间隔（帧数）

# 视觉范围
PREDATOR_SIGHT_RANGE = 600  # 调整捕食者的视觉范围
PREY_SIGHT_RANGE = 600  # 调整猎物的视觉范围

# 最大迭代次数
MAX_ITERATIONS = 10000  # 设置为一个合理的值以控制运行时间

# 繁殖所需的最低生命值
PREDATOR_MIN_HEALTH_FOR_REPRODUCTION = PREDATOR_INITIAL_HEALTH * 0.1
PREY_MIN_HEALTH_FOR_REPRODUCTION = PREY_INITIAL_HEALTH * 0.2

# 繁殖迭代计数器阈值
PREY_REPRODUCTION_ITERATION_THRESHOLD = 50
PREDATOR_REPRODUCTION_ITERATION_THRESHOLD = 50
