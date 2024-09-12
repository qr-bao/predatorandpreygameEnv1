# constants.py

# # window setting
# SCREEN_WIDTH = 1600
# SCREEN_HEIGHT = 900

SCREEN_WIDTH1 = 3840
SCREEN_HEIGHT1 = 2160
# Control bar and game space width
CONTROL_PANEL_WIDTH = 400

# Button properties
BUTTON_COLOR = (0, 255, 0)
BUTTON_HOVER_COLOR = (255, 0, 0)
BUTTON_WIDTH = 150
BUTTON_HEIGHT = 50
BUTTON_MARGIN = 20

# Font settings
FONT_SIZE = 36
BUTTON_TEXTS = ["Start Game", "Button 2", "Button 3", "Slow Down"]

# Added a new constant to define the central 75% range of food generation
CENTER_AREA_WIDTH = int((SCREEN_WIDTH1 -CONTROL_PANEL_WIDTH)*0.75 )
CENTER_AREA_HEIGHT = int(SCREEN_HEIGHT1 * 0.75)
CENTER_AREA_X_START = int(CONTROL_PANEL_WIDTH+(SCREEN_WIDTH1-CONTROL_PANEL_WIDTH-CENTER_AREA_WIDTH)//2) 
CENTER_AREA_Y_START = int((SCREEN_HEIGHT1-CENTER_AREA_HEIGHT)//2)
# game state
MAIN_MENU = 0
IN_GAME = 1

# number of agent
NUM_PREDATORS =30  # 捕食者初始数量
NUM_PREY = 20 # 猎物初始数量
BLOCK_SIZE = 30

# Predator and prey health and health decay
PREDATOR_INITIAL_HEALTH = 45  # Increase initial health
PREY_INITIAL_HEALTH = 20  # Increase initial health
PREDATOR_HEALTH_DECAY = 0.05  # Reduce health decay rate
PREY_HEALTH_DECAY = 0.1  # Reduces the speed at which prey’s health decays
HEALTH_RENEW = 0.33
# Predator and prey health caps
PREDATOR_MAX_HEALTH = float('inf')  # Increase health limit
PREY_MAX_HEALTH = float('inf')  # Increase health limit

# 食物属性
NUM_FOOD = 100  # Increase initial food quantity
FOOD_SIZE = 20
FOOD_COLOR = (0, 0, 255)
FOOD_HEALTH_GAIN = 4  #Increase the health value provided by food
#FOOD_GENERATION_INTERVAL = 80 # Generate food every 5 seconds
#New constants
RANDOM_FOOD_PROPORTION = 0.1  # proportion of random food
FOOD_GENERATION_INTERVAL = 450  # Iteration interval for food generation
FOOD_SPAWN_DISTANCE = 55  #The offset distance when food is generated
MAX_FOOD_COUNT = float('inf') # The maximum amount of food generated
# Predation related
PREDATOR_HEALTH_GAIN_FACTOR = 0.66  #Adjust health gained from preying

# Adjust health gained from preying
DEFAULT_FPS = 30
SLOW_FPS = 10

# Genetic algorithm related
MUTATION_CHANCE = 0.01  # 增加突变几率
PREY_REPRODUCTION_PROBABILITY = 0.2
PREDATOR_REPRODUCTION_PROBABILITY = 0.2
PREDATOR_SPEED = 5

# speed and acceleration
PREY_MAX_SPEED = 10 
PREY_MAX_ACCELERATION = 2.5
PREY_MAX_TURNING_ANGLE = 0.5

PREDATOR_MAX_SPEED = 10 
PREDATOR_MAX_ACCELERATION = 4
PREDATOR_MAX_TURNING_ANGLE = 0.8

# hearing range
PREDATOR_HEARING_RANGE = 800
PREY_HEARING_RANGE = 800
max_observation_count = 5
max_hearing_count = 5

# Health value reduction related constants
PREDATOR_ACCELERATION_HEALTH_DECAY_FACTOR = 0.02# Reduce health decay caused by acceleration
PREY_ACCELERATION_HEALTH_DECAY_FACTOR = 0.10  # Reduce health decay caused by acceleration

#  New constants
PREDATOR_MIN_DISTANCE = 10  #Predator minimum approach distance
PREDATOR_ROTATION_CHANCE = 0.05  # Reduce the chance of stopping and spinning
PREDATOR_ROTATION_SPEED = 1  # rotation speed
PREDATOR_ACCELERATION_FACTOR = 9.2  # Adjusted Predator acceleration factor

PREY_EVASION_FACTOR = 2.5  # Increased predator protection factor
PREY_RANDOM_MOVE_CHANCE = 0.05  # Reduce the chance of random movement
PREY_RANDOM_MOVE_SPEED = 1.5  # random movement speed
PREY_TURN_INTERVAL = 50  #Increase the interval (number of frames) of regular look back and observe

#visual range
PREDATOR_SIGHT_RANGE = 600  #Adjust Predator's visual range
PREY_SIGHT_RANGE = 600  #Adjust the visual range of prey

#Maximum number of iterations
MAX_ITERATIONS = 10000  #Set to a reasonable value to control runtime

#Minimum health required to reproduce
PREDATOR_MIN_HEALTH_FOR_REPRODUCTION = PREDATOR_INITIAL_HEALTH * 0.5
PREY_MIN_HEALTH_FOR_REPRODUCTION = PREY_INITIAL_HEALTH * 0.5

#Breed iteration counter threshold
PREY_REPRODUCTION_ITERATION_THRESHOLD = 50
PREDATOR_REPRODUCTION_ITERATION_THRESHOLD = 50
