import math


# 计算相对于agent的dx, dy

dx = 2 - 0
dy = 0 - 0

# 计算agent的移动方向角度
agent_movement_angle = math.atan2(1, 0)

# 使用旋转矩阵将坐标转换到agent的参考系
relative_x = dx * math.cos(-agent_movement_angle) - dy * math.sin(-agent_movement_angle)
relative_y = dx * math.sin(-agent_movement_angle) + dy * math.cos(-agent_movement_angle)
    

print(relative_x,relative_y)
print(math.atan2(1,0))
