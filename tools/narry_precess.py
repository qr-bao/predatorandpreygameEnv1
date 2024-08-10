# import numpy as np

# def process_matrix(matrix, target_shape=(5, 3)):
#     """
#     裁剪或填充矩阵，使其保持 target_shape 的大小。

#     参数:
#     - matrix: 输入的矩阵，形状为 (N, 3)
#     - target_shape: 目标形状，默认为 (5, 3)

#     返回:
#     - 处理后的矩阵，形状为 target_shape
#     """
#     current_shape = matrix.shape

#     # 确保输入矩阵的列数正确
#     assert current_shape[1] == target_shape[1], f"输入矩阵的列数应为 {target_shape[1]}，但得到了 {current_shape[1]}"

#     # 如果当前矩阵行数大于目标行数，则进行裁剪
#     if current_shape[0] > target_shape[0]:
#         processed_matrix = matrix[:target_shape[0], :]
#     # 如果当前矩阵行数小于目标行数，则进行填充
#     elif current_shape[0] < target_shape[0]:
#         padding_rows = target_shape[0] - current_shape[0]
#         padding = np.zeros((padding_rows, target_shape[1]))
#         processed_matrix = np.vstack((matrix, padding))
#     else:
#         processed_matrix = matrix

#     return processed_matrix

# # 示例使用
# input_matrix = np.array([
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9],
#     [10, 11, 12],
#     [13, 14, 15],
#     [16, 17, 18]
# ])

# processed_matrix = process_matrix(input_matrix)
# print(processed_matrix)


def process_matrix(matrix, target_shape=(5, 3)):
    """
    裁剪或填充矩阵，使其保持 target_shape 的大小。

    参数:
    - matrix: 输入的矩阵，形状为 (N, 3)
    - target_shape: 目标形状，默认为 (5, 3)

    返回:
    - 处理后的矩阵，形状为 target_shape
    """
    if not matrix:
        # 如果矩阵为空，直接填充目标大小的零矩阵
        return [[0] * target_shape[1] for _ in range(target_shape[0])]
    current_rows = len(matrix)
    current_cols = len(matrix[0]) if current_rows > 0 else 0

    # 确保输入矩阵的列数正确
    assert current_cols == target_shape[1], f"输入矩阵的列数应为 {target_shape[1]}，但得到了 {current_cols}"

    # 如果当前矩阵行数大于目标行数，则进行裁剪
    if current_rows > target_shape[0]:
        processed_matrix = matrix[:target_shape[0]]
    # 如果当前矩阵行数小于目标行数，则进行填充
    elif current_rows < target_shape[0]:
        processed_matrix = matrix[:]
        padding_rows = target_shape[0] - current_rows
        padding = [[0] * target_shape[1]] * padding_rows
        processed_matrix.extend(padding)
    else:
        processed_matrix = matrix

    return processed_matrix

matrix = []
print(process_matrix(matrix))
