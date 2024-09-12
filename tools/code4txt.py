import os

# 定义文件夹路径
parent_folder = os.path.dirname(os.path.abspath(__file__))  # 当前父文件夹
two_parent_folder = os.path.dirname(parent_folder)
env_folder = os.path.join(two_parent_folder, 'env')  # env 文件夹路径

# 定义要生成的文本文件路径
output_file = os.path.join(parent_folder, '0905combined_python_files.txt')

# 函数用于读取文件内容并写入到目标文件中
def append_file_content(file_path, output):
    with open(file_path, 'r') as f:
        content = f.read()
    with open(output, 'a') as f:
        f.write(f'\n\n# File: {file_path}\n\n')
        f.write(content)

# 获取父文件夹中的所有Python文件，并将内容写入目标文件
for root, _, files in os.walk(parent_folder):
    if root == env_folder:  # 跳过env文件夹
        continue
    for file in files:
        if file.endswith('.py'):
            append_file_content(os.path.join(root, file), output_file)

# 获取env文件夹中的所有Python文件，并将内容写入目标文件
for root, _, files in os.walk(env_folder):
    for file in files:
        if file.endswith('.py'):
            append_file_content(os.path.join(root, file), output_file)

print(f"所有Python文件的内容已成功粘贴到 {output_file} 中。")
