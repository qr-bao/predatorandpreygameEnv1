import os

def find_py_files_in_parent_directory():
    """
    查找当前脚本所在目录的父目录及其子目录中的所有 .py 文件。

    Returns:
        py_files (list): 发现的所有 .py 文件的完整路径列表。
    """
    # 获取当前脚本的父目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    py_files = []
    
    # 遍历父目录及其子目录
    for dirpath, _, filenames in os.walk(parent_dir):
        for filename in filenames:
            if filename.endswith('.py'):
                py_files.append(os.path.join(dirpath, filename))
    
    return py_files

def display_and_select_files(py_files):
    """
    显示所有找到的 .py 文件并让用户选择要整合的文件。

    Args:
        py_files (list): .py 文件路径列表。

    Returns:
        selected_files (list): 用户选择的 .py 文件路径列表。
    """
    print("找到以下 Python 文件：")
    for idx, file_path in enumerate(py_files):
        print(f"{idx}: {file_path}")
    
    selected_files = input("\n请输入要选择的文件编号（用逗号分隔，例如：0,1,3）：")
    selected_files_indices = [int(i.strip()) for i in selected_files.split(',') if i.strip().isdigit()]
    
    return [py_files[i] for i in selected_files_indices]

def collect_selected_py_files_to_txt(selected_files, output_file):
    """
    将用户选择的 .py 文件的内容整合到一个 .txt 文件中。

    Args:
        selected_files (list): 用户选择的 .py 文件路径列表。
        output_file (str): 输出的 .txt 文件路径。
    """
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for py_file_path in selected_files:
            with open(py_file_path, 'r', encoding='utf-8') as infile:
                outfile.write(f'# File: {py_file_path}\n')  # 写入文件名
                outfile.write(infile.read())  # 写入文件内容
                outfile.write('\n\n')  # 添加分隔符
    print(f"所选 .py 文件的内容已整合到 {output_file}")

def main():
    # 查找父目录及其子目录中的所有 .py 文件
    py_files = find_py_files_in_parent_directory()

    # 让用户选择要整合的 .py 文件
    selected_files = display_and_select_files(py_files)

    # 定义输出的 txt 文件路径
    output_txt_file = 'output_file.txt'

    # 将所选文件内容整合到 txt 文件中
    collect_selected_py_files_to_txt(selected_files, output_txt_file)

if __name__ == "__main__":
    main()
