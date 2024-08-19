
import os

def combine_python_files_to_txt():
    # 获取当前脚本文件夹的上一级文件夹路径
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    
    # 输出文件路径
    output_file = os.path.join(os.path.dirname(__file__), "combined_python_files.txt")
    
    # 打开输出文件（如果文件已存在，则覆盖；如果不存在，则新建）
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # 遍历上一级文件夹中的所有文件
        for filename in os.listdir(parent_dir):
            # 构建文件路径
            file_path = os.path.join(parent_dir, filename)
            # 检查文件是否为 .py 文件并且是否是文件而不是目录
            if filename.endswith('.py') and os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as infile:
                    # 写入文件名作为注释
                    outfile.write(f"# {filename}\n")
                    # 读取 .py 文件内容并写入到输出文件中
                    outfile.write(infile.read())
                    # 每个文件内容之间添加两个换行符
                    outfile.write("\n\n")
    
    print(f"All .py files from {parent_dir} have been combined into {output_file}")

if __name__ == "__main__":
    combine_python_files_to_txt()
