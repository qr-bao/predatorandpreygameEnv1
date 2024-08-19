import os

def collect_py_files_to_txt(parent_dir, output_file):
    with open(output_file, 'w') as outfile:
        for root, dirs, files in os.walk(parent_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    # print(file_path)
                    with open(file_path, 'r') as infile:
                        outfile.write(f'# File: {file_path}\n')
                        outfile.write(infile.read())
                        outfile.write('\n\n')

if __name__ == "__main__":
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))  # 获取上一层文件夹路径
    output_file = os.path.join(os.getcwd(), 'all_python_files3.txt')  # 输出文件路径

    collect_py_files_to_txt(parent_dir, output_file)
    print(f'所有的 .py 文件已整合到 {output_file}')

