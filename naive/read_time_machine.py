import os

def read_literals(file_path):
    # add exception process
    with open(file_path, 'r') as f:
        return [line for line in f]

if __name__ == '__main__':
    data_dir = '/home/zhaoyu/Dataset/TimeMachine'
    extension = '.txt'
    files_path = [data_dir+'/'+f for f in os.listdir(data_dir) if extension in f]
    print(files_path)
    lines = read_literals(files_path[0])
    print(lines[0])
