# python basicsr/data/generate_hq.py --input_dir data/LOLv1/Test/input --ensemble_models retinexformer --output_dir ensemble_output/LOLv1
import os
import subprocess
import argparse
# ensemble_models = ['retinexformer','difflle']
parser = argparse.ArgumentParser()
import time
from basicsr.utils import imfrombytes, img2tensor
import cv2
import numpy as np
import torch
import concurrent.futures

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def readline(proc):
    line = proc.stdout.readline().decode()
    assert line[-1]=='\n'
    return line.strip()


def sendline(proc, msg): # msg==path
    assert type(msg)==str
    assert msg.count('\n')==0
    msg = msg+'\n'
    msg = msg.encode()
    proc.stdin.write(msg)
    proc.stdin.flush()
    return readline(proc)


def path_to_tensor(filepath):
    # print(filepath, '-----------')
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = np.float32(img) / 255.
    # img = torch.from_numpy(img).permute(2, 0, 1)
    img = torch.from_numpy(img).to(device).permute(2, 0, 1)
    # img /= 255.
    # print(img.shape, img.device)
    return img


global_proc_dict = {} # 每个模块只被加载一次


class Processes_Controller():
    def __init__(self, ensemble_models): # 给每个子模型创建进程
        
        self.models = ensemble_models
        self.processes = []
        
        for mod_name in ensemble_models:
            print(f"--- init model {mod_name} ---") 
            cmd = "python ensemble_work.py" # 进程要执行的脚本
            if mod_name in global_proc_dict:
                process = global_proc_dict[mod_name]
            else:
                process = subprocess.Popen(args=cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=f'ensemble_models/{mod_name}', shell=True)  # shell=True时参数应为字符串
                assert readline(process)=='init_ok'
                global_proc_dict[mod_name] = process
            self.processes.append(process)
            print(f"--- successfully init model {mod_name} ---")
    
    def get_hqs(self, path):
        # start_time = time.time()
        def process_task(proc):
            output_path = sendline(proc, path)
            # output_path = readline(proc)
            return path_to_tensor(output_path).unsqueeze(0)
        with concurrent.futures.ThreadPoolExecutor() as executor: # 多线程
            res = list(executor.map(process_task, self.processes))
        if len(res)==0:
            raise ValueError('no model selected!')
        res = torch.cat(res, dim=0)
        # print(f"submodel_time: {time.time() - start_time}")
        return res

    # def get_hqs(self, path):
    #     res = []
    #     for i,proc in enumerate(self.processes):
    #         start_time = time.time()
    #         # if moe_res[i].item()==0:
    #         #     continue
    #         # print(self.models[i])
    #         output_path = sendline(proc, path)
    #         # print(i, output_path, '--------')
    #         res.append(path_to_tensor(output_path).unsqueeze(0))
    #         print(f"{self.models[i]} time = {time.time() - start_time}")              
    #     if len(res)==0:
    #         raise ValueError('no model selected!')
    #     res = torch.cat(res, dim=0)
    #     return res

    def terminte_processes(self):
        # 终止所有当前进程
        for process in self.processes:
            process.terminate()  # 或使用 process.kill() 强制终止
            process.wait()  # 等待进程完全终止

        # 清空进程列表和全局进程字典
        self.processes = []
        global_proc_dict.clear()

    def close_env(self):
        pass


# class copy_Processes_Controller():
#     def __init__(self, ensemble_models): # 给每个子模型创建进程
        
#         self.models = ensemble_models
#         self.processes = []
        
#         for mod_name in ensemble_models:
#             print(f"--- init model {mod_name} ---") 
#             cmd = "python ensemble_work.py" # 进程要执行的脚本
#             if mod_name in global_proc_dict:
#                 process = global_proc_dict[mod_name]
#             else:
#                 process = subprocess.Popen(args=cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=f'ensemble_models/{mod_name}', shell=True)  # shell=True时参数应为字符串
#                 assert readline(process)=='init_ok'
#                 global_proc_dict[mod_name] = process
#             self.processes.append(process)
#             print(f"--- successfully init model {mod_name} ---")
    
#     def get_hqs(self, path):
#         res = []
#         def process_task(proc):
#             output_path = sendline(proc, path)
#             return path_to_tensor(output_path).unsqueeze(0)
#         # 创建线程池，并行执行
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             res = list(executor.map(process_task, self.processes))
#         if len(res)==0:
#             raise ValueError('no model selected!')
#         res = torch.cat(res, dim=0)
#         return res


# for mod in ensemble_models:
#     print(f"--- begin generating output_img from model {mod} ---")
#     # process = subprocess.Popen(["python", "ensemble_work.py", "--input_dir", input_dir, "--output_dir", output_dir],  # shell=False时,作为列表
#     cmd = ' '.join(["python", "ensemble_work.py", "--input_dir", input_dir, "--output_dir", output_dir]) 
#     process = subprocess.Popen(args=cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=f'ensemble_models/{mod}', shell=True)  # shell=True时参数应为字符串
#     # 使子进程能接收communicate input,需要指定 stdin=subprocess.PIPE
#     for i in range(5):
#         s = f'{i}+{i}\n'.encode()
#         process.stdin.write(s)
#         process.stdin.flush()
#         line = process.stdout.readline().decode()[:-1]
#         print(line)
#     # 如何terminate
#     exit(1)
#     ret_code = 0
#     print("std out:\n", stdout.decode(), end='')
#     if ret_code == 0:
#         print(f"--- successfully generate output_img from model {mod} ---\n")
#     else:
#         print(f"process {mod} failed with code {ret_code} !")
#         print("std error: ", stderr.decode())
#         exit(1)
# # for mod in ensemble_models:
    
