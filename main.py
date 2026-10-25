from shot_detector import Shot_Detector
import time
import os
import tkinter as tk

import threading
from tkinter import filedialog

current_dir = None
output_path_dir = None
endswith = ".MOV"

root = tk.Tk() # 创建主窗口对象
root.title("投篮进球剪辑工具")
root.geometry("400x300")  # 设置窗口大小（宽x高）
root.configure(bg="#f0f0f0")  # 设置背景颜色


# 打开文件对话框
def open_file_dialog():
    global current_dir
    current_dir = filedialog.askdirectory()
    print('\n获取的文件目录地址:', current_dir)

#创建打开文件目录按钮
file_open_button = tk.Button(root, text="打开文件目录", command = open_file_dialog)
file_open_button.pack(side = tk.TOP, anchor = tk.W, padx = 10, pady = 10)

open_dir_var = tk.StringVar(value="打开文件目录位置:--")
tk.Label(
    root,
    textvariable=open_dir_var,
    font=("Consolas", 11),
    width=20,
    anchor="w"
).pack(side = tk.TOP, anchor = tk.W, padx = 10, pady = 10)

#创建保存按钮
file_dest_button = tk.Button(root, text="文件保存位置", command = open_file_dialog)
file_dest_button.pack(side = tk.TOP, anchor = tk.W, padx = 10, pady = 10)

dest_dir_var = tk.StringVar(value="文件保存位置:--")
tk.Label(
    root,
    textvariable=dest_dir_var,
    font=("Consolas", 11),
    width=20,
    anchor="w"
).pack(side = tk.TOP, anchor = tk.W, padx = 10, pady = 10)

def detect_shots():
    global current_dir
    global output_path_dir

    if current_dir and output_path_dir is not None:
        filelist = os.listdir(current_dir)

        for file in filelist:
            if file.endswith(endswith):
                detector = Shot_Detector(source=os.path.join(current_dir, file), output_path=output_path_dir, step=2, display_object_info=True, model="./bball_model.pt", verbose=False, record=True, device="cuda")
                makes, attempts = detector.run()
                print(f"step is {detector.step}")
                print(f"Successful shots: {makes}/{attempts}")
                print("-----------------")

    time.sleep(10)

#============thread====================
if __name__ == "__main__":
     threading.Thread(target = detect_shots, daemon = True).start() 
     
root.mainloop()