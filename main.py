#from shot_detector import Shot_Detector
import time
import os
import tkinter as tk
from tkinter import ttk

import threading
from tkinter import filedialog

version = "v1.0"
current_source_dir = None
output_path_dir = None
endswith = None
clip_flag = 0

root = tk.Tk() # 创建主窗口对象
root.title(f"投篮进球剪辑工具{version}")
root.geometry("400x360")  # 设置窗口大小（宽x高）
root.resizable (0, 0) 
root.configure(bg="#f0f0f0")  # 设置背景颜色

# 打开文件对话框
def open_source_file_dialog():
    global current_source_dir
    current_source_dir = filedialog.askdirectory()
    open_dir_var.set(f"{current_source_dir}")

def open_dest_file_dialog():
    global output_path_dir
    output_path_dir = filedialog.askdirectory()
    dest_dir_var.set(f"{output_path_dir}")

def clip_begain():
    global clip_flag
    if 0 == clip_flag:
        clip_flag = 1
        threading.Thread(target = detect_shots, daemon = True).start() 

#创建打开文件目录按钮
file_open_button = tk.Button(root, text="打开文件目录", command = open_source_file_dialog)
file_open_button.pack(side = tk.TOP, anchor = tk.W, padx = 10, pady = 10)

open_dir_fix = tk.StringVar(value="打开文件目录位置:")
tk.Label(
    root,
    textvariable=open_dir_fix,
    font=("Consolas", 11),
    width=50,
    anchor="w"
).pack(side = tk.TOP, anchor = tk.W, padx = 10, pady = 5)

open_dir_var = tk.StringVar(value="--")
tk.Label(
    root,
    textvariable=open_dir_var,
    font=("Consolas", 11),
    width=50,
    anchor="w"
).pack(side = tk.TOP, anchor = tk.W, padx = 10, pady = 5)

#创建保存按钮
file_dest_button = tk.Button(root, text="文件保存位置", command = open_dest_file_dialog)
file_dest_button.pack(side = tk.TOP, anchor = tk.W, padx = 10, pady = 10)

dest_dir_fix = tk.StringVar(value="文件保存位置:")
tk.Label(
    root,
    textvariable=dest_dir_fix,
    font=("Consolas", 11),
    width=20,
    anchor="w"
).pack(side = tk.TOP, anchor = tk.W, padx = 10, pady = 5)

dest_dir_var = tk.StringVar(value="--")
tk.Label(
    root,
    textvariable=dest_dir_var,
    font=("Consolas", 11),
    width=50,
    anchor="w"
).pack(side = tk.TOP, anchor = tk.W, padx = 10, pady = 5)

frommat_fix = tk.StringVar(value="选择视频格式:")
tk.Label(
    root,
    textvariable=frommat_fix,
    font=("Consolas", 11),
    width=20,
    anchor="w"
).pack(side = tk.TOP, anchor = tk.W, padx = 10, pady = 5)

video_formats = (".mp4", ".avi", ".mov", ".mkv")
x_var = tk.StringVar(value=video_formats[0])
endswith = video_formats[0]
com = ttk.Combobox(
    root,
    textvariable=x_var,
    values=video_formats,
    state="readonly",
    font=("Consolas", 11),
    width=10
)
com.pack(side = tk.TOP, anchor = tk.W, padx = 10, pady = 5)
def on_format_change(event=None):
    global endswith
    endswith = x_var.get()

com.bind("<<ComboboxSelected>>", on_format_change)


#开始剪辑视频
clip_button = tk.Button(root, text="开始进球剪辑", command = clip_begain)
clip_button.pack(side = tk.TOP, anchor = tk.W, padx = 10, pady = 10)

def detect_shots():
    global current_source_dir
    global output_path_dir
    global clip_flag
    global endswith

    try:
        if  current_source_dir and output_path_dir:
            filelist = os.listdir(current_source_dir)

            for file in filelist:
                if file.endswith(endswith):
                    #detector = Shot_Detector(source=os.path.join(current_dir, file), output_path=output_path_dir, step=2, display_object_info=True, model="./bball_model.pt", verbose=False, record=True, device="cuda")
                    #makes, attempts = detector.run()
                    #print(f"step is {detector.step}")
                    #print(f"Successful shots: {makes}/{attempts}")
                    print("-----------------")

        time.sleep(10)
        clip_flag = 0
    finally:
        clip_flag = 0
     
root.mainloop()