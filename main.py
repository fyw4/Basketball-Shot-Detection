from shot_detector import Shot_Detector
import time
import os

current_dir = "E:\新建文件夹"
filelist = os.listdir(current_dir)

for file in filelist:
    if file.endswith(".mov"):
        detector = Shot_Detector(source=os.path.join(current_dir, file), output_path="goal_clips", step=2, display_object_info=True, model="./bball_model.pt", verbose=False, record=True, device="cuda")
        makes, attempts = detector.run()
        print(f"step is {detector.step}")
        print(f"Successful shots: {makes}/{attempts}")
        print("-----------------")

time.sleep(10)