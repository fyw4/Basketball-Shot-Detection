from shot_detector import Shot_Detector
import time

detector = Shot_Detector(source="./IMG_4674.mp4", output_path="goal_clips", step=2, display_object_info=True, model="./bball_model.pt", verbose=False, record=True, device="cuda")
makes, attempts = detector.run()
print(f"step is {detector.step}")
print(f"Successful shots: {makes}/{attempts}")

time.sleep(10)