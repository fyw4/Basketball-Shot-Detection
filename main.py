from shot_detector import Shot_Detector

detector = Shot_Detector(source="./WeChat_20260328100503.mp4", output_path="output", step=1, display_object_info=True, model="./bball_model.pt", verbose=False)
makes, attempts = detector.run()
print(f"Successful shots: {makes}/{attempts}")