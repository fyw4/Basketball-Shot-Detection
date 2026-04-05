from shot_detector import Shot_Detector

detector = Shot_Detector(source="./IMG_4674.mp4", output_path="output", step=2, display_object_info=False, model="./bball_model.pt", verbose=False, record=False)
makes, attempts = detector.run()
print(f"Successful shots: {makes}/{attempts}")