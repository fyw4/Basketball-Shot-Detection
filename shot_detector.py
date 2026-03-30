from ultralytics import YOLO
import numpy as np
import cv2
from copy import deepcopy
from DetectedObject import DetectedObject
from DetectedBall import DetectedBall

class Shot_Detector:
    '''
        PARAMETERS:

            source - video source

            output_path - path to put the resulting video

            step - detect every step frames

            display_object_info - bool, used to display a detected objects class, confidence, and index in its list of positions (balls or hoops)

            model - path of a yolo object detection model

            verbose - YOLO verbose parameter
    '''

    def __init__(self, source: str, output_path: str | None = None, step: int = 1, display_object_info: bool = True, model: str = './bball_model.pt', verbose: bool = False) -> None:

        # SET PARAMETERS
        self.verbose = verbose
        self.model = YOLO(model, verbose=self.verbose)
        self.source = cv2.VideoCapture(source)
        self.output_path = output_path
        self.display_object_info = display_object_info
        self.step = step

        # HOOP & BALL POSITIONS
        self.hoops = {}
        self.hoopUid = 0

        self.balls = {}
        self.ballUid = 0

        self.frame_count = 0

        # BALLS IN THE AREA OF A HOOP
        self.up_ball = []
        self.down_ball = []

        self.attempts = 0
        self.makes = 0

    def run(self) -> tuple[int, int]:
        '''
            PERFORM SHOT DETECTION

            returns: makes, attempts
        '''

        # VIDEO PROPERTIES
        fps = int(self.source.get(cv2.CAP_PROP_FPS))
        frame_width = int(self.source.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.source.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if self.output_path:
            out = cv2.VideoWriter(f'{self.output_path}.mp4', fourcc, fps, (frame_width, frame_height))

        while True:
            ret, frame = self.source.read()
            if not ret:
                break
            self.frame_count += 1

            # PROCESS EVERY STEP FRAMES
            if self.frame_count % self.step == 0: #每step帧处理一次，比如step=2则每两帧处理一次

                self.clean_detections() # 清除过时的检测结果：移除超过20帧未检测到的球和篮筐；限制球的检测记录不超过30条

                # DETECT OBJECTS
                '''
                #result包含检测所有对象信息，包括：
                边界框坐标
                类别标签
                置信度分数
                其他检测元数据

                '''
                results = self.model.predict(frame, conf=0.2, stream=True, verbose=self.verbose) #置信程度设置为0.2
                class_names = self.model.names #获取模型的类别名称列表，例如["ball", "hoop"]
                for r in results:
                    for box in r.boxes:

                        # DETECTED OBJECT INFO
                        x1, y1, x2, y2 = box.xyxy[0] #获取检测框的左上角和右下角坐标，box.xyxy[0]返回一个包含四个元素的列表，分别是x1, y1, x2, y2
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        w, h = x2 - x1, y2 - y1
                        center = (int(x1 + w / 2), int(y1 + h / 2))

                        cls = int(box.cls[0].tolist()) #获取检测框的类别标签，box.cls[0]返回一个包含一个元素的列表，元素是类别标签的索引，tolist()将其转换为Python的整数类型
                        cls_name = class_names[cls] #根据类别标签获取类别名称:例如，cls=0可能对应“ball”，cls=1可能对应“hoop”
                        conf = int(box.conf[0].tolist()*100) / 100 #获取检测框的置信度分数，box.conf[0]返回一个包含一个元素的列表，元素是置信度分数，tolist()将其转换为Python的浮点数类型

                        # STORE DETECTED OBJECT IN CORRECT CLASS
                        object = DetectedObject(center[0], center[1], w, h, self.frame_count, conf)
                        if cls == 1:
                            index = self.add_hoop(object)
                        else:
                            index = self.add_ball(object)

                        if index == None:
                            continue

                        # PERFORM SHOT DETECTION
                        self.detect_up()
                        self.detect_down()
                        self.update_score()

                        # DRAW OBJECT INFO
                        if self.display_object_info:
                            font = cv2.FONT_HERSHEY_SIMPLEX #定义字体样式，SIMPLEX 表示"简单"或"标准"的字体样式
                            text = f"INDEX: {index}, CLASS: {cls_name}, CONF: {conf}"
                            text_size, _ = cv2.getTextSize(text, font, 0.5, 1)
                            text_x = x1 + (x2 - x1) // 2 - text_size[0] // 2 #文本水平居中于检测框，x1是检测框左上角的x坐标，x2是检测框右下角的x坐标，(x2 - x1) // 2计算检测框的宽度的一半，text_size[0]是文本的宽度的一半
                            text_y = y1 - 5

                            background_x1 = text_x - 5
                            background_y1 = text_y - text_size[1] - 5 # text_size[1] 是文本的高度
                            background_x2 = text_x + text_size[0] + 5 # text_size[0] 是文本的宽度
                            background_y2 = text_y + 5

                            # 绘制文本背景矩形，使用黑色填充，并在其上绘制文本信息，使用绿色字体显示对象的索引、类别和置信度分数
                            cv2.rectangle(frame, (background_x1, background_y1), (background_x2, background_y2), (0, 0, 0), -1)
                            cv2.putText(frame, text, (text_x, text_y), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                        # DRAW BOX AROUND OBJECT
                        # 绘制检测到的对象的边界框，使用绿色矩形框表示，边界框的坐标由x1, y1, x2, y2定义
                        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

                        # DISPLAY STATS
                        # 显示当前检测到的球和篮筐的数量，以及成功次数和总次数的百分比
                        percent = 0 if self.attempts == 0 else self.makes / self.attempts * 100
                        cv2.putText(frame, f'{self.makes}/{self.attempts} {percent:.2f}%', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 8)
                        cv2.putText(frame, f'{self.makes}/{self.attempts} {percent:.2f}%', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)

                    if self.output_path:
                        out.write(frame)

        self.source.release()
        if self.output_path:
            out.release()

        return self.makes, self.attempts

    def add_hoop(self, hoop: DetectedObject) -> int:
        '''
            Adds a detected hoop to the hoops dictionary. If the hoop is already detected, updates the detected hoop.

            PARAMETERS:
                hoop - DetectedObject object representing the hoop detected in the current frame

            RETURNS:
                key of detected hoop in hoops dictionary. Returns None if not added.
        '''

        if hoop.conf < 0.3:
            return None

        # NO HOOPS TO CHECK => ADD TO HOOPS
        if len(self.hoops) == 0:
            self.hoops[self.hoopUid] = hoop
            self.hoopUid += 1
            return self.hoopUid - 1

        # DETERMINE IF HOOP ALREADY DETECTED
        x, y = hoop.x, hoop.y
        for hoopKey, detectedHoop in self.hoops.items():
            # CALCULATE DISTANCE B/W THE HOOPS
            x_, y_ = detectedHoop.x, detectedHoop.y
            w_, h_ = detectedHoop.w, detectedHoop.h
            distance = np.sqrt( ((x_ - x)**2) + ((y_- y)**2) )
            hypotenuse =  np.sqrt( (w_**2) + (h_**2) )

            # HOOP RELATIVELY CLOSE TO ALREADY DETECTED HOOP => UPDATE DETECTED HOOP
            if distance < hypotenuse:
                self.hoops[hoopKey] = hoop
                return hoopKey

        # HOOP NOT DETECTED => ADD TO HOOPS
        self.hoops[self.hoopUid] = hoop
        self.hoopUid += 1
        return self.hoopUid-1

    def add_ball(self, ball: DetectedObject) -> int:
        '''
            Adds a detected ball to the balls dictionary. If the ball is already detected, adds to the detected ball.

            Paramaters:
                ball - DetectedObject object representing the ball detected in the current frame

            RETURNS:
                key of detected ball in balls dictionary. Returns None if not added.
        '''

        if ball.conf < 0.4 and not (self.hoop_area(ball) and ball.conf > 0.3):
            return None

        # NO BALLS TO CHECK => ADD TO BALLS
        if len(self.balls) < 1:
            self.balls[self.ballUid] = DetectedBall(ball)
            self.ballUid += 1
            return 0

        # DETERMINE IF BALL ALREADY DETECTED
        x, y = ball.x, ball.y
        valid_ball = []
        for ballKey, detectedBall in self.balls.items():
            # CALCULATE DISTANCE B/W THE BALLS
            detectedBallPrev = detectedBall.get_last_detection()
            x_, y_ = detectedBallPrev.x, detectedBallPrev.y
            distance = np.sqrt( ((x_ - x)**2) + ((y_- y)**2) )
            w_, h_ = detectedBallPrev.w, detectedBallPrev.h
            hypotenuse = np.sqrt( (w_**2) + (h_**2) )

            # DETERMINE IF BALL BELONGS TO DETECTED BALL
            if distance < hypotenuse*2 or (ballKey in [b[0] for b in self.up_ball] and distance < hypotenuse*4):
                if len(self.balls) < 2:
                    detectedBall.add_detection(ball)
                    return ballKey
                elif not valid_ball:
                    valid_ball.append(ballKey)
                    valid_ball.append(distance)
                else:
                    if distance < valid_ball[1]:
                        valid_ball[0] = ballKey
                        valid_ball[1] = distance

        # VALID BALL FOUND => ADD BALL TO VALID BALL
        if len(valid_ball) == 2:
            self.balls[valid_ball[0]].add_detection(ball)
            return valid_ball[0]

        # NO VALID BALLS => ADD TO BALLS
        self.balls[self.ballUid] = DetectedBall(ball)
        self.ballUid += 1
        return self.ballUid - 1

    def detect_up(self) -> None:
        '''
            Detects if a ball is in the area of a backboard
        '''

        # ALL BALLS ARE DETECTED AS UP => NOTHING TO CHECK
        if len(self.up_ball) == len(self.balls):
            return

        # FIND BALL IN AREA OF A HOOPS BACKBOARD
        for ballKey, ball in self.balls.items():

            # BALL IN UP_BALL OR DOWN_BALL => CONTINUE
            if ballKey in [ball_[0] for ball_ in self.up_ball] or ballKey in [ball_[0] for ball_ in self.down_ball] or len(ball.detections) < 3:
                continue

            for hoopKey, hoop in self.hoops.items():
                prevBallDetection = ball.get_last_detection()

                # SIZE(BALL) > SIZE(HOOP) => CONTINUE
                if hoop.w * hoop.h < prevBallDetection.w * prevBallDetection.h:
                    continue

                # CAlCULATE COORDINATES OF BACKBOARD
                x1 = int(hoop.x - (hoop.w * 2))
                x2 = int(x1 + (hoop.w * 4))
                y1 = int(hoop.y)
                y2 = int(y1 - (hoop.h * 3))

                # BALL IN AREA OF BACKBOARD => ADD TO UP_BALL
                if x1 < prevBallDetection.x < x2 and y2 < prevBallDetection.y < y1:
                    self.up_ball.append([ballKey, hoopKey])

    def detect_down(self) -> None:
        '''
            Detects if a ball in up_ball is below the hoop. Adds the ball-hoop pair to down_ball, and removes it from the up_ball list.
        '''

        # UP_BALL EMPTY => NOTHING TO CHECK
        if len(self.up_ball) == 0:
            return

        for pair in deepcopy(self.up_ball):

            # VALIDATE PAIR
            if len(pair) < 2 or None in pair:
                self.up_ball.remove(pair)
                continue

            ballKey, hoopKey = pair

            # VALIDATE BALL AND HOOP
            if ballKey not in self.balls or hoopKey not in self.hoops:
                self.up_ball.remove(pair)
                continue

            hoop = self.hoops[hoopKey]
            ball = self.balls[ballKey]

            # BALL BELOW HOOP => ADD TO DOWN_BALL
            y1 = int(hoop.y + (hoop.h / 2)) # BOTTOM OF NET
            if ball.get_last_detection().y > y1:
                self.down_ball.append(pair)
                self.up_ball.remove(pair)

    def update_score(self) -> None:
        '''
            Updates the makes and attempts variables by iterating through the down_ball list and calculating if the ball was between the rim when it was at the height of the center of the hoop.
        '''

        # DOWN_BALL EMPTY => NOTHING TO CHECK
        if len(self.down_ball) == 0:
            return

        for ballKey, hoopKey in deepcopy(self.down_ball):

            # VALIDATE BALL AND HOOP
            if ballKey not in self.balls or hoopKey not in self.hoops:
                self.down_ball.remove([ballKey, hoopKey])
                continue

            hoop = self.hoops[hoopKey]
            ball = self.balls[ballKey]

            # BALL INFORMATION
            prevDetection = ball.get_last_detection()
            x1, y1 = prevDetection.x, prevDetection.y
            x2, y2 = None, None

            # HOOP INFORMATION
            x_hoop, y_hoop = hoop.x, hoop.y
            hoop_top = y_hoop - (hoop.h / 2)

            # FIND THE POSITION OF THE BALL WHEN IT WAS ABOVE THE HOOP
            for b in reversed(ball.detections):
                if b.y < hoop_top:
                    x2, y2 = b.x, b.y
                    break

            # IF NO POSITION ABOVE HOOP => REMOVE PAIR FROM DOWN_BALL
            if x2 == None or y2 == None:
                self.down_ball.remove([ballKey, hoopKey])
                continue

            # CALCULATE LINE OF BALL WHEN ABOVE HOOP & BELOW
            m = (y2 - y1) / (x2 - x1)
            b = y1 - (m * x1)

            # CALCULATE X-COORDINATE OF BALL WHEN IT WAS AT HOOP HEIGHT
            x_pred = (y_hoop - b) / m

            # BALL BETWEEN RIM WHEN AT HOOP HEIGHT => INCREMENT MAKES
            x1_rim = x_hoop - (hoop.w/2)
            x2_rim = x_hoop + (hoop.w/2)
            if x1_rim < x_pred < x2_rim:
                self.makes += 1

            self.attempts += 1
            self.down_ball.remove([ballKey, hoopKey])

    def clean_detections(self) -> None:
        '''
            Cleans balls and hoops
        '''

        # REMOVE OLD DETECTIONS
        for ballkey, ball in deepcopy(self.balls).items(): #保证遍历时删除或增加数据不对造成错误；获取拷贝后字典的所有键值对
            if self.frame_count - ball.get_last_detection().frame > 20: #若球的最后一次检测到的帧数与当前帧数之差大于20帧
                self.balls.pop(ballkey) #从字典中删除该球
            if len(ball.detections) > 30: # 若球的检测记录数大于30条
                ball.detections.popleft() #从球的检测记录中删除最早的一条记录

        for hoopKey, hoop in deepcopy(self.hoops).items():
            if self.frame_count - hoop.frame > 20: #若篮筐的最后一次检测到的帧数与当前帧数之差大于20帧
                self.hoops.pop(hoopKey) #从字典中删除该篮筐

    def hoop_area(self, ball: DetectedObject) -> bool:
        '''
            Returns if a given ball position is in the area of a hoop.
        '''

        for hoop in self.hoops.values():

            # CALCULATE COORDINATES OF BACKBOARD
            x1 = int(hoop.x - (hoop.w * 2))
            x2 = int(x1 + (hoop.w * 4))
            y1 = int(hoop.y + (hoop.h / 2))
            y2 = int(y1 - (hoop.h * 3))

            # BALL IN AREA OF BACKBOARD => RETURN TRUE
            if x1 < ball.x < x2 and y2 < ball.y < y1:
                return True

        return False