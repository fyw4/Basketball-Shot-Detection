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
        self.hoops = {} # 所有检测到的篮筐 ，key为唯一标识符，value为检测到的篮筐对象
        self.hoopUid = 0 # 篮筐的唯一标识符

        self.balls = {} # 所有检测到的球，key为唯一标识符，value为DetectedBall对象队列，包含该球的检测历史记录
        self.ballUid = 0 # 球的唯一标识符

        self.frame_count = 0 # 当前帧序号

        # BALLS IN THE AREA OF A HOOP
        self.up_ball = [] # 所有向上的球
        self.down_ball = [] # 所有向下的球

        self.attempts = 0 # 投篮总次数
        self.makes = 0 # 成功投篮次数

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
                        object = DetectedObject(center[0], center[1], w, h, self.frame_count, conf) #创建一个DetectedObject对象，包含检测框的中心坐标、宽度、高度、当前帧序号、置信度分数
                        if cls == 1:
                            index = self.add_hoop(object)
                        else:
                            index = self.add_ball(object)

                        if index == None:
                            continue

                        # PERFORM SHOT DETECTION
                        self.detect_up() # 球进入篮板上方区域判定为方向向上
                        self.detect_down() # 低于篮筐判定为方向向下
                        self.update_score() # 更新投篮统计信息

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

        if hoop.conf < 0.3:#置信程度小于0.3则不认为是篮筐
            return None

        # NO HOOPS TO CHECK => ADD TO HOOPS
        if len(self.hoops) == 0:
            self.hoops[self.hoopUid] = hoop
            self.hoopUid += 1
            return self.hoopUid - 1

        # DETERMINE IF HOOP ALREADY DETECTED
        #判断是否已经检测过该篮筐，如果检测过更新检测到的篮筐对象，如果没有检测过则添加到篮筐字典中
        x, y = hoop.x, hoop.y
        for hoopKey, detectedHoop in self.hoops.items():
            # CALCULATE DISTANCE B/W THE HOOPS
            x_, y_ = detectedHoop.x, detectedHoop.y
            w_, h_ = detectedHoop.w, detectedHoop.h
            distance = np.sqrt( ((x_ - x)**2) + ((y_- y)**2) ) #求当前篮筐与已检测的篮筐之间的距离
            hypotenuse =  np.sqrt( (w_**2) + (h_**2) ) #求篮筐对角线长度

            # HOOP RELATIVELY CLOSE TO ALREADY DETECTED HOOP => UPDATE DETECTED HOOP
            # 如果distance距离小于对角线hypotenuse长度，认为是同一个篮筐
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

        if ball.conf < 0.4 and not (self.hoop_area(ball) and ball.conf > 0.3): #如果球的信度小于0.4且不在篮筐区域，也不认为是球
            return None

        # NO BALLS TO CHECK => ADD TO BALLS
        if len(self.balls) < 1:
            self.balls[self.ballUid] = DetectedBall(ball) # 初始化时，如果提供了球的位置，就添加到检测队列中
            self.ballUid += 1
            return 0

        # DETERMINE IF BALL ALREADY DETECTED
        x, y = ball.x, ball.y
        valid_ball = [] #列表list
        for ballKey, detectedBall in self.balls.items():
            # CALCULATE DISTANCE B/W THE BALLS
            detectedBallPrev = detectedBall.get_last_detection() #获取球最后的位置数据
            x_, y_ = detectedBallPrev.x, detectedBallPrev.y
            distance = np.sqrt( ((x_ - x)**2) + ((y_- y)**2) ) #求当前球与已检测的球之间的距离
            w_, h_ = detectedBallPrev.w, detectedBallPrev.h #获取已检测的球的宽度和高度
            hypotenuse = np.sqrt( (w_**2) + (h_**2) ) #求球对角线长度

            # DETERMINE IF BALL BELONGS TO DETECTED BALL
            #若当前球与已检测的球之间的距离小于对角线长度的2倍，或者当前球是向上的球，且当前球与已检测的球之间的距离小于对角线长度的4倍，认为是同一个球
            if distance < hypotenuse*2 or (ballKey in [b[0] for b in self.up_ball] and distance < hypotenuse*4):
                if len(self.balls) < 2: #检查当前系统中已跟踪的篮球数量是否少于2个
                    detectedBall.add_detection(ball)  #将新检测到的篮球位置添加到已存在的篮球跟踪记录中
                    return ballKey
                elif not valid_ball: # 如果当前系统中没有已跟踪的篮球，那么就将当前检测到的篮球位置添加到 valid_ball 中
                    valid_ball.append(ballKey)
                    valid_ball.append(distance)
                else:
                    if distance < valid_ball[1]: #如果当前检测到的篮球位置与已存在的篮球跟踪记录之间的距离小于 valid_ball 中记录的距离，那么就更新 valid_ball 中的篮球键值对和距离
                        valid_ball[0] = ballKey
                        valid_ball[1] = distance

        # VALID BALL FOUND => ADD BALL TO VALID BALL
        if len(valid_ball) == 2:# 找到了有效匹配
            self.balls[valid_ball[0]].add_detection(ball) # 更新篮球位置
            return valid_ball[0]  # 返回匹配的篮球键

        # NO VALID BALLS => ADD TO BALLS
        self.balls[self.ballUid] = DetectedBall(ball)
        self.ballUid += 1
        return self.ballUid - 1

    def detect_up(self) -> None:
        '''
            Detects if a ball is in the area of a backboard
        '''

        # ALL BALLS ARE DETECTED AS UP => NOTHING TO CHECK
        if len(self.up_ball) == len(self.balls): # 所有球都被检测为向上了，那么就没有必要继续检查了
            return

        # FIND BALL IN AREA OF A HOOPS BACKBOARD
        for ballKey, ball in self.balls.items(): # 遍历所有球

            # BALL IN UP_BALL OR DOWN_BALL => CONTINUE
            if ballKey in [ball_[0] for ball_ in self.up_ball] or ballKey in [ball_[0] for ball_ in self.down_ball] or len(ball.detections) < 3:
                continue

            for hoopKey, hoop in self.hoops.items(): # 遍历所有篮筐
                prevBallDetection = ball.get_last_detection()

                # SIZE(BALL) > SIZE(HOOP) => CONTINUE
                if hoop.w * hoop.h < prevBallDetection.w * prevBallDetection.h: # 球的面积大于 hoop 的面积，那么球一定不在 hoop 的 backboard 上
                    continue

                # CAlCULATE COORDINATES OF BACKBOARD
                # 计算篮板区域坐标
                x1 = int(hoop.x - (hoop.w * 2))  #左边界：hoop 的中心减去 hoop 的宽度的两倍
                x2 = int(hoop.x + (hoop.w * 2))  #右边界：hoop 的中心加上 hoop 的宽度的两倍
                y1 = int(hoop.y)  #上边界：hoop 的中心
                y2 = int(y1 - (hoop.h * 3))  #下边界：hoop 的中心减去 hoop 的高度的三倍

                # BALL IN AREA OF BACKBOARD => ADD TO UP_BALL
                if x1 < prevBallDetection.x < x2 and y2 < prevBallDetection.y < y1: # 球的中心坐标在篮板区域内
                    self.up_ball.append([ballKey, hoopKey]) # 将球和篮筐的键值对添加到 up_ball 中

    def detect_down(self) -> None:
        '''
            Detects if a ball in up_ball is below the hoop. Adds the ball-hoop pair to down_ball, and removes it from the up_ball list.
        '''

        # UP_BALL EMPTY => NOTHING TO CHECK
        if len(self.up_ball) == 0:
            return

        for pair in deepcopy(self.up_ball): # 遍历 up_ball 中的所有键值对

            # VALIDATE PAIR
            if len(pair) < 2 or None in pair: # 如果键值对的长度小于2或者包含 None，说明这个键值对无效，继续检查下一个键值对
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
                self.down_ball.append(pair) # 将球和篮筐的键值对添加到 down_ball 中
                self.up_ball.remove(pair) # 从 up_ball 中移除这个键值对

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
            for b in reversed(ball.detections): #反向遍历篮球的检测历史记录
                if b.y < hoop_top:  # 如果球的位置高于篮筐顶部
                    x2, y2 = b.x, b.y  # 记录这个位置
                    break

            # IF NO POSITION ABOVE HOOP => REMOVE PAIR FROM DOWN_BALL
            if x2 == None or y2 == None: # 如果没有找到高于篮筐顶部的位置
                self.down_ball.remove([ballKey, hoopKey])
                continue

            # CALCULATE LINE OF BALL WHEN ABOVE HOOP & BELOW
            m = (y2 - y1) / (x2 - x1)# 计算球在高于篮筐位置和低于篮筐位置之间的斜率，避免除以零的情况，如果 x2 和 x1 相等，则斜率设为0
            b = y1 - (m * x1)# 计算球在高于篮筐位置和低于篮筐位置之间的截距，避免除以零的情况，如果 x2 和 x1 相等，则截距设为0

            # CALCULATE X-COORDINATE OF BALL WHEN IT WAS AT HOOP HEIGHT
            x_pred = (y_hoop - b) / m # 计算球在 hoop筐中心高度下的 x 坐标，避免除以零的情况，如果 m 等于0，则 x_pred 等于 b

            # BALL BETWEEN RIM WHEN AT HOOP HEIGHT => INCREMENT MAKES
            x1_rim = x_hoop - (hoop.w/2) # 篮筐的左边界：hoop 的中心减去 hoop 的宽度的一半
            x2_rim = x_hoop + (hoop.w/2) # 篮筐的右边界：hoop 的中心加上 hoop 的宽度的一半
            if x1_rim < x_pred < x2_rim: # 如果球在 hoop筐中心高度下的 x 坐标在篮筐的左右边界之间
                self.makes += 1

            self.attempts += 1
            self.down_ball.remove([ballKey, hoopKey]) # 从 down_ball 中移除这个键值对

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

    # 检测是否在篮筐区域内，篮筐区域定义为以篮筐中心为中心，宽度为篮筐宽度的4倍，高度为篮筐高度的3倍的矩形区域
    def hoop_area(self, ball: DetectedObject) -> bool:
        '''
            Returns if a given ball position is in the area of a hoop.
        '''

        for hoop in self.hoops.values(): # .values()意思是获取字典中所有的值（value），而不包含键（key）。

            # CALCULATE COORDINATES OF BACKBOARD
            x1 = int(hoop.x - (hoop.w * 2))
            x2 = int(hoop.x + (hoop.w * 2))
            y1 = int(hoop.y + (hoop.h / 2))
            y2 = int(y1 - (hoop.h * 3))

            # BALL IN AREA OF BACKBOARD => RETURN TRUE
            if x1 < ball.x < x2 and y2 < ball.y < y1:
                return True

        return False