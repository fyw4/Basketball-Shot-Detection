class DetectedObject:
    def __init__(self, x: int, y: int, w: int, h: int, frame: int, conf: float):
        self.x = x # 物体的 x 坐标，表示物体在图像中的水平位置
        self.y = y # 物体的 y 坐标，表示物体在图像中的垂直位置
        self.w = w # 物体的宽度，表示物体在图像中的水平大小
        self.h = h # 物体的高度，表示物体在图像中的垂直大小
        self.frame = frame # 物体检测到的帧号，表示物体在视频中的位置
        self.conf = conf # 物体检测的置信度，表示物体被正确检测到的概率，值越大表示越有可能是正确的检测结果

    def __eq__(self, other):
        if not other:
            return False
        return self.x == other.x and self.y == other.y and self.w == other.w and self.h == other.h and self.frame == other.frame and self.conf == other.conf