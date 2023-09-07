import cv2
import numpy as np
import time

class spec_check:
    def __init__(self):
        pass

    def suffix_processing(self, image):
        kernel1 = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        kernel2 = np.array([[1 / 4, 1 / 4],
                           [1 / 4, 1 / 4]])
        image = cv2.medianBlur(image, 5)
        image = cv2.filter2D(image, -1, kernel1)
        image = cv2.filter2D(image, -1, kernel2)
        return image

    def yolo_to_ltrb(self, img, yolo_format):
        dh, dw, dc = img.shape
        # Split string to float
        c, x, y, w, h = map(float, yolo_format.split(' '))

        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)

        if l < 0:
            l = 0
        if r > dw - 1:
            r = dw - 1
        if t < 0:
            t = 0
        if b > dh - 1:
            b = dh - 1

        return c, l, t, r, b

    def spec_check(self, image="", labels="", pixel=30):
        self.image = image
        self.labels = labels
        self.pixel = pixel
        self.classes = {0: "M오염",
                        1: "MMS",
                        2: "엣지기포",
                        3: "M찍힘",
                        4: "표면기포",
                        5: "MS폭",
                        6: "F이물",
                        7: "M꺾임",
                        8: "M이물",
                        9: "테프론",
                        10: "진접",
                        11: "MS길이",
                        12: "PFS",
                        13: "뜯김",
                        14: "PMS",
                        15: "P오염",
                        16: "그리퍼",
                        17: "F찍힘",
                        18: "P찍힘",
                        19: "P꺾임",
                        20: "결무늬",
                        21: "날개처짐",
                        22: "도금박리",
                        23: "MFS",
                        24: "빨래판",
                        25: "MF뭉침",
                        26: "M음영",
                        27: "필름농",
                        28: "표면뭉침",
                        29: "날개눌림",
                        30: "MMB",
                        31: "기타",
                        32: "날개기포",
                        33: "산화",
                        34: "M갈변",
                        35: "총두께",
                        36: "실링",
                        37: "P이물",
                        38: "F꺾임",
                        39: "콜론자국",
                        40: "날개뭉침",
                        41: "엣지뭉침",
                        42: "필름겹침",
                        43: "미진접",
                        44: "미부착",
                        45: "반미착",
                        46: "치수",
                        47: "날개이물컷팅",
                        48: "표면눌림",
                        49: "엣지눌림",
                        50: "레이어",
                        51: "백색테이프",
                        52: "아지랑이",
                        53: "돌기",
                        54: "피딩",
                        55: "들뜸",
                        56: "날개주름",
                        57: "역부착",
                        58: "날개이물",
                        59: "엣지이물",
                        60: "3매부착",
                        61: "1매부착"}

        for key, val in self.classes.items():
            self.classes[key] = []

        for dt in self.labels:
            c, l, t, r, b = self.yolo_to_ltrb(image, dt)

            crob_img = image[t:b, l:r]
            cimg = image[t:b, l:r]

            cimg = self.suffix_processing(cimg)
            cimg = cv2.cvtColor(cimg, cv2.COLOR_RGB2GRAY)
            if int(c) in [0]:
                ret, thr = cv2.threshold(cimg, 150, 255, cv2.THRESH_BINARY)
            elif int(c) in [6]:
                ret, thr = cv2.threshold(cimg, 100, 255, cv2.THRESH_BINARY)
            elif int(c) in [23]:
                ret, thr = cv2.threshold(cimg, 40, 255, cv2.THRESH_BINARY)
            else:
                ret, thr = cv2.threshold(cimg, 150, 255, cv2.THRESH_BINARY)
            thr = 255 - thr
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(thr)

            max_label = [0, 0, 0, 0]  # 0: index, 1: count, 2: width, 3: height
            if retval > 1:
                temp_label = [i[-1] for i in stats]
                temp_label = temp_label[1:]
                max_label[1] = max(temp_label)  # count
                max_label[0] = temp_label.index(max_label[1])+1  # index
                max_label[2] = stats[max_label[0]][2]  # width
                max_label[3] = stats[max_label[0]][3]  # height

                self.classes[int(c)].append(max_label)

            # crob_img[labels == max_label[0]] = [0,0,255]

        # 양불 판정 --------------------------
        self.faulty_status = True
        self.found_classes = {}
        for key, val in self.classes.items():
            if len(self.classes[key]):
                self.found_classes[key] = val

        for key, val in self.found_classes.items():
            # 불량유형: M오염, F이물, M이물
            # SPEC: 기준면적 Size 0.6㎟ 이하 / 3EA 이하
            if key in [0, 6, 8, 15]:
                if len(self.found_classes[key]) > 3:
                    self.faulty_status = False
                else:
                    for c in self.found_classes[key]:
                        area_size = self.pixel * self.pixel * c[1]
                        if area_size > 600000:
                            print("area_size:", format(area_size/1000000, ','), "㎟")
                            self.faulty_status = False
            # 불량유형: MMS, M찍힘
            # SPEC: 폭 1.0mm / 개수 10EA 이하
            elif key in [1, 3]:
                if len(self.found_classes[key]) > 10:
                    self.faulty_status = False
                else:
                    for c in self.found_classes[key]:
                        area_size = self.pixel * c[2]
                        if area_size > 1000:
                            print("area_size:", format(area_size/1000, ','), "㎜")
                            self.faulty_status = False
            # 불량유형: 엣지기포, 표면기포, M꺾임, 진접, 뜯김, P꺾임, 필름농, 날개기포, F꺾임, 필름겹침, 미진접, 미부착, 반미착,
            #         날개이물컷팅, 표면눌림, 엣지눌림, 백색테이프, 들뜸, 날개주름, 역부착, 날개이물, 엣지이물, 3매부착, 1매부착
            # SPEC: 없을 것.
            elif key in [2, 4, 7, 10, 13, 19, 27, 32, 38, 42, 43, 44, 45, 47, 48, 49, 51, 55, 56, 57, 58, 59, 60, 61]:
                if len(self.found_classes[key]) > 0:
                    self.faulty_status = False
            else:
                self.faulty_status = False

        return self.faulty_status