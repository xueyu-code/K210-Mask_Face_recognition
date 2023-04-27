import sensor,image,lcd,time
import KPU as kpu
import _thread#引入多线程模块
import utime
import gc#引入垃圾回收机制
from fpioa_manager import fm
from board import board_info
from Maix import FPIOA,GPIO


WARNING={'cheak':'cheaking','unmask':'please mask','masks':'take off mask and waiting FACE cheak'}
return_meg=['cheak']

# #开始设置驱动boot按键相关的配置
BOUNCE_PROTECTION = 50
start_processing = False
#按键状态设置true函数
def set_key_state(*_):
    global start_processing
    start_processing = True
    utime.sleep_ms(BOUNCE_PROTECTION)


color_R = (255, 0, 0)
color_G = (0, 255, 0)
color_B = (0, 0, 255)


#设置摄像头参数和模式
def step1():
    lcd.init(freq=15000000)
    sensor.reset(dual_buff=False)#应设置为False否则会内存溢出
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)
    # sensor.set_windowing((224, 224))
    sensor.set_vflip(1)
    sensor.run(1)
    #开始设置驱动boot按键相关的配置
    fm.register(board_info.BOOT_KEY, fm.fpioa.GPIOHS0)
    key_gpio = GPIO(GPIO.GPIOHS0, GPIO.IN)
    #按键中断函数
    key_gpio.irq(set_key_state, GPIO.IRQ_RISING, GPIO.WAKEUP_NOT_SUPPORT)
#挂载口罩模型
def loadmodel():
    global return_meg
    classID = ['unmask','masks']
    status=0
    task = kpu.load("/sd/mask.smodel")
    anchor = (0.1606, 0.3562, 0.4712, 0.9568, 0.9877, 1.9108, 1.8761, 3.5310, 3.4423, 5.6823)
    _ = kpu.init_yolo2(task, 0.5, 0.3, 5, anchor)
    # img_lcd = image.Image()
    while status==0:
        img = sensor.snapshot()
        code = kpu.run_yolo2(task, img)
        status = 1 if code else 0
        if code:
            totalRes = len(code)

            for item in code:
                confidence = float(item.value())
                itemROL = item.rect()
                classID = int(item.classid())

                if confidence < 0.52:
                    _ = img.draw_rectangle(itemROL, color=color_B, tickness=5)
                    continue

                if classID == 1 and confidence > 0.65:
                    _ = img.draw_rectangle(itemROL, color_G, tickness=5)
                    if totalRes == 1:
                        drawConfidenceText(img, (0, 0), 1, confidence)
                        # print("MASK!!!!")
                        return_meg=['masks']
                else:
                    _ = img.draw_rectangle(itemROL, color=color_R, tickness=5)
                    if totalRes == 1:
                        drawConfidenceText(img, (0, 0), 0, confidence)

        else:
            # lcd.display(img)
            lcd.draw_string(50,10,'cheaking', lcd.RED,lcd.WHITE)
        _ = lcd.display(img)
    kpu.deinit(task)
    gc.collect()
#挂载人脸模型
def facemodel():
    # global return_meg
    global BOUNCE_PROTECTION
    global start_processing
    # sensor.set_windowing((224, 224))

    task_fd = kpu.load("/sd/FaceDetection.smodel")
    task_ld = kpu.load("/sd/FaceLandmarkDetection.smodel")
    task_fe = kpu.load("/sd/FeatureExtraction.smodel")

    anchor = (1.889, 2.5245, 2.9465, 3.94056, 3.99987, 5.3658, 5.155437, 6.92275, 6.718375, 9.01025) #anchor for face detect 用于人脸检测的Anchor
    dst_point = [(44,59),(84,59),(64,82),(47,105),(81,105)] #standard face key point position 标准正脸的5关键点坐标 分别为 左眼 右眼 鼻子 左嘴角 右嘴角
    a = kpu.init_yolo2(task_fd, 0.5, 0.3, 5, anchor) #初始化人脸检测模型
    # img_lcd=image.Image() # 设置显示buf
    img_face=image.Image(size=(128,128)) #设置 128 * 128 人脸图片buf
    a=img_face.pix_to_ai() # 将图片转为kpu接受的格式
    record_ftr=[] #空列表 用于存储当前196维特征
    name = ['person_Xu']
    names = ['M1', 'M2', 'M3'] # 人名标签，与上面列表特征值一一对应。
    record_ftrs=[] #空列表 用于存储按键记录下人脸特征， 可以将特征以txt等文件形式保存到sd卡后，读取到此列表，即可实现人脸断电存储。
    person_Xu=[(b'\x15\x13}w\'\xb6\xe3\x0b\xd2O)o\x89\x0f4\xd6\x80\x95\xde0\xcd\x12\xff\x01\xe2\xca\xca\xce#\xd3\x19\'\x02H\xd5\xa9s\xebR\'0\x0e"\x00\xf1\x03\x15\x80\xf6\xf3\x11\xfb\xb9\x1f\x1b\xb9o\xb9\x8ck\x80\xbe#\xd5@\xdb`\x13F\xbc\xf5!\x80\x1b\xd6\xd5\xf3\x11\xe9]\xe5<%@\xc4\r\xb5\x97\x12)\xf5\x06\xa8HD\t\xb5\x1e\xd7\x81\xe9\xf6\xaaL\xb4\x9c)\xd1\xf5/\x01d\xb0\x1b\xb5:\xd6\xf5\xe6/\xac\xd0\xd7\xe63;\x80\x02C\x05\x17\x8c<\xd9i\x1d\x16\x1b\x13\xe3\xbe\xea\r\xd2\x94\xe2\n\xef\x89\xe6BK.Je\x7fp\x17|\xef/d\xae\t\xfb!>\xf2S6\x80\xe5\x9fX\x13\xff\x16N\x07\x11\xc4\xfd\xf1\xde\x803\x11\xe1\x1eL\xe3\xe7\xd0\xa0\xe9\xed')]
    while(1): # 主循环
        img = sensor.snapshot() #从摄像头获取一张图片
        obj = kpu.run_yolo2(task_fd, img) # 运行人脸检测模型，获取人脸坐标位置
        status = 1 if obj else 0
        if obj: # 如果检测到人脸
            for i in obj: # 迭代坐标框
                # Cut face and resize to 128x128
                a = img.draw_rectangle(i.rect(),(138,43,226),thickness=3) # 在屏幕显示人脸方框
                face_cut=img.cut(i.x(),i.y(),i.w(),i.h()) # 裁剪人脸部分图片到 face_cut
                face_cut_128=face_cut.resize(128,128) # 将裁出的人脸图片 缩放到128 * 128像素
                a=face_cut_128.pix_to_ai() # 将猜出图片转换为kpu接受的格式
                #a = img.draw_image(face_cut_128, (0,0))
                # Landmark for face 5 points
                fmap = kpu.forward(task_ld, face_cut_128) # 运行人脸5点关键点检测模型
                plist=fmap[:] # 获取关键点预测结果
                le=(i.x()+int(plist[0]*i.w() - 10), i.y()+int(plist[1]*i.h())) # 计算左眼位置， 这里在w方向-10 用来补偿模型转换带来的精度损失
                re=(i.x()+int(plist[2]*i.w()), i.y()+int(plist[3]*i.h())) # 计算右眼位置
                nose=(i.x()+int(plist[4]*i.w()), i.y()+int(plist[5]*i.h())) #计算鼻子位置
                lm=(i.x()+int(plist[6]*i.w()), i.y()+int(plist[7]*i.h())) #计算左嘴角位置
                rm=(i.x()+int(plist[8]*i.w()), i.y()+int(plist[9]*i.h())) #右嘴角位置
                a = img.draw_circle(le[0], le[1], 4)
                a = img.draw_circle(re[0], re[1], 4)
                a = img.draw_circle(nose[0], nose[1], 4)
                a = img.draw_circle(lm[0], lm[1], 4)
                a = img.draw_circle(rm[0], rm[1], 4) # 在相应位置处画小圆圈
                # align face to standard position
                src_point = [le, re, nose, lm, rm] # 图片中 5 坐标的位置
                T=image.get_affine_transform(src_point, dst_point) # 根据获得的5点坐标与标准正脸坐标获取仿射变换矩阵
                a=image.warp_affine_ai(img, img_face, T) #对原始图片人脸图片进行仿射变换，变换为正脸图像
                a=img_face.ai_to_pix() # 将正脸图像转为kpu格式
                #a = img.draw_image(img_face, (128,0))
                del(face_cut_128) # 释放裁剪人脸部分图片
                # calculate face feature vector
                fmap = kpu.forward(task_fe, img_face) # 计算正脸图片的196维特征值
                feature=kpu.face_encode(fmap[:]) #获取计算结果
                scores = [] # 存储特征比对分数
                mscores = [] #存储我们的特征比对分数
                #识别其他人
                for x in range(len(record_ftrs)): #迭代已存特征值
                    score = kpu.face_compare(record_ftrs[x], feature) #计算当前人脸特征值与已存特征值的分数
                    scores.append(score) #添加分数总表
                max_score = 0
                index = 0
                #识别王昌胜
                for j in range(len(person_Xu)): #迭代已存特征值
                    mscore = kpu.face_compare(person_Xu[j], feature) #计算当前人脸特征值与已存特征值的分数
                    mscores.append(mscore) #添加分数总表
                mmax_score = 0

                for k in range(len(scores)): #迭代所有比对分数，找到最大分数和索引值
                    if max_score < scores[k]:
                        max_score = scores[k]
                        index = k
                if max_score > 75: # 如果最大分数大于85， 可以被认定为同一个人
                    a = img.draw_string(i.x(),i.y(), ("%s :%2.1f" % (names[index], max_score)), color=(0,255,0),scale=3) # 显示人名 与 分数
                for x in range(len(mscores)): #迭代所有比对分数，找到最大分数和索引值
                    if mmax_score < mscores[x]:
                        mmax_score = mscores[x]
                        num = x
                if mmax_score > 75: # 如果最大分数大于85， 可以被认定为同一个人
                        a = img.draw_string(i.y(),i.x(), ("%s :%2.1f" % (name[num], mmax_score)), color=(0,255,0),scale=3) # 显示人名 与 分数
                else:
                    a = img.draw_string(i.h(),i.w(), ("%s Not" % (max_score)), color=(255,0,0),scale=4) #显示未知 与 分数
                if start_processing: #如果检测到按键
                    record_ftr = feature
                    record_ftrs.append(record_ftr) #将当前特征添加到已知特征列表
                    start_processing = False#重置按键状态标志位
                break
        else:
            lcd.display(img)
            lcd.draw_string(50,10,'wait FACE cheak', lcd.RED,lcd.WHITE)
        a = lcd.display(img) #刷屏显示
    kpu.deinit(task_fd)
    kpu.deinit(task_ld)
    kpu.deinit(task_fe)
    gc.collect()
    # time.sleep(2)

#在显示屏上输出字符
def drawConfidenceText(image, rol, classid, value):
    text = ""
    _confidence = int(value * 100)

    if classid == 1:
        text = 'masks: ' + str(_confidence) + '%'
        color_text=color_G
    else:
        text = 'unmask: ' + str(_confidence) + '%'
        color_text=color_R
    image.draw_string(rol[1], rol[0], text, color=color_text, scale=2.5)



if __name__ == "__main__":
    _thread.start_new_thread(step1(),(0,))
    # _thread.start_new_thread(loadmodel(),(1,))
    while True:
        _thread.start_new_thread(loadmodel(),(1,))
        if return_meg[0] == 'masks':
            _thread.start_new_thread(facemodel(),(1,))
            return_meg[0] = 'unmask'
            # time.sleep(2)
        #gc.collect()




