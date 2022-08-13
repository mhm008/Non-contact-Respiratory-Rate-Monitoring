import argparse
import time
import cv2
import torch
import numpy as np
from scipy import signal

from siamfc import TrackerSiamFC
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import jadeR    #  独立成分分析(ICA)

# https://github.com/NklausMikealson/SiameseFC-pytorch

# #  呼吸率对应的频率:0.15-0.40Hz，https://www.nature.com/articles/s41598-019-53808-9
# RR_Min_HZ = 0.15
RR_Min_HZ = 0.15
RR_Max_HZ = 0.70
# RR_Max_HZ = 0.40
# 采样频率
FPS = 25

def parse_args():
    """
    args for testing.
    """
    parser = argparse.ArgumentParser(
        description='PyTorch SiamFC Tracking Demo')
    parser.add_argument('--video', dest='video',
                        default='./test.mp4', help='video path')
    parser.add_argument('--model', dest='model',
                        default='pretrained/siamfc/model.pth', help='pretrained model')
    args = parser.parse_args()

    return args


def _x1y1wh_to_xyxy(bbox_x1y1wh):
    x1, y1, w, h = bbox_x1y1wh
    x2 = int(x1+w)
    y2 = int(y1+h)
    return x1, y1, x2, y2

def readvideo_infrared(datapath_infrared):
    img_num = 1500
    vc = cv2.VideoCapture(datapath_infrared)  # 读取视频文件
    c = 0
    count_imgs_num = 0
    videonpy = []
    timeF_infrared = 1
    if vc.isOpened():  # 判断是否正常打开
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        rval, frame = vc.read()
        if (c % timeF_infrared == 0):  # 每隔timeF帧加到数组内;  # timeF 视频帧计数间隔频率
            if frame is not None:
                videonpy.append(frame)
                count_imgs_num = count_imgs_num + 1
        c = c + 1
        cv2.waitKey(1)
        if c >= img_num:
            break
    vc.release()
    videonpy = np.array(videonpy)
    return videonpy

# 计算img_ROI转化成灰度图之后的平均像素值
def get_avg_gray_pixel(img_ROI):
    gray_img = cv2.cvtColor(img_ROI, cv2.COLOR_BGR2GRAY)
    avg_pixel = np.mean(gray_img)
    # cv2.imshow('gray_img', gray_img)
    return avg_pixel


def infrared_preprocessing(signals):
    # 去趋势
    detrend_signals = detrend(signals, 100)
    # detrend_signals = detrend(signals_pre, 100)
    detrend_signals = detrend_signals.flatten()
    # 标准化
    normalized_signals = normalize(detrend_signals)
    # 滤波
    filtered_signals = filter_signal_infrared(normalized_signals)
    return filtered_signals

def normalize(signals):
    # print('func:normalize>>{},type:{}'.format(signals,type(signals)))
    # signals 空值或者zero值，分母计算会提示 warning，
    # 虽然已经在文件头添加忽略,但建议判断最佳
    normalized_signals = (signals - np.mean(signals)) / np.std(signals, ddof=1)
    # normalized_signals = (signals - np.min(signals)) / (np.max(signals)-np.min(signals))
    return normalized_signals

# def detrend(signals, param_lambda):
#     # https://blog.csdn.net/piaoxuezhong/article/details/79211586
#     signal_len = len(signals)
#     I = np.identity(signal_len)
#     B = np.array([1, -2, 1])

#     # 当 signal_len - 2<0情况，出现负值问题未判断！ 例如signals:[],这部分不懂判断后处理方式
#     ones = np.ones((signal_len - 2, 1))
#     multi = B * ones
#     D2 = np.zeros((signal_len - 2, signal_len))
#     for i in range(D2.shape[0]):
#         D2[i, i:i + 3] = multi[i]
#     tr_D2 = np.transpose(D2)
#     multi_D2 = np.dot(tr_D2, D2)
#     inverse = I - (np.linalg.inv(I + (multi_D2 * pow(param_lambda, 2))))
#     detrend_signals = np.dot(inverse, signals)
#     return detrend_signals


def detrend(X, detLambda=10):  # new detrend
    import scipy
    """
    desc: get rid of a randomness trend might deal with sudden increase trend coming from head movements

    args:
        - X::[array<float>]
            signal
    ret:
        - detrendedX::[array<float>]
            detrended signal
    """
    # Smoothness prior approach as in the paper appendix:
    # "An advanced detrending method with application to HRV analysis"
    # by Tarvainen, Ranta-aho and Karjaalainen
    t = X.shape[0]
    l = t / detLambda  # lambda
    I = np.identity(t)
    D2 = scipy.sparse.diags([1, -2, 1], [0, 1, 2], shape=(t - 2, t)).toarray()  # this works better than spdiags in python
    detrendedX = (I - np.linalg.inv(I + l ** 2 * (np.transpose(D2).dot(D2)))).dot(X)

    return detrendedX

def filter_signal_infrared(signals):
    filtered_signals = butterworth_filter(signals, RR_Min_HZ, RR_Max_HZ, FPS, order=5)
    return filtered_signals

def butterworth_filter(data, low, high, sample_rate, order=2):
    nyquist_rate = sample_rate * 0.5
    low /= nyquist_rate
    high /= nyquist_rate
    b, a = signal.butter(N=order, Wn=[low, high], btype='bandpass')
    return signal.lfilter(b, a, data)

def rr_extraction(PPG_values):
    # 傅里叶变换
    fft = np.abs(np.fft.rfft(PPG_values))
    buffer_size = len(PPG_values)
    # 当 buffer_size==0 的问题需要判断
    if buffer_size == 0: #3-19
        rr_value = 0
    else:
        freqs = FPS / buffer_size * np.arange(buffer_size / 2 + 1)
        # 找到在正常呼吸率范围内的频率最高值
        while True:
            max_idx = fft.argmax()  # 寻找数组的最大索引值
            bps = freqs[max_idx]
            if bps < RR_Min_HZ or bps > RR_Max_HZ:
                fft[max_idx] = 0
            else:
                rr_value = bps * 60.0
                break
        # print('rr:',rr_value)
    return rr_value


def eliminate_abnormal_peaks(index_arr, PPG_nose, rate, total_num):
    peaks_values_arr = []
    for i in range(len(index_arr)):
        if PPG_nose[index_arr[i]] > 0:
            peaks_value = PPG_nose[index_arr[i]]
            peaks_values_arr.append(peaks_value)

    # print('peaks_values_arr:',peaks_values_arr)
    avg_value_peaks = np.mean(peaks_values_arr)

    # print('avg_value_peaks:',avg_value_peaks)

    rr_peak_count = 0
    max_value_peak = avg_value_peaks + avg_value_peaks * rate
    min_value_peak = avg_value_peaks-avg_value_peaks * rate
    # print('max_value_peak:',max_value_peak)
    # print('min_value_peak:',min_value_peak)
    peak_index = []
    for i in range(len(index_arr)):
        if PPG_nose[index_arr[i]] >= min_value_peak:
            peak_index.append(index_arr[i])
            rr_peak_count = rr_peak_count+1
    # print('peak_index:', peak_index)

    peak_distance_sum = []
    for j in range(len(peak_index)):
        if j >= 1:
            peak_distance = peak_index[j]-peak_index[j-1]
            peak_distance_sum.append(peak_distance)

    avg_peak_dis = np.mean(peak_distance)
    # print('peak_distance:', peak_distance)
    # print('avg_peak_dis:', avg_peak_dis)
    # print('peak_index[0]:', peak_index[0])
    # print('peak_index[len(peak_index)]:', peak_index[len(peak_index)-1])
    # print('total_num - peak_index[len(peak_index)-1]:', total_num - peak_index[len(peak_index)-1])

    # print(' peak_index[len(peak_index)-1]', peak_index[len(peak_index)-1])
    # print(' peak_index[0]', peak_index[0])

    # print('(total_num - peak_index[len(peak_index)-1])/avg_peak_dis:',
    #       (total_num - peak_index[len(peak_index) - 1]) / avg_peak_dis)
    # 0601
    decimal = (total_num - peak_index[len(peak_index) - 1] + peak_index[0]) / avg_peak_dis
    # print('decimal:', decimal)
    # 0601
    rr_peak_count = rr_peak_count + decimal - 1

    return rr_peak_count

def peak_index(pre_signal_arr):
    max_data = pre_signal_arr[0]
    max_index = 0
    for i in range(len(pre_signal_arr)):
        if pre_signal_arr[i] >= max_data:
            max_index = i
    return max_index


#  滑动窗口：经过预处理之后的信号，使用滑动窗口，算出波峰波谷的数量
def sliding_window(signals, width):    # signals:信号数组1*M；width：窗口大小；代码来自erfuyuan0409.py
    peak_array = []
    valley_array = []
    peak_index_arr = []
    # 检测波峰和波峰位置，滑动窗口，用当前窗和上一窗及下一窗的最大值比较，比它们大则为波峰
    for i in range(width, len(signals) - width, width):
        # print('i:',i)
        pre_max = np.max(signals[i - width:i])   # 前一时刻滑动窗口
        max = np.max(signals[i:i + width])       # 当前时刻的滑动窗口
        max_index = peak_index(signals[i:i + width])
        if i > width:
            max_index = max_index + width
        next_max = np.max(signals[i + width:i + width + width])  # 后一时刻的滑动窗口
        if max > pre_max or max > next_max:
            peak_array.append(max)
            peak_index_arr.append(max_index)
    print('peak_index_arr:',peak_index_arr)
    # 检测波谷和波谷位置，滑动窗口，用当前窗和上一窗及下一窗的最小值比较，比它们小则为波谷

    print('peak_array:', peak_array)
    print('valley_array:', valley_array)
    print('len(peak_array):', len(peak_array))
    print('len(valley_array):', len(valley_array))
    return peak_array
    
def smooth_data(signal): # 对信号进行平滑处理，相当于高斯拟合-2022-07-08
    # https://www.delftstack.com/zh/howto/python/smooth-data-in-python/
    from scipy.signal import savgol_filter
    win_len = 91 # 影响波峰拟合
    polyorder = 3 # 影响呼吸信号的幅值拟合
    signal_smooth = savgol_filter(signal, win_len, polyorder)
    return signal_smooth

def draw_ROI_line(x_data_arr, y_data_arr, image, index_ROI):
    points = []
    img_ROI = []
    xmin = x_data_arr[index_ROI[0]] * image.shape[1]
    xmax = x_data_arr[index_ROI[0]] * image.shape[1]
    ymin = y_data_arr[index_ROI[0]] * image.shape[0]
    ymax = y_data_arr[index_ROI[0]] * image.shape[0]

    for kk in range(len(index_ROI) - 1):
        x1 = int(x_data_arr[index_ROI[kk]] * image.shape[1])
        y1 = int(y_data_arr[index_ROI[kk]] * image.shape[0])
        x2 = int(x_data_arr[index_ROI[kk + 1]] * image.shape[1])
        y2 = int(y_data_arr[index_ROI[kk + 1]] * image.shape[0])
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)  # 连线

        # 求最大最小值
        if x1 >= xmax:
            xmax = x1
        else:
            xmin = x1

        if y1 >= ymax:
            ymax = y1
        else:
            ymin = y1

    # print('xmin = ', xmin)
    # print('xmax = ', xmax)
    # print('ymin = ', ymin)
    # print('ymax = ', ymax)
    img_ROI = image[int(ymin):int(ymax), int(xmin):int(xmax)]

    # 初始状态的位置信息 [x, y, w, h]
    w = xmax - xmin
    h = ymax - ymin
    init_state = [xmin, ymin, w, h]  

    return img_ROI, init_state

def get_face_landmarks(face_landmarks):

    # https://so.muouseo.com/qa/pvw0v2yxx6j1.html
    x_data_arr, y_data_arr, z_data_arr =[],[], []
    for landmark in face_landmarks.landmark:
        x = landmark.x
        y = landmark.y
        z = landmark.z
        x_data_arr.append(x)
        y_data_arr.append(y)
        z_data_arr.append(z)
   
    return x_data_arr,  y_data_arr, z_data_arr

def main(args):

    #################### mediapipe人脸检测 #####################
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    # For static images:
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5)
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    # For webcam input:
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    #################### mediapipe人脸检测 #####################

    # 目标追踪算法初始化
    trk = TrackerSiamFC(net_path=args.model)
    # for k in range(274):
    start_time_0 =  time.clock()
    # cap = cv2.VideoCapture(args.video)
    rr_arr = []
    rr_arr_time_frequency = []
    # datapath_infrared = '2022-0803-1-GT13.mp4'  # nose = [169, 420, 55, 45]
    # datapath_infrared = '2022-0803-2-GT12.mp4' # nose =  [204, 413, 57, 42]

    datapath_infrared  = '2022-08-04-1.mp4' # init_state =  [158, 368, 60, 39]
    # datapath_infrared  = '2022-08-04-2.mp4' # init_state =  [114, 410, 62, 32]
    # datapath_infrared  = '2022-08-04-3.mp4'
   
    img_arr = readvideo_infrared(datapath_infrared)
    print('len(img_arr):',len(img_arr))
    total_num = len(img_arr)
    ppg_infrared_nose = []
    face_mark = 0
    init_state = []
    for i in range(total_num):
        image = img_arr[i]
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # cv2.imshow('image', img)

        #################### mediapipe人脸检测 #####################
        if face_mark == 0:
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 获取特征点
                    # face_landmarks_str = str(face_landmarks)
                    x_data_arr, y_data_arr, z_data_arr = get_face_landmarks(face_landmarks)
                    for k in range(len(x_data_arr)):
                          # print('len(x_data_arr) = ',len(x_data_arr))
                          ## 画特征点
                          x = int(x_data_arr[k]*image.shape[1])
                          y = int(y_data_arr[k]*image.shape[0])
                          pt_pos = (x, y)
                          # cv2.circle(image, pt_pos, 1, (0, 255, 0), 1) # 画点
                          # 眼睛
                          eyeRight = [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382,362]
                          eyeLeft = [7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33, 7]
                          # 眉毛
                          eyebrowRight = [336,296,334,293,300,276,283,282,295,285,336]
                          eyebrowLeft =[70,63,105,66,107,55,65,52,53,46,70]
                          
                          # 嘴唇
                          lipUpper = [61,76,62,78,191,80,81,82,13,312,311,310,415,308,291,409,270,269,267,0,37,39,40,185,61]
                          lipLower = [61,76,62,78,95,88,178,87,14,317,402,318,324,308,291,375,321,405,314,17,84,181,91,146,61]

                          # 鼻子
                          # nose = [129, 358, 322, 92, 129]
                          nose = [209, 429, 393, 165, 129]

                          # 额头
                          forehead = [109, 338, 337, 108, 109]

                    nose_roi, init_state = draw_ROI_line(x_data_arr, y_data_arr, image, nose)
                    print('init_state = ', init_state)
                    if len(init_state) != 0 and init_state[3]> 5: # 初始化追踪模型；并标记被检测到人脸
                        trk.init(image, init_state)
                        face_mark = 1
                        cv2.imshow('nose_roi', nose_roi)
                        signal_infrared_nose = get_avg_gray_pixel(nose_roi)
                        ppg_infrared_nose.append(signal_infrared_nose)
                        # cv2.waitKey(0)
        #################### mediapipe人脸检测 #####################    
        else:
            # print('init_state = ', init_state)
            pos = trk.update(image)
            pos = _x1y1wh_to_xyxy(pos)
            pos = [int(l) for l in pos]

            img_infrared = image
            nose_roi = img_infrared[pos[1]:pos[3], pos[0]:pos[2]] 
            signal_infrared_nose = get_avg_gray_pixel(nose_roi)
            ppg_infrared_nose.append(signal_infrared_nose)
           
            # cv2.imshow('image', image)
            cv2.imshow('nose_roi', nose_roi)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    # 方法一： 先平滑处理，再对信号进行预处理
    ppg_infrared_nose = np.array(ppg_infrared_nose)
    signal_smooth = smooth_data(ppg_infrared_nose)
    PPG_nose = infrared_preprocessing(signal_smooth)

    # 方法二： 没有平滑处理的步骤，直接对信号进行预处理。
    PPG_nose_1 = infrared_preprocessing(ppg_infrared_nose)
    # fig1 = plt.figure()
    # plt.plot(ppg_infrared_nose)
    # plt.show()

    fig0 = plt.figure()
    plt.plot(signal_smooth)
    plt.show()
    
    # 波峰检测
    indices = find_peaks(PPG_nose, height=None, threshold=None, distance=5,
                                 prominence=None, width=None, wlen=None, rel_height=None,
                                 plateau_size=None)
    indices_1 = find_peaks(PPG_nose_1, height=None, threshold=None, distance=5,
                                 prominence=None, width=None, wlen=None, rel_height=None,
                                 plateau_size=None)
    print('indices:', indices)
    print('indices_1:', indices_1)

    # 剔除错误波峰
    rate1 = 0.25
    RR1 = eliminate_abnormal_peaks(indices[0], PPG_nose, rate1, total_num)
    RR2 = eliminate_abnormal_peaks(indices_1[0], PPG_nose_1, rate1, total_num)
    print('RR1 = ', RR1)
    print('RR2 = ', RR2)
  


if __name__ == "__main__":
    args = parse_args()
    main(args)
