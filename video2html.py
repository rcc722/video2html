# -*- coding:utf-8 -*-

import os
import pickle
from threading import Thread

import cv2
import invoke
from PIL import Image

# 用于生成字符画的像素，越往后视觉上越明显。。这是我自己按感觉排的，你可以随意调整。写函数里效率太低，所以只好放全局了
pixels = " .,-'`:!1+*abcdefghijklmnopqrstuvwxyz<>()\/{}[]?234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ%&@#$"


def video2imgs(video_name, size, seconds):
    """

    :param video_name: 字符串, 视频文件的路径
    :param size: 二元组，(宽, 高)，用于指定生成的字符画的尺寸
    :param seconds: 指定需要解码的时长（0-seconds）
    :return: 一个 img 对象的列表，img对象实际上就是 numpy.ndarray 数组
    """
    img_list = []

    # 从指定文件创建一个VideoCapture对象
    cap = cv2.VideoCapture(video_name)

    # 帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 需要提取的帧数
    frames_count = fps * seconds

    count = 0
    # cap.isOpened(): 如果cap对象已经初始化完成了，就返回true
    while cap.isOpened() and count < frames_count:
        # cap.read() 返回值介绍：
        #   ret 表示是否读取到图像
        #   frame 为图像矩阵，类型为 numpy.ndarry.
        ret, frame = cap.read()
        if ret:
            # 转换成灰度图，也可不做这一步，转换成彩色字符视频。
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # resize 图片，保证图片转换成字符画后，能完整地在命令行中显示。
            img = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

            # 分帧保存转换结果
            img_list.append(img)

            count += 1
        else:
            break

    # 结束时要释放空间
    cap.release()

    return img_list, fps


def get_char(r, g, b, alpha=256):
    """
    rbg 转字符
    :param r:
    :param g:
    :param b:
    :param alpha:
    :return:
    """
    if alpha == 0:
        return ' '
    gary = (2126 * r + 7152 * g + 722 * b) / 10000
    ascii_char = list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")
    # gary / 256 = x / len(ascii_char)
    x = int(gary / (alpha + 1.0) * len(ascii_char))
    return ascii_char[x]

def img2chars(img):
    text=''
    """

    :param img: numpy.ndarray, 图像矩阵
    :return: 字符串的列表：图像对应的字符画，其每一行对应图像的一行像素
    """

    # 要注意这里的顺序和 之前的 size 刚好相反
    im = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    height, width = img.shape
    for j in range(height):
        for i in range(width):
            # 灰度是用8位表示的，最大值为255。
            text += get_char(*im.getpixel((i, j)))
        text+='<br>'
    return text

def imgs2chars(imgs):
    video_chars = ''
    for img in imgs:
        video_chars+=img2chars(img)

    return video_chars

def get_file_name(file_path):
    """
    从文件路径中提取出不带拓展名的文件名
    """
    # 从文件路径获取文件名 _name
    path, file_name_with_extension = os.path.split(file_path)

    # 拿到文件名前缀
    file_name, file_extension = os.path.splitext(file_name_with_extension)

    return file_name


def get_video_chars(video_path, size, seconds):
    print("开始逐帧读取")
    # 视频转字符动画
    imgs, fps = video2imgs(video_path, size, seconds)

    print("视频已全部转换到图像， 开始逐帧转换为字符画")
    video_chars = imgs2chars(imgs)

    print("转换完成")
    print(video_chars)

    return video_chars, fps


def main():
    # 宽，高
    size = (64, 48)
    # 视频路径，换成你自己的
    video_path = "桃园恋歌.mp4"
    seconds = 2  # 只转换三十秒
    video_chars, fps = get_video_chars(video_path, size, seconds)




if __name__ == "__main__":
    main()