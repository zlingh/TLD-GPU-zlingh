this code project is exploited based on arthurv's TLD code

our work is:
1. we parallel implement TLD using cuda. Mainly the detection part.
1. We get a vs2010 vision.
2. We write Ncc_CCORR_NORMED by ourself instead of using the opencv code, and then improve the  real-time performance greatly.
3. we use a higher version of OpenCV(2.4.1), and some code of PatchGenerator is not surpported any more by OpenCV, so we import the 
patchGenerator.h and patchGenerator.cpp file.
====================================
Thanks
====================================
To arthurv for his C++ implementation of TLD
To Zdenek Kalal for realeasing his awesome algorithm

track_video.exe为跟踪视频的程序
car.mpg为跟踪视频
car.txt为跟着视频中目标的初始位置
若未给定car.txt，则在程序中可以使用鼠标自己选定目标

track_image.exe为跟踪图像片序列的程序
data是图像片序列的文件夹，其中init.txt中设定目标初始位置，
若未给定init.txt则在程序中可以使用鼠标自己选定目标

程序运行过程中控制键
q:退出
r:重置目标框框
p:不跟踪只是播放
x:显示跟踪的特征点