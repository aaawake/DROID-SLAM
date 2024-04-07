#coding=utf-8
import cv2
import os

path = '/home/dongjialin/data/droidSLAM/pipe_video'
video_list = os.listdir(path)
for video in video_list:
    video_path = os.path.join(path,video)
    vc=cv2.VideoCapture(video_path)
    if vc.isOpened():
        rval,frame=vc.read()
    else:
        rval=False

    c =1
    f = open("./data/steel_pipe/rgb.txt", 'a+')
    while rval:
        rval,frame=vc.read()
        if rval==False:
            break
        img_dir = 'rgb/'+str(video)[:-4]+'_'+str(c).zfill(5)+'.jpg'
        cv2.imwrite('./data/steel_pipe/'+img_dir,frame)
        f.writelines(str(c).zfill(5)+' '+img_dir+'\n')
        # print(c)
        c=c+1
        cv2.waitKey(1)
    vc.release()
