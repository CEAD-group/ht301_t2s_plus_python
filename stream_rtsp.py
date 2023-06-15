# %%
import cv2
import subprocess as sp
from pathlib import Path
import time

import numpy as np
import cv2
import math
import ht301_hacklib
import utils
import time
from skimage.exposure import rescale_intensity, equalize_hist
draw_temp = True

cap = ht301_hacklib.HT301()
# cv2.namedWindow("HT301", cv2.WINDOW_NORMAL)


def start_ffmpeg_process(im):
    width, height = im.shape[1], im.shape[0]
    sizeStr = str(width) + "x" + str(height)
    rtsp_server = "rtsp://0.0.0.0:8554/thermal"  # push server (output server)
    fps = 25
    command = [
        "ffmpeg",
        "-re",
        "-s",
        sizeStr,
        "-r",
        str(fps),  # rtsp fps (from input server)
        "-i",
        "-",
        # You can change ffmpeg parameter after this item.
        "-pix_fmt",  # pixel format, options are yuv420p, yuv422p, yuv444p, yuvj420p, yuvj422p, yuvj444p, gbrp, gbrap, gray, monow, monob
        "yuv420p", #works with sinumerik
        "-r",
        str(fps),  # output fps
        "-g",  # key frame interval
        "10",
        "-c:v",  # codec
        "libx264",
        "-b:v",  # bitrate
        "16M",
        "-bufsize",  # buffer size
        "0M",
        "-maxrate",  # max bitrate
        "32M",
        "-preset",  # preset
        "ultrafast",
        "-rtsp_transport",  # transport
        "udp",
        # "-filter:v",  # swap red and blue channels
        # "colorchannelmixer=0:0:1:0:0:1:0:0:1:0:0",
        "-segment_times",  # segment time
        "1",
        "-f",
        "rtsp",
        rtsp_server,
    ]

    process = sp.Popen(command, stdin=sp.PIPE)
    return process


def increase_luminance_contrast(frame):
    lab= cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))
    frame = enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return frame

def get_frame(cap, draw_temp=True):

    ret, frame = cap.read()

    info, lut = cap.info()
    frame = frame.astype(np.float32)

    # Sketchy auto-exposure
    frame = rescale_intensity(equalize_hist(frame), in_range='image', out_range=(0,255)).astype(np.uint8)
    frame = cv2.applyColorMap(frame, cv2.COLORMAP_INFERNO)
    
    frame = increase_luminance_contrast(frame)

    if draw_temp:
        # utils.drawTemperature(frame, info['Tmin_point'], info['Tmin_C'], (55,0,0))
        utils.drawTemperature(frame, info['Tmax_point'], info['Tmax_C'], (255,255,255))
        utils.drawTemperature(frame, info['Tcenter_point'], info['Tcenter_C'], (255,255,255))

    return frame

if __name__=="__main__":
    frame = get_frame(cap, draw_temp=True)

    ffmpeg_process = start_ffmpeg_process(frame)
    cv2.namedWindow("HT301", cv2.WINDOW_NORMAL)
    while(True):
        
        frame = get_frame(cap, draw_temp=True)

        #frame2 = frame2.reshape(288, 384)
        # cv2.imshow('HT301',frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('u'):
            cap.calibrate()
        if key == ord('s'):
            cv2.imwrite(time.strftime("%Y-%m-%d_%H:%M:%S") + '.png', frame)

        ret2, frame2 = cv2.imencode(".png", frame)
        ffmpeg_process.stdin.write(frame2.tobytes())
        cv2.imshow('HT301',frame)
    else:
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
    cap.release()
    cv2.destroyAllWindows()

