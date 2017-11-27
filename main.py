#/home/jren/env/bin/python
from __future__ import unicode_literals

import sys, os
import cv2
import youtube_dl
import numpy as np
from urlparse import parse_qs

from AbstractDetector import AbstractDetector
from SSDDetector import SSD_VGG16Detector
from TensorflowDetector import TensorflowDetector

youtube_ids = {
    'alberta_cam': 'P75NIeJPV3I'
}

streams = {
    'rush_hour': '../uds_video_demo/rush_hour.mp4',
    'alberta_cam_day_demo': '../uds_video_demo/alberta_cam_original_2017-10-27_10-00-30.mp4',
    'alberta_cam_night_demo': '../uds_video_demo/alberta_cam_original_2017-10-26_21-00-39.mp4',
}

class M3U8Logger(object):

    def debug(self, msg):
        self.url = msg

    def warning(self, msg):
        print('[WARNING]' + msg)

    def error(self, msg):
        print('[ERROR]' + msg)

"""
    Format parameter guide:
    format code  extension  resolution note
    92           mp4        240p       HLS , h264, aac  @ 48k
    93           mp4        360p       HLS , h264, aac  @128k
    94           mp4        480p       HLS , h264, aac  @128k
    95           mp4        720p       HLS , h264, aac  @256k
    96           mp4        1080p      HLS , h264, aac  @256k (best)
"""
def getYoutubeStreamURL(stream_name, youtube_video_id):
    m3u8 = M3U8Logger()
    ydl_opts = {
        'format': '95',
        'simulate': True,
        'forceurl': True,
        'quiet': True,
        'logger': m3u8,
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download(['https://www.youtube.com/watch?v=%s' % youtube_video_id])
        streams[stream_name] = m3u8.url

def saveAlbertaCam():
    ssd_detector.saveAnnotatedFrames('alberta_cam_night_demo', root_path='../uds_video_demo/annotated')

    no_detector = AbstractDetector('original', streams['alberta_cam'])
    getYoutubeStreamURL('alberta_cam', youtube_ids['alberta_cam'])
    dir_size_maxed = False
    while not dir_size_maxed:
        getYoutubeStreamURL('alberta_cam', youtube_ids['alberta_cam'])
        no_detector.setStreamURL(streams['alberta_cam'])
        dir_size_maxed = no_detector.saveAnnotatedFrames('alberta_cam', 
                                                         root_path='/nfs/diskstation/jren/alberta_cam/',
                                                         segment_length=60,
                                                         dir_size_limit=1e12)

def saveFramesWithCar():

    def scanFramesForCar(video_name, output_dir, output_limit):
        ssd_detector.setStreamURL(os.path.join(VIDEO_ROOT_DIR, video_name))
        if not AbstractDetector.openCapture(ssd_detector):
            raise ValueError('Video file %s failed to open.' % video_name)
        print 'Scanning %s...' % (video_name)

        output_count = 0
        while ssd_detector.cap.isOpened():
            ret, frame = ssd_detector.cap.read()
            if frame is None:
                break

            # Process frame here
            rclasses, rscores, rbboxes = ssd_detector.process_image(frame)
            unique, counts = np.unique(rclasses, return_counts=True)
            classes_counts = dict(zip(unique, counts))
            car_count = classes_counts.get(7)
            if car_count is not None and car_count > 0:
                image_name = '%s_%d.png' % (video_name[:-4], output_count)
                cv2.imwrite(os.path.join(output_dir, image_name), frame)
                output_count += 1
            if output_count >= output_limit:
                return

        print 'Writen %d images.' % (output_count)
        ssd_detector.cap.release()

    def outputLimitReached(output_dir, output_limit=1000):
        count = len([name for name in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, name))])
        return count >= output_limit

    ssd_detector = SSD_VGG16Detector('ssd_vgg16', 'VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt')
    VIDEO_ROOT_DIR = '/nfs/diskstation/jren/alberta_cam'
    OUTPUT_DAY_DIR = '/nfs/diskstation/jren/alberta_cam_frames_with_car/day'
    OUTPUT_NIGHT_DIR = '/nfs/diskstation/jren/alberta_cam_frames_with_car/night'
    OUTPUT_LIMIT = 1000
    day_limit_reached = False
    night_limit_reached = False

    for video_name in os.listdir(VIDEO_ROOT_DIR):
        day_limit_reached = True if day_limit_reached else outputLimitReached(OUTPUT_DAY_DIR, OUTPUT_LIMIT)
        night_limit_reached = True if night_limit_reached else outputLimitReached(OUTPUT_NIGHT_DIR, OUTPUT_LIMIT)

        if day_limit_reached and night_limit_reached:
            return

        video_name_splitted = [str(x.strip()) for x in video_name[:-4].split("_")]
        video_timestamp = [int(x.strip()) for x in video_name_splitted[-1].split("-")]
        
        if not day_limit_reached and video_timestamp[0] > 9 and video_timestamp[0] < 17:
            day_limit_reached = scanFramesForCar(video_name, OUTPUT_DAY_DIR, OUTPUT_LIMIT)
        elif not night_limit_reached and (video_timestamp[0] > 21 or video_timestamp[0] < 5):
            night_limit_reached = scanFramesForCar(video_name, OUTPUT_NIGHT_DIR, OUTPUT_LIMIT)

def main():
    # ssd_detector = SSD_VGG16Detector('ssd_vgg16', 'VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt', streams['alberta_cam_night_demo'])
    # tf_detector = TensorflowDetector('ssd_mobilenet_v1', 'ssd_mobilenet_v1_coco_11_06_2017', streams['rush_hour'])
    tf_detector = TensorflowDetector('ssd_inception_v2', 'ssd_inception_v2_coco_11_06_2017', streams['rush_hour'])
    # tf_detector.displayAnnotatedFrames()

    tf_detector.saveAnnotatedFrames('rush_hour', root_path='../uds_video_demo/annotated')

    # img = cv2.imread('../uds_video_demo/alberta_nobox.png', cv2.IMREAD_COLOR)
    # ssd_detector = SSD_VGG16Detector('ssd_vgg16', 'VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt')
    # print ssd_detector.process_image(img)

if __name__ == "__main__":
    # main()
    saveFramesWithCar()