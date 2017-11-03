#/home/jren/env/bin/python
from __future__ import unicode_literals

import cv2
import youtube_dl
from urlparse import parse_qs

from AbstractDetector import AbstractDetector
from SSDDetector import SSDDetector

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

def main():
    # ssd_detector = SSDDetector('ssd', 'VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt', streams['alberta_cam_night_demo'])
    # ssd_detector.displayAnnotatedFrames()

    img = cv2.imread('../uds_video_demo/alberta_nobox.png', cv2.IMREAD_COLOR)
    ssd_detector = SSDDetector('ssd', 'VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt')
    print ssd_detector.process_image(img)

if __name__ == "__main__":
    main()