from __future__ import unicode_literals

import youtube_dl
from urlparse import parse_qs

from SSDDetector import SSDDetector

streams = {
    'rush_hour': '../uds_video_demo/rush_hour.mp4'
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

def main():
    getYoutubeStreamURL('alberta_cam', 'P75NIeJPV3I')
    ssd_detector = SSDDetector('ssd', 'VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt', streams['alberta_cam'])
    ssd_detector.displayAnnotatedFrames()
    # ssd_detector.saveAnnotatedFrames('output/rush_hour')

if __name__ == "__main__":
    main()