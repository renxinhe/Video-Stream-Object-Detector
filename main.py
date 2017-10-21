from SSDDetector import SSDDetector

def main():
	stream_url = 'https://manifest.googlevideo.com/api/manifest/hls_playlist/id/P75NIeJPV3I.1/itag/95/source/yt_live_broadcast/requiressl/yes/ratebypass/yes/live/1/cmbypass/yes/goi/160/sgoap/gir%3Dyes%3Bitag%3D140/sgovp/gir%3Dyes%3Bitag%3D136/hls_chunk_host/r5---sn-n4v7snee.googlevideo.com/ei/G03qWbnSB8er_AOK4I2YCg/playlist_type/DVR/gcr/us/mm/32/mn/sn-n4v7snee/ms/lv/mv/m/pl/18/dover/6/mt/1508527276/ip/128.32.192.73/ipbits/0/expire/1508548986/sparams/ip,ipbits,expire,id,itag,source,requiressl,ratebypass,live,cmbypass,goi,sgoap,sgovp,hls_chunk_host,ei,playlist_type,gcr,mm,mn,ms,mv,pl/signature/6CD20946F9F5D814D29257BA9B1D659EAF34795C.96EDB7E8458F69B2E0C59CA7C98A517CA06C8F63/key/dg_yt0/playlist/index.m3u8'
	ssd_detector = SSDDetector('ssd', 'VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt', stream_url)
	ssd_detector.displayAnnotatedFrames()

if __name__ == "__main__":
	main()