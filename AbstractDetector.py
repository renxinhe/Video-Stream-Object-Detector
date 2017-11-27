import os
import cv2
import numpy as np

from time import time, localtime, mktime, strftime

class AbstractDetector(object):
    FPS_CAP = 60

    def __init__(self, architechture, stream_url=None):
        self.architechture = architechture
        self.stream_url = stream_url
        self.cap = None
        # raise NotImplementedError('Abstract detector cannot be instantiated.')

    def setStreamURL(self, stream_url):
        self.stream_url = stream_url

    """
        Draws object bounding boxes on FRAME.

        This function intends to modify the original FRAME destructively.
    """
    def drawBoundingBox(self, frame):
        return frame

    def checkCapture(self):
        if not hasattr(self, 'cap') or self.cap is None or not self.cap.isOpened():
            opened = self.openCapture()
            if not opened:
                raise ValueError('Stream cannot be opened.')

    """
        Open a cv2 capture object.

        :param stream_url: A String representing the stream URL. It can be a file path or .m3u8 URL for live streams.
        :return: True if opening capture was successful, false otherwise.
        :rtype: Boolean
    """
    def openCapture(self, stream_url=None):
        if stream_url is not None:
            self.stream_url = stream_url
        if self.stream_url is None:
            raise ValueError("Stream URL isn't defined. Call 'self.setStreamURL(stream_url)'.")

        self.cap = cv2.VideoCapture(self.stream_url)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return self.cap.isOpened()

    def displayAnnotatedFrames(self):
        self.checkCapture()

        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if fps >= AbstractDetector.FPS_CAP:
            fps = AbstractDetector.FPS_CAP
        print 'Displaying with FPS = %d' % fps
        try:
            print 'Press Q to stop video stream.'
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if frame is None:
                    break

                # Process frame here
                self.drawBoundingBox(frame)
                
                cv2.imshow('%s Video Stream' % self.architechture, frame)
                if cv2.waitKey(1000 / fps) & 0xFF == ord('q'):
                    self.cap.release()
                    cv2.destroyAllWindows()
                    break
        except KeyboardInterrupt:
            self.cap.release()
            cv2.destroyAllWindows()

    def saveAnnotatedFrames(self, filename, root_path='./', segment_length=None, dir_size_limit=None):
        self.checkCapture()

        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if fps >= AbstractDetector.FPS_CAP:
            fps = AbstractDetector.FPS_CAP
        assert fps > 0, "FPS can't be negative."

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'X264')

        filepath = os.path.join(root_path, filename)
        timestamp = localtime()
        timestr = strftime('%Y-%m-%d_%H-%M-%S', timestamp)
        out = cv2.VideoWriter('%s_%s_%s.mp4' % (filepath, self.architechture, timestr), fourcc, fps, (self.width, self.height))
        
        dir_size_maxed = False
        try:
            while self.cap.isOpened():
                if segment_length is not None:
                    if time() - mktime(timestamp) > segment_length:
                        timestamp = localtime()
                        timestr = strftime('%Y-%m-%d_%H-%M-%S', timestamp)
                        out.open('%s_%s_%s.mp4' % (filepath, self.architechture, timestr), fourcc, fps, (self.width, self.height))

                ret, frame = self.cap.read()
                if frame is None:
                    break

                # Process frame here
                self.drawBoundingBox(frame)
                out.write(frame)

                if dir_size_limit is not None:
                    dir_size = sum(os.path.getsize(os.path.join(root_path, f)) for f in os.listdir(root_path) if os.path.isfile(os.path.join(root_path, f)))
                    if dir_size > dir_size_limit:
                        dir_size_maxed = True
                        break

            out.release()
            self.cap.release()
            return dir_size_maxed
        except KeyboardInterrupt:
            out.release()
            self.cap.release()
            return dir_size_maxed
