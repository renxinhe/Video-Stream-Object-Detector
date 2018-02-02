import os
import sys
import math
import random
import csv
import pickle
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

matplotlib.rcParams['keymap.back'].remove('left')
matplotlib.rcParams['keymap.forward'].remove('right')

# DATA_ROOT_DIR = '/nfs/diskstation/jren/alberta_cam_with_data/'
# DATA_DIRS = sorted(os.listdir(DATA_ROOT_DIR))
# print 'First directory is "%s"' % DATA_DIRS[0]

# DATA_DIR = os.path.join(DATA_ROOT_DIR, DATA_DIRS[0])
# img_nums = sorted([int(img_path.lstrip('raw_').rstrip('.jpg')) for img_path in os.listdir(DATA_DIR + '/imgs') if 'raw' in img_path])
# print '%d frames' % len(img_nums)

# trajectories_path = os.path.join(DATA_DIR , 'trajectories.pkl')

img_size = (1280, 720) #(width, height)
img_nums = None
trajectories = None
trajectory_label = 0
frame_i = 0
quit_loop = False

"""
Trajectories is a list of size LEN(IMG_NUMS), or the number of frames present in DATA_DIR.
Each element of TRAJECTORIES is a list of tuples (x, y, detection_class, trajectory_label).
Each tuple from above represents one bounding box in the frame.
"""
def load_trajectories(trajectories_path):
    try:
        trajectories = pickle.load(open(trajectories_path, 'r'))
        print '"%s" loaded.' % trajectories_path
    except IOError:
        print 'No "%s" found. Loading empty trajectory.' % trajectories_path
        trajectories = np.empty((len(img_nums), 0)).tolist()
    return trajectories

"""
AX is a matplotlib axis object.
BBOX is an array like object: [xmin, ymin, xmax, ymax].
IM_SIZE is image (width, height).
"""
def draw_bbox(ax, bbox, label, bbox_traj, im_size, color=None, picker=True):
    width = (bbox[2] - bbox[0]) * im_size[0]
    height = (bbox[3] - bbox[1]) * im_size[1]
    
    if color is None:
        color = 'gray'
        linestyle = 'dashed'
    else:
        linestyle = 'solid'
    # patches.Rectangle(top_left_xy, width, height)
    xmin = bbox[0] * im_size[0]
    ymin = bbox[1] * im_size[1]
    rect = patches.Rectangle((xmin, ymin), width, height,
                             linewidth=1.5, linestyle=linestyle, label=label,
                             edgecolor=color, facecolor='none', picker=picker)
    ax.add_patch(rect)
    if bbox_traj is not None and color is not None:
        ax.text(xmin, ymin - 5, '#%s' % str(bbox_traj), color=color, backgroundcolor=(1, 1, 1, 0.5), fontweight='bold')
    
def YX_to_XY(bbox):
    return (bbox[1], bbox[0], bbox[3], bbox[2])

def get_midpoint(xmin, ymin, xmax, ymax):
    mid_x = (xmin + (xmax - xmin) / 2)
    mid_y = (ymin + (ymax - ymin) / 2)
    return mid_x, mid_y

def on_pick(event):
    global img_size
    global trajectories
    global trajectory_label
    global frame_i

    this = event.artist
    bbox = this.get_bbox()
    mid_x, mid_y = get_midpoint(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
    mid_x /= img_size[0]
    mid_y /= img_size[1]

    trajectories[frame_i].append((mid_x, mid_y, str(this._label), trajectory_label))
    
def on_key_press(event):
    global img_size
    global img_nums
    global trajectories
    global trajectory_label
    global frame_i
    global quit_loop
    
    if event.key == 'q':
        quit_loop = True
    elif event.key == 'down':
        trajectory_label -= 1
        trajectory_label = max(0, trajectory_label)
    elif event.key == 'up':
        trajectory_label += 1
    elif event.key == 'left':
        frame_i -= 1
        frame_i = max(0, frame_i)
    elif event.key == 'right':
        frame_i += 1
        frame_i = min(len(img_nums), frame_i)
    elif event.key == '.':
        frame_i += 10
        frame_i = min(len(img_nums), frame_i)
    elif event.key == ',':
        frame_i -= 10
        frame_i = max(0, frame_i)
    elif event.key == ']':
        frame_i += 100
        frame_i = min(len(img_nums), frame_i)
    elif event.key == '[':
        frame_i -= 100
        frame_i = max(0, frame_i)
    elif event.key == 'w':
        pickle.dump(trajectories, open(trajectories_path, 'w+'))
        print 'Written trajectories to %s.' % trajectories_path
        
    frame_i -= 1


instruction_str = """
Instructions
To label new trajectories, run the next code block. This script uses matplotlib and Qt for window rendering.
Gray dashed rectangles mark unlabeled bounding boxes. Click on a dashed bounding box to label it as `trajectory_label` and move on to the next frame.

Key Controls
 - Left Mouse Click: Annotate a bounding box and go to next frame. If empty area is clicked, then no bounding box is annotated.
 - Q: Quit.
 - W: Save trajectory to pickle file.
 - Right Arrow: Next frame.
 - Left Arrow: Previous frame.
 - Up Arrow: Next trajectory.
 - Down Arrow: Previous trajectory.
 - .: Next 10 frames.
 - ,: Previous 10 frames.
 - ]: Next 100 frames.
 - [: Previous 100 frames.
"""

def main(data_dir):
    global img_size
    global img_nums
    global trajectories
    global trajectory_label
    global frame_i
    global quit_loop

    img_nums = sorted([int(img_path.lstrip('raw_').rstrip('.jpg')) for img_path in os.listdir(data_dir + '/imgs') if 'raw' in img_path])
    print '%d frames detected' % len(img_nums)

    trajectories_path = os.path.join(data_dir , 'trajectories.pkl')

    plt.ioff()

    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.set_aspect('auto')
    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    plt.tight_layout()
    fig.show()

    trajectories = load_trajectories(trajectories_path)
    while not quit_loop:
        img_num = img_nums[frame_i]
        img = mpimg.imread(data_dir + '/imgs/raw_%d.jpg' % img_num)
        height, width, channel = img.shape
        img_size = (width, height)
        bboxes = np.load(data_dir + '/bboxes/bboxes_%d.npy' % img_num)
        classes = np.load(data_dir + '/classes/classes_%d.npy' % img_num)
        
        ax.imshow(img)
        ax.set_title('Trajectory #%d Frame: %d/%d' % (trajectory_label, frame_i, len(img_nums) - 1))
        plt.draw()
        
        for i, bbox in enumerate(bboxes):
            bbox = YX_to_XY(bbox)
            midpoint = get_midpoint(*bbox)
            color = None
            bbox_traj = None
            for tmp_x, tmp_y, tmp_class, tmp_traj in trajectories[frame_i]:
                if np.allclose(midpoint, (tmp_x, tmp_y)):
                    color = 'C%d' % (int(tmp_traj) % 10)
                    bbox_traj = tmp_traj
            draw_bbox(ax, bbox, classes[i], bbox_traj, img_size, color=color, picker=(color is None))
        
        fig.waitforbuttonpress(timeout=-1)
        ax.clear()
        plt.draw()
        frame_i += 1
        if frame_i >= len(img_nums):
            pickle.dump(trajectories, open(trajectories_path, 'w+'))
            print 'Wrote trajectory %d to %s' % (trajectory_label, trajectories_path)
            trajectory_label += 1
            frame_i = 0
        if quit_loop:
            break
    ax.clear()
    plt.close(fig)
    # print trajectories
    # pickle.dump(trajectories, open(trajectories_path, 'w+'))


# In[8]:


# trajectories = load_trajectories(trajectories_path)
# for i, trajectories_in_frame in enumerate(trajectories):
#     if len(trajectories_in_frame) >= 2:
#         bbox_tuple = trajectories[i].pop()
#         bbox_tuple = (bbox_tuple[0], bbox_tuple[1], bbox_tuple[2], 2)
#         trajectories[i].append(bbox_tuple)
# pickle.dump(trajectories, open(trajectories_path, 'w+'))

if __name__ == "__main__":
    # DATA_DIR = '/nfs/diskstation/jren/alberta_cam_with_data/alberta_cam_original_2017-10-26_15-24-08'
    if len(sys.argv) == 2:
        print instruction_str
        main(sys.argv[1])
    else:
        print 'usage: frame_based_trajectory_annotator.py data_dir'
