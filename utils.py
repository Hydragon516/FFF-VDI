import cv2
import random
import numpy as np
import math
from PIL import Image, ImageOps

import torch
import matplotlib
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib import pyplot as plt
matplotlib.use('agg')


# ###########################################################################
# ###########################################################################

def find_square_mask(input_mask):
    input_mask = input_mask.convert('L')
    width, height = input_mask.size
    input_mask = np.array(input_mask)

    if np.max(input_mask) == 0:
        return input_mask

    # check horizontal min and max
    for i in range(width):
        if np.max(input_mask[:, i]) > 0:
            x_min = i
            break
    for i in range(width-1, -1, -1):
        if np.max(input_mask[:, i]) > 0:
            x_max = i
            break

    # check vertical min and max
    for i in range(height):
        if np.max(input_mask[i, :]) > 0:
            y_min = i
            break
    for i in range(height-1, -1, -1):
        if np.max(input_mask[i, :]) > 0:
            y_max = i
            break
    
    square_mask = np.zeros((height, width))
    square_mask[y_min:y_max, x_min:x_max] = 255
    square_mask = Image.fromarray(square_mask.astype(np.uint8))
    square_mask = square_mask.convert('RGB')
    return square_mask
    

class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    # invert flow pixel values when flipping
                    ret[i] = ImageOps.invert(ret[i])
            return ret
        else:
            return img_group


class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        mode = img_group[0].mode
        if mode == '1':
            img_group = [img.convert('L') for img in img_group]
            mode = 'L'
        if mode == 'L':
            return np.stack([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif mode == 'RGB':
            if self.roll:
                return np.stack([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.stack(img_group, axis=2)
        else:
            raise NotImplementedError(f"Image mode {mode}")


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # numpy img: [L, C, H, W]
            img = torch.from_numpy(pic).permute(2, 3, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255) if self.div else img.float()
        return img


# ##########################################
# ##########################################
def create_random_square_with_random_motion(video_length, imageHeight=240, imageWidth=432):
    min_h = int(imageHeight * 0.2)
    min_w = int(imageWidth * 0.2)
    max_h = int(imageHeight * 0.8)
    max_w = int(imageWidth * 0.8)

    base_h = random.randint(min_h, max_h)
    base_w = random.randint(min_w, max_w)

    weight_h = random.uniform(-0.1, 0.1)
    weight_w = random.uniform(-0.1, 0.1)

    x, y = random.randint(0, imageHeight - max_h), random.randint(0, imageWidth - max_w)
    velocity = get_random_velocity(max_speed=3)

    masks = []
    for t in range(video_length):

        mask_h = int(base_h + base_h * weight_h * (1 + math.sin(2 * math.pi * t / video_length)) / 2)
        mask_w = int(base_w + base_w * weight_w * (1 + math.sin(2 * math.pi * t / video_length)) / 2)

        x, y, velocity = random_move_control_points(x, y, imageHeight, imageWidth, velocity, (mask_h, mask_w), maxLineAcceleration=(3, 0.5), maxInitSpeed=3)

        square = Image.fromarray(np.zeros((imageHeight, imageWidth)).astype(np.uint8))
        square.paste(Image.fromarray(np.ones((mask_h, mask_w)).astype(np.uint8) * 255), (y, x))
        masks.append(square.convert('L'))
    return masks


def create_random_shape_with_random_motion(video_length, imageHeight=240, imageWidth=432):
    # get a random shape
    height = random.randint(imageHeight//3, imageHeight-1)
    width = random.randint(imageWidth//3, imageWidth-1)
    edge_num = random.randint(6, 8)
    ratio = random.randint(6, 8)/10
    region = get_random_shape(edge_num=edge_num, ratio=ratio, height=height, width=width)
    # get random position
    x, y = random.randint(0, imageHeight), random.randint(0, imageWidth)
    velocity = get_random_velocity(max_speed=3)
    m = Image.fromarray(np.zeros((imageHeight, imageWidth)).astype(np.uint8))
    m.paste(region, (y, x, y+region.size[0], x+region.size[1]))
    masks = [m.convert('L')]
    # return fixed masks
    if random.uniform(0, 1) > 0.5:
        return masks*video_length
    # return moving masks
    for _ in range(video_length-1):
        x, y, velocity = random_move_control_points(x, y, imageHeight, imageWidth, velocity, region.size, maxLineAcceleration=(3, 0.5), maxInitSpeed=3)
        m = Image.fromarray(np.zeros((imageHeight, imageWidth)).astype(np.uint8))
        m.paste(region, (y, x, y+region.size[0], x+region.size[1]))
        masks.append(m.convert('L'))
    return masks


def get_random_shape(edge_num=9, ratio=0.7, width=432, height=240):
    '''
      There is the initial point and 3 points per cubic bezier curve. 
      Thus, the curve will only pass though n points, which will be the sharp edges.
      The other 2 modify the shape of the bezier curve.
      edge_num, Number of possibly sharp edges
      points_num, number of points in the Path
      ratio, (0, 1) magnitude of the perturbation from the unit circle, 
    '''
    points_num = edge_num*3 + 1
    angles = np.linspace(0, 2*np.pi, points_num)
    codes = np.full(points_num, Path.CURVE4)
    codes[0] = Path.MOVETO
    # Using this instad of Path.CLOSEPOLY avoids an innecessary straight line
    verts = np.stack((np.cos(angles), np.sin(angles))).T * \
        (2*ratio*np.random.random(points_num)+1-ratio)[:, None]
    verts[-1, :] = verts[0, :]
    path = Path(verts, codes)
    # draw paths into images
    fig = plt.figure()
    ax = fig.add_subplot(111)
    patch = patches.PathPatch(path, facecolor='black', lw=2)
    ax.add_patch(patch)
    ax.set_xlim(np.min(verts)*1.1, np.max(verts)*1.1)
    ax.set_ylim(np.min(verts)*1.1, np.max(verts)*1.1)
    ax.axis('off')  # removes the axis to leave only the shape
    fig.canvas.draw()
    # convert plt images into numpy images
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape((fig.canvas.get_width_height()[::-1] + (3,)))
    plt.close(fig)
    # postprocess
    data = cv2.resize(data, (width, height))[:, :, 0]
    data = (1 - np.array(data > 0).astype(np.uint8))*255
    corrdinates = np.where(data > 0)
    xmin, xmax, ymin, ymax = np.min(corrdinates[0]), np.max(corrdinates[0]), np.min(corrdinates[1]), np.max(corrdinates[1])
    region = Image.fromarray(data).crop((ymin, xmin, ymax, xmax))
    return region


def random_accelerate(velocity, maxAcceleration, dist='uniform'):
    speed, angle = velocity
    d_speed, d_angle = maxAcceleration
    if dist == 'uniform':
        speed += np.random.uniform(-d_speed, d_speed)
        angle += np.random.uniform(-d_angle, d_angle)
    elif dist == 'guassian':
        speed += np.random.normal(0, d_speed / 2)
        angle += np.random.normal(0, d_angle / 2)
    else:
        raise NotImplementedError(
            f'Distribution type {dist} is not supported.')
    return (speed, angle)


def get_random_velocity(max_speed=3, dist='uniform'):
    if dist == 'uniform':
        speed = np.random.uniform(max_speed)
    elif dist == 'guassian':
        speed = np.abs(np.random.normal(0, max_speed / 2))
    else:
        raise NotImplementedError(
            f'Distribution type {dist} is not supported.')
    angle = np.random.uniform(0, 2 * np.pi)
    return (speed, angle)


def random_move_control_points(X, Y, imageHeight, imageWidth, lineVelocity, region_size, maxLineAcceleration=(3, 0.5), maxInitSpeed=3):
    region_width, region_height = region_size
    speed, angle = lineVelocity
    X += int(speed * np.cos(angle))
    Y += int(speed * np.sin(angle))
    lineVelocity = random_accelerate(
        lineVelocity, maxLineAcceleration, dist='guassian')
    if ((X > imageHeight - region_height) or (X < 0) or (Y > imageWidth - region_width) or (Y < 0)):
        lineVelocity = get_random_velocity(maxInitSpeed, dist='guassian')
    new_X = np.clip(X, 0, imageHeight - region_height)
    new_Y = np.clip(Y, 0, imageWidth - region_width)
    return new_X, new_Y, lineVelocity



# ##############################################
# ##############################################

if __name__ == '__main__':

    trials = 10
    for _ in range(trials):
        video_length = 10
        # The returned masks are either stationary (50%) or moving (50%)
        masks = create_random_shape_with_random_motion(
            video_length, imageHeight=240, imageWidth=432)

        for m in masks:
            cv2.imshow('mask', np.array(m))
            cv2.waitKey(500)