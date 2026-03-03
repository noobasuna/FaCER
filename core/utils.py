import torch
import os
import yaml
import copy
import numpy as np
import torch.nn as nn
from PIL import Image
from os import path as osp
import cv2
import dlib

import torch.nn.functional as F
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class fc_net(nn.Module):
    def __init__(self, out_dim,use_sigmoid):
        super(fc_net, self).__init__()
        # image: (3 x 224 x 224)
        # insize: (batch_size x 256 x 56 x 56)
        self.use_sigmoid = use_sigmoid
        
        self.fc = nn.Sequential(*[
                    nn.AdaptiveAvgPool2d((1, 1)),
                    Flatten(),
                    nn.Linear(512, out_dim, bias=use_bias)
                    ]) 
        
    def forward(self, x):
        x = self.fc(x)
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        return x
    
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
def restore_img_to_adv(img, x, norm_fn=lambda x: x * 0.5 + 0.5):
    img = norm_fn(img)
    img = img.astype('float32') * 255.0
    img = np.transpose(img, axes=(2, 0, 1))
    a,b,c = img.shape
    img = np.reshape(img, (1, a, b, c))
    return img

def restore_img(img, x, norm_fn=lambda x: (x - 0.5) / 0.5):
    a,b,c = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((1, a, b, c))
    img = img.astype('float32') / 255.0
    img = np.transpose(img, axes=(0, 3, 1, 2))
    img = norm_fn(img)
    return img

def presence_of_heavy_makeup(image, image2, idx, batch_size):
    # Load the face detector and shape predictor models
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/home/tpei0009/shape_predictor_68_face_landmarks.dat")
    image = np.array(image)
    image2 = np.array(image2)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector(gray)

    # Check if a face is detected, and if not, skip this detection
    if not faces:
        print('Debug: Nofaces.')
        img_npy = restore_img(image, batch_size)
        img_npy2 = restore_img(image2, batch_size)
        return image, img_npy, img_npy2

    # Initialize a single mask for both eyebrow and nose regions
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    face = faces[0]
    # Get the facial landmarks for the detected face
    landmarks = predictor(gray, face)

    # Extract the landmarks for the eyebrow region (17, 18, 19, 20, 21) and nose region (27, 28, 29, 30, 31, 32, 33, 34)
    eyebrow_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in [19, 17, 26, 24]]
    # nose_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in [27, 31, 33, 35]]
    left_eye_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
    right_eye_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

    # Create a mask for the eyebrow region
    mask_eyebrow = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_eyebrow, [np.array(eyebrow_landmarks)], (255, 255, 255))

    # Create a mask for the nose region
    mask_left_eye = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_left_eye, [np.array(left_eye_landmarks)], (255, 255, 255))
    mask_right_eye = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_right_eye, [np.array(right_eye_landmarks)], (255, 255, 255))
    # Merge the masks into the single mask variable using bitwise OR
    mask = cv2.bitwise_or(mask, mask_eyebrow)
    mask = cv2.bitwise_or(mask, mask_left_eye)
    mask = cv2.bitwise_or(mask, mask_right_eye)
    
    mask = cv2.bitwise_not(mask)
    
    # Apply the mask to the image
    image = cv2.bitwise_and(image, image, mask=mask)
    image2 = cv2.bitwise_and(image2, image2, mask=mask)

    # Restore images
    img_npy = restore_img(image, batch_size)
    img_npy2 = restore_img(image2, batch_size)
    return image, img_npy, img_npy2

def presence_of_gender(image, image2, idx, batch_size):
    # Load the face detector and shape predictor models
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/home/tpei0009/shape_predictor_68_face_landmarks.dat")
    image = np.array(image)
    image2 = np.array(image2)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector(gray)

    # Check if a face is detected, and if not, skip this detection
    if not faces:
        print('Debug: Nofaces.')
        img_npy = restore_img(image, batch_size)
        img_npy2 = restore_img(image2, batch_size)
        return image, img_npy, img_npy2

    # Initialize a single mask for both eyebrow and nose regions
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    face = faces[0]
    # Get the facial landmarks for the detected face
    landmarks = predictor(gray, face)

    # Extract the landmarks for the eyebrow region (17, 18, 19, 20, 21) and nose region (27, 28, 29, 30, 31, 32, 33, 34)
    eyebrow_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in [19, 17, 26, 24]]
    nose_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in [27, 31, 33, 35]]

    # Create a mask for the eyebrow region
    mask_eyebrow = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_eyebrow, [np.array(eyebrow_landmarks)], (255, 255, 255))

    # Create a mask for the nose region
    mask_nose = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_nose, [np.array(nose_landmarks)], (255, 255, 255))

    # Merge the masks into the single mask variable using bitwise OR
    mask = cv2.bitwise_or(mask, mask_eyebrow)
    mask = cv2.bitwise_or(mask, mask_nose)
    
    mask = cv2.bitwise_not(mask)
    
    # Apply the mask to the image
    image = cv2.bitwise_and(image, image, mask=mask)
    image2 = cv2.bitwise_and(image2, image2, mask=mask)

    # Restore images
    img_npy = restore_img(image, batch_size)
    img_npy2 = restore_img(image2, batch_size)
    return image, img_npy, img_npy2

def presence_of_narroweyes(image, image2, idx, batch_size):
    # Load the face detector and shape predictor models
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/home/tpei0009/shape_predictor_68_face_landmarks.dat")
    image = np.array(image)
    image2 = np.array(image2)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector(gray)

    # Check if a face is detected, and if not, skip this detection
    if not faces:
        print('Debug: Nofaces.')
        img_npy = restore_img(image, batch_size)
        img_npy2 = restore_img(image2, batch_size)
        return image, img_npy, img_npy2

    # Initialize a single mask for both eyebrow and nose regions
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    face = faces[0]
    # Get the facial landmarks for the detected face
    landmarks = predictor(gray, face)

    # Extract the landmarks for the eyebrow region (17, 18, 19, 20, 21) and nose region (27, 28, 29, 30, 31, 32, 33, 34)
    # Extract the landmarks for the mouth region
    mouth_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]

    # Create a mask for the mouth region
    mask_mouth = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_mouth, [np.array(mouth_landmarks)], (255, 255, 255))
    # Merge the masks into the single mask variable using bitwise OR
    mask = cv2.bitwise_or(mask, mask_mouth)
    mask = cv2.bitwise_not(mask)
    
    # Apply the mask to the image
    image = cv2.bitwise_and(image, image, mask=mask)
    image2 = cv2.bitwise_and(image2, image2, mask=mask)

    # Restore images
    img_npy = restore_img(image, batch_size)
    img_npy2 = restore_img(image2, batch_size)
    return image, img_npy, img_npy2
        
def presence_of_dchin(image, image2, idx, batch_size):
    # Load the face detector and shape predictor models
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/home/tpei0009/shape_predictor_68_face_landmarks.dat")
    image = np.array(image)
    image2 = np.array(image2)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector(gray)

    # Check if a face is detected, and if not, skip this detection
    if not faces:
        print('Debug: Nofaces.')
        img_npy = restore_img(image, batch_size)
        img_npy2 = restore_img(image2, batch_size)
        return image, img_npy, img_npy2

    # Initialize a single mask for both lower face and chin regions
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    face = faces[0]
    # Get the facial landmarks for the detected face
    landmarks = predictor(gray, face)

    # Extract the landmarks for the lower face (17, 1, 15, 26) and lower chin (7, 8, 9, 10, 11)
    lower_face_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in [17, 1, 15, 26]]
    lower_chin_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in [6, 7, 8, 9, 10, 11]]

    # Create a mask for the lower face region
    mask_lower_face = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_lower_face, [np.array(lower_face_landmarks)], (255, 255, 255))

    # Create a mask for the lower chin region
    mask_lower_chin = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_lower_chin, [np.array(lower_chin_landmarks)], (255, 255, 255))

    # Merge the masks into the single mask variable using bitwise OR
    mask = cv2.bitwise_or(mask, mask_lower_face)
    mask = cv2.bitwise_or(mask, mask_lower_chin)
    
    mask = cv2.bitwise_not(mask)
    
    # Apply the mask to the image
    image = cv2.bitwise_and(image, image, mask=mask)
    image2 = cv2.bitwise_and(image2, image2, mask=mask)

    # Restore images
    img_npy = restore_img(image, batch_size)
    img_npy2 = restore_img(image2, batch_size)
    return image, img_npy, img_npy2

def presence_of_smile(image, image2, idx, batch_size):
    # Load the face detector and shape predictor models
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/home/tpei0009/shape_predictor_68_face_landmarks.dat")
    image = np.array(image)
    image2 = np.array(image2)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector(gray)

    # Check if a face is detected, and if not, skip this detection
    if not faces:
        print('Debug: Nofaces.')
        img_npy = restore_img(image, batch_size)
        img_npy2 = restore_img(image2, batch_size)
        return image, img_npy, img_npy2

    face = faces[0]
    landmarks = predictor(gray, face)
    
    # Create mask for the whole face
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Extract the coordinates of the facial landmarks
    landmarks_coords = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])
    # Create a convex hull from the landmarks to outline the facial features
    hull = cv2.convexHull(landmarks_coords)
    # Fill the convex hull with white (255) on the mask
    cv2.fillConvexPoly(mask, hull, 255)
    
    # Apply the mask to your image, setting non-facial regions to pixel 0
    image = cv2.bitwise_and(image, image, mask=mask)
    image2 = cv2.bitwise_and(image2, image2, mask=mask)
    
    # Restore images
    img_npy = restore_img(image, batch_size)
    img_npy2 = restore_img(image2, batch_size)
    return image, img_npy, img_npy2

def confounder_mask(img, img2, batch_size, denorm_fn=lambda x: x * 0.5 + 0.5):
    # img = denorm_fn(img.detach().cpu().numpy())
    img = denorm_fn(img)
    img2 = denorm_fn(img2)
    img = np.transpose(img, axes=(0, 2, 3, 1))
    img2 = np.transpose(img2, axes=(0, 2, 3, 1))
    img = (img * 255).astype('uint8')
    img2 = (img2 * 255).astype('uint8')
    output_images = []
    output_images2 = []

    for idx, i in enumerate(img):
        i = Image.fromarray(i)
        img2 = Image.fromarray(img2[idx])
        # print(type(x))
        image, img_npy, img_npy2 = presence_of_smile(i,img2, idx, batch_size)
        # For stacking, add each image to the list
        output_images.append(img_npy)
        output_images2.append(img_npy2)
    stacked_img_npy = np.vstack(output_images)    
    stacked_img_npy2 = np.vstack(output_images2)     

    return stacked_img_npy, stacked_img_npy2


def save_imgs(img, denorm_fn=lambda x: x * 0.5 + 0.5):
    img = denorm_fn(img.detach().cpu().numpy())
    img = np.transpose(img, axes=(0, 2, 3, 1))
    img = (img * 255).astype('uint8')

    for idx, i in enumerate(img):
        i = Image.fromarray(i)
        i.save(f'{idx}.png')

def save_imgs_1(img, denorm_fn=lambda x: x * 0.5 + 0.5):
    img = denorm_fn(img.detach().cpu().numpy())
    img = np.transpose(img, axes=(0, 2, 3, 1))
    img = (img * 255).astype('uint8')

    for idx, i in enumerate(img):
        i = Image.fromarray(i)
        i.save(f'{idx}_1.png')
        
def print_dict(d, prefix=''):
    for k, v in d.items():
        if isinstance(v, dict):
            print(f'{prefix}{k}:')
            print_dict(v, prefix=prefix + '  ')
        else:
            print(f'{prefix}{k}: {v}')


def merge_all_chunks(num_chunks, path, exp_name):

    stats = {
        'cf': 0,
        'cf5': 0,
        'loic-flip': 0,
        'untargeted': 0,
        'untargeted5': 0,
        'l1': 0,
        'l inf': 0,
    }

    stats = {
        'n': 0,
        'clean acc': 0,
        'clean acc5': 0,
        'filtered': copy.deepcopy(stats),
        'attack': copy.deepcopy(stats),
    }

    for chunk in range(num_chunks):
        with open(osp.join(path, 'Results', exp_name,
                           f'c-{chunk}_{num_chunks}-summary.yaml'),
                  'r') as f:
            chunk_summary = yaml.load(f, Loader=yaml.FullLoader)

        stats['n'] += chunk_summary['n']
        stats['clean acc'] += chunk_summary['clean acc'] * chunk_summary['n']
        stats['clean acc5'] += chunk_summary['clean acc5'] * chunk_summary['n']

        for data_type in ['attack', 'filtered']:
            for k, v in stats[data_type].items():
                stats[data_type][k] += chunk_summary[data_type][k] * chunk_summary['n']

    for data_type in ['attack', 'filtered']:
        for k, v in stats[data_type].items():
            stats[data_type][k] /= stats['n']
    stats['clean acc'] /= stats['n']
    stats['clean acc5'] /= stats['n']

    with open(osp.join(path, 'Results', exp_name, 'summary.yaml'), 'w') as f:
        f.write(str(stats))

    print('=' * 50, '\nMerged Results:\n\n')
    print_dict(stats)
    print('=' * 50)



@torch.no_grad()
def generate_mask(x1, x2, dilation):
    x1_np = x1.detach().cpu().numpy()
    x2_np = x2.detach().cpu().numpy()
    batch_size = x1_np.shape[0]
    output_images = []
    # i = confounder_mask(x1)
    x1_mask, x2_mask = confounder_mask(x1_np,x2_np, batch_size)
    # output_images = []
    # x2_mask = confounder_mask(x2_np, batch_size)
    x1_mask = torch.from_numpy(x1_mask).to(device)
    x2_mask = torch.from_numpy(x2_mask).to(device)
    assert (dilation % 2) == 1, 'dilation must be an odd number'
    mask =  (x1_mask - x2_mask).abs().sum(dim=1, keepdim=True)
    mask = mask / mask.view(mask.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
    
    # Add thresholding - values >= 0.8 become 1, others become 0
    # mask = (mask >= 0.5).float()
    mask = torch.where(mask > 0.35, mask, torch.tensor(0.0))
    # mask = torch.where(mask >= 0.5, torch.tensor(1.0), mask)
    
    dil_mask = F.max_pool2d(mask,
                        dilation, stride=1,
                        padding=(dilation - 1) // 2)
    return mask, dil_mask
