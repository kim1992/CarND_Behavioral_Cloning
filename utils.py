import numpy as np
import cv2
from sklearn.utils import shuffle

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
BATCH_SIZE = 64
EPOCH = 4

# Load image randomly（left / middle / right）
# Add the steering angle by +0.2/-0.2 for left/right image respectively
def random_img_choose(sample):
    choice = np.random.choice(3, 1)
    if choice == 0:
        name = './data/IMG/' + sample[0].split('/')[-1]
        center_image = cv2.imread(name)
        center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
        center_angle = float(sample[3])
    elif choice == 1:
        name = './data/IMG/' + sample[1].split('/')[-1]
        center_image = cv2.imread(name)
        center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
        center_angle = float(sample[3]) + 0.2
    else:
        name = './data/IMG/' + sample[2].split('/')[-1]
        center_image = cv2.imread(name)
        center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
        center_angle = float(sample[3]) - 0.2
    return center_image, center_angle 

# Flip the image randomly in horizontal direction
def random_img_flip(img, angle):
    rand = np.random.rand()
    if rand > 0.5:
        # 1:    horizontal flip
        # 0:    vertical flip 
        # -1:   horizontal & vertical flip
        img = cv2.flip(img, 1)
        angle = -angle
    return img, angle

# Add some shadow to the image randomly
def random_shadow(image):
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line:
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


# Ajust brightness of the image randomly
def random_brightness(image):
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# Define a generator to yield each single batch of the training data
# in order to save the memory of CPU 
def generator(samples, batch_size):
    shuffle(samples)
    num_samples = len(samples)
    while 1:    # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset+batch_size]
            images = []
            angles = []
            for i, sample in enumerate(batch_samples):
                img, angle = random_img_choose(sample)
                img, angle = random_img_flip(img,angle)
                img = random_shadow(img)
                img = random_brightness(img)
                
                images.append(img)
                angles.append(angle)

            yield shuffle(np.array(images), np.array(angles))
