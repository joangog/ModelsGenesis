from __future__ import print_function
import math
import os
import random
import copy
import scipy
import imageio
import string
import numpy as np
from skimage.transform import resize
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb
from glob import glob
import torch
from PIL import ImageFilter
import torch.nn.functional as F


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def seed_worker(worker_id):  # This is for the workers of the dataloader that need different seeds
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_luna_pretrain_list(ratio):
    x_train = []
    with open('train_val_txt/luna_train.txt', 'r') as f:
        for line in f:
            x_train.append(line.strip('\n'))
    return x_train[:int(len(x_train) * ratio)]


def get_luna_finetune_list(ratio, path, train_fold):
    x_train = []
    with open('train_val_txt/luna_train.txt', 'r') as f:
        for line in f:
            x_train.append(line.strip('\n'))
    return x_train[int(len(x_train) * ratio):]


def get_luna_list(config, train_fold, valid_fold, test_fold, suffix, file_list):
    x_train = []
    x_valid = []
    x_test = []
    for i in train_fold:
        for file in os.listdir(os.path.join(config.data, 'subset' + str(i))):
            if suffix in file:
                if file_list is not None and file.split('_')[0] in file_list:
                    x_train.append(os.path.join(config.data, 'subset' + str(i), file))
                elif file_list is None:
                    x_train.append(os.path.join(config.data, 'subset' + str(i), file))
    for i in valid_fold:
        for file in os.listdir(os.path.join(config.data, 'subset' + str(i))):
            if suffix in file:
                x_valid.append(os.path.join(config.data, 'subset' + str(i), file))
    for i in test_fold:
        for file in os.listdir(os.path.join(config.data, 'subset' + str(i))):
            if suffix in file:
                x_test.append(os.path.join(config.data, 'subset' + str(i), file))
    return x_train, x_valid, x_test

def get_brats_list(data, ratio):
    val_patients_list = []
    train_patients_list = []
    test_patients_list = []
    with open('./pytorch/train_val_txt/brats_train.txt', 'r') as f:
        for line in f:
            line = line.strip('\n')
            train_patients_list.append(os.path.join(data, line))
    with open('./pytorch/train_val_txt/brats_valid.txt', 'r') as f:
        for line in f:
            line = line.strip('\n')
            val_patients_list.append(os.path.join(data, line))
    with open('./pytorch/train_val_txt/brats_test.txt', 'r') as f:

        for line in f:
            line = line.strip('\n')
            test_patients_list.append(os.path.join(data, line))
    train_patients_list = train_patients_list[: int(len(train_patients_list) * ratio)]
    print(
        f"train patients: {len(train_patients_list)}, valid patients: {len(val_patients_list)},"
        f"test patients {len(test_patients_list)}")
    return train_patients_list, val_patients_list, test_patients_list

def get_brats_pretrain_list(data, ratio, suffix):
    val_patients_list = []
    train_patients_list = []
    test_patients_list = []
    with open('./pytorch/train_val_txt/brats_train.txt', 'r') as f:
        for line in f:
            line = line.strip('\n')
            train_patient_path = os.path.join(data, line)
            for file in os.listdir(train_patient_path):
                if suffix in file:
                    train_patients_list.append(os.path.join(train_patient_path, file))
    with open('./pytorch/train_val_txt/brats_valid.txt', 'r') as f:
        for line in f:
            line = line.strip('\n')
            val_patient_path = os.path.join(data, line)
            for file in os.listdir(val_patient_path):
                if suffix in file:
                    val_patients_list.append(os.path.join(val_patient_path, file))
    with open('./pytorch/train_val_txt/brats_test.txt', 'r') as f:
        for line in f:
            line = line.strip('\n')
            test_patient_path = os.path.join(data, line)
            for file in os.listdir(test_patient_path):
                if suffix in file:
                    test_patients_list.append(os.path.join(test_patient_path, file))
    train_patients_list = train_patients_list[: int(len(train_patients_list) * ratio)]
    print(
        f"train patients: {len(train_patients_list)}, valid patients: {len(val_patients_list)},"
        f"test patients {len(test_patients_list)}")
    return train_patients_list, val_patients_list, test_patients_list

def get_luna_finetune_nodule(config, train_fold, valid_txt, test_txt, suffix, file_list):
    x_train = []
    x_valid = []
    x_test = []
    for i in train_fold:
        for file in os.listdir(os.path.join(config.data, 'subset' + str(i))):
            if suffix in file:
                if file_list is not None and file.split('_')[0] in file_list:
                    x_train.append(os.path.join(config.data, 'subset' + str(i), file))
                elif file_list is None:
                    x_train.append(os.path.join(config.data, 'subset' + str(i), file))
    with open(valid_txt, 'r') as f:
        for line in f:
            x_valid.append(line.strip('\n'))
    with open(test_txt, 'r') as f:
        for line in f:
            x_test.append(line.strip('\n'))
    return x_train, x_valid, x_test


def divide__luna_true_positive(data_list):
    true_list = []
    false_list = []
    for i in data_list:
        name = os.path.split(i)[-1]
        label = name.split('_')[1]
        if label == '1':
            true_list.append(i)
        else:
            false_list.append(i)
    return true_list, false_list


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def adjust_learning_rate(epoch, args, optimizer):
    # iterations = opt.lr_decay_epochs.split(',')
    # opt.lr_decay_epochs_list = list([])
    # for it in iterations:
    #     opt.lr_decay_epochs_list.append(int(it))
    # steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs_list))
    # if steps > 0:
    #     new_lr = opt.lr * (opt.lr_decay_rate ** steps)
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = new_lr
    lr = args.lr
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def bceDiceLoss(input, target, train=True):
    bce = F.binary_cross_entropy_with_logits(input, target)
    smooth = 1e-5
    num = target.size(0)
    input = input.view(num, -1)
    target = target.view(num, -1)
    intersection = (input * target)
    dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
    dice = 1 - dice.sum() / num
    if train:
        return dice + 0.2 * bce
    return dice


def dice_coeff(input, target):
    smooth = 1e-5
    num = target.size(0)
    input = input.view(num, -1)
    target = target.view(num, -1)
    intersection = (input * target)
    dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
    dice = dice.sum() / num
    dice = dice.item()
    return dice


def thor_dice_loss(input, target, train=True):
    # print(input.shape, target.shape)
    es_dice = bceDiceLoss(input[:, 0], target[:, 0], train)
    tra_dice = bceDiceLoss(input[:, 1], target[:, 1], train)
    aor_dice = bceDiceLoss(input[:, 2], target[:, 2], train)
    heart_dice = bceDiceLoss(input[:, 3], target[:, 3], train)
    print(f'label1 dice {es_dice}, label2 dice {tra_dice}, label3 dice{aor_dice}, label4 dice{heart_dice}')
    return es_dice + tra_dice + aor_dice + heart_dice


def brats_dice_loss(input, target, train=True):
    wt_loss = bceDiceLoss(input[:, 0], target[:, 0], train)
    tc_loss = bceDiceLoss(input[:, 1], target[:, 1], train)
    et_loss = bceDiceLoss(input[:, 2], target[:, 2], train)
    print(f'wt loss: {wt_loss}, tc_loss : {tc_loss}, et_loss: {et_loss}')
    return wt_loss + tc_loss + et_loss

def sinkhorn(args, Q: torch.Tensor, nmb_iters: int) -> torch.Tensor:
    with torch.no_grad():
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        K, B = Q.shape

        if not args.cpu:
            u = torch.zeros(K).cuda()
            r = torch.ones(K).cuda() / K
            c = torch.ones(B).cuda() / B
        else:
            u = torch.zeros(K)
            r = torch.ones(K) / K
            c = torch.ones(B) / B

        for _ in range(nmb_iters):
            u = torch.sum(Q, dim=1)

            Q *= (r / u).unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])
    
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def data_augmentation(x, y, prob=0.5):
    # augmentation by flipping
    cnt = 3
    while random.random() < prob and cnt > 0:
        degree = random.choice([0, 1, 2])
        x = np.flip(x, axis=degree)
        y = np.flip(y, axis=degree)
        cnt = cnt - 1

    return x, y

def nonlinear_transformation(x, prob=0.5):
    if random.random() >= prob:
        return x
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x

def local_pixel_shuffling(x, prob=0.5):
    if random.random() >= prob:
        return x
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    _, img_rows, img_cols, img_deps = x.shape
    num_block = 10000
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows//10)
        block_noise_size_y = random.randint(1, img_cols//10)
        block_noise_size_z = random.randint(1, img_deps//10)
        noise_x = random.randint(0, img_rows-block_noise_size_x)
        noise_y = random.randint(0, img_cols-block_noise_size_y)
        noise_z = random.randint(0, img_deps-block_noise_size_z)
        window = orig_image[0, noise_x:noise_x+block_noise_size_x, 
                               noise_y:noise_y+block_noise_size_y, 
                               noise_z:noise_z+block_noise_size_z,
                           ]
        window = window.flatten()
        np.random.shuffle(window)
        window = window.reshape((block_noise_size_x, 
                                 block_noise_size_y, 
                                 block_noise_size_z))
        image_temp[0, noise_x:noise_x+block_noise_size_x, 
                      noise_y:noise_y+block_noise_size_y, 
                      noise_z:noise_z+block_noise_size_z] = window
    local_shuffling_x = image_temp

    return local_shuffling_x

def image_in_painting(x):
    _, img_rows, img_cols, img_deps = x.shape
    cnt = 5
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = random.randint(img_rows//6, img_rows//3)
        block_noise_size_y = random.randint(img_cols//6, img_cols//3)
        block_noise_size_z = random.randint(img_deps//6, img_deps//3)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        noise_z = random.randint(3, img_deps-block_noise_size_z-3)
        x[:, 
          noise_x:noise_x+block_noise_size_x, 
          noise_y:noise_y+block_noise_size_y, 
          noise_z:noise_z+block_noise_size_z] = np.random.rand(block_noise_size_x, 
                                                               block_noise_size_y, 
                                                               block_noise_size_z, ) * 1.0
        cnt -= 1
    return x

def image_out_painting(x):
    _, img_rows, img_cols, img_deps = x.shape
    image_temp = copy.deepcopy(x)
    x = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], ) * 1.0
    block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
    block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)
    block_noise_size_z = img_deps - random.randint(3*img_deps//7, 4*img_deps//7)
    noise_x = random.randint(3, img_rows-block_noise_size_x-3)
    noise_y = random.randint(3, img_cols-block_noise_size_y-3)
    noise_z = random.randint(3, img_deps-block_noise_size_z-3)
    x[:, 
      noise_x:noise_x+block_noise_size_x, 
      noise_y:noise_y+block_noise_size_y, 
      noise_z:noise_z+block_noise_size_z] = image_temp[:, noise_x:noise_x+block_noise_size_x, 
                                                       noise_y:noise_y+block_noise_size_y, 
                                                       noise_z:noise_z+block_noise_size_z]
    cnt = 4
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
        block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)
        block_noise_size_z = img_deps - random.randint(3*img_deps//7, 4*img_deps//7)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        noise_z = random.randint(3, img_deps-block_noise_size_z-3)
        x[:, 
          noise_x:noise_x+block_noise_size_x, 
          noise_y:noise_y+block_noise_size_y, 
          noise_z:noise_z+block_noise_size_z] = image_temp[:, noise_x:noise_x+block_noise_size_x, 
                                                           noise_y:noise_y+block_noise_size_y, 
                                                           noise_z:noise_z+block_noise_size_z]
        cnt -= 1
    return x
                


def generate_pair(img, batch_size, config, status="test"):
    img_rows, img_cols, img_deps = img.shape[2], img.shape[3], img.shape[4]
    while True:
        index = [i for i in range(img.shape[0])]
        random.shuffle(index)
        y = img[index[:batch_size]]
        x = copy.deepcopy(y)
        for n in range(batch_size):
            
            # Autoencoder
            x[n] = copy.deepcopy(y[n])
            
            # Flip
            x[n], y[n] = data_augmentation(x[n], y[n], config.flip_rate)

            # Local Shuffle Pixel
            x[n] = local_pixel_shuffling(x[n], prob=config.local_rate)
            
            # Apply non-Linear transformation with an assigned probability
            x[n] = nonlinear_transformation(x[n], config.nonlinear_rate)
            
            # Inpainting & Outpainting
            if random.random() < config.paint_rate:
                if random.random() < config.inpaint_rate:
                    # Inpainting
                    x[n] = image_in_painting(x[n])
                else:
                    # Outpainting
                    x[n] = image_out_painting(x[n])

        # Save sample images module
        if config.save_samples is not None and status == "train" and random.random() < 0.01:
            n_sample = random.choice( [i for i in range(config.batch_size)] )
            sample_1 = np.concatenate((x[n_sample,0,:,:,2*img_deps//6], y[n_sample,0,:,:,2*img_deps//6]), axis=1)
            sample_2 = np.concatenate((x[n_sample,0,:,:,3*img_deps//6], y[n_sample,0,:,:,3*img_deps//6]), axis=1)
            sample_3 = np.concatenate((x[n_sample,0,:,:,4*img_deps//6], y[n_sample,0,:,:,4*img_deps//6]), axis=1)
            sample_4 = np.concatenate((x[n_sample,0,:,:,5*img_deps//6], y[n_sample,0,:,:,5*img_deps//6]), axis=1)
            final_sample = np.concatenate((sample_1, sample_2, sample_3, sample_4), axis=0)
            final_sample = final_sample * 255.0
            final_sample = final_sample.astype(np.uint8)
            file_name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(10)])+'.'+config.save_samples
            imageio.imwrite(os.path.join(config.sample_path, config.exp_name, file_name), final_sample)

        yield (x, y)
