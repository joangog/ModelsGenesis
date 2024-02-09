import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch
import torchio.transforms
import copy
import time
import numpy
import argparse
import unet3d
from utils import *
from data import *

parser = argparse.ArgumentParser(description='Self Training benchmark')
parser.add_argument('--data', metavar='DIR', default='/mnt/5C5C25FB5C25D116/data/BraTS2018',
                    help='path to dataset')
parser.add_argument('--b', default=4, type=int, help='batch size')
parser.add_argument('--epochs', default=300, type=int, help='epochs to train')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--output', default='./brats_finetune', type=str, help='output path')
parser.add_argument('--workers', default=4, type=int, help='num of workers')
parser.add_argument('--gpus', default='0', type=str, help='gpu indexs')
parser.add_argument('--ratio', default=1, type=float, help='ratio of data used for pretraining')
parser.add_argument('--momentum', default=0.9)
parser.add_argument('--weight', default='./pytorch/pretrained_weights/Genesis_Chest_CT.pt', type=str)
parser.add_argument('--weight_decay', default=0)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--patience', default=None, type=int)
parser.add_argument('--tensorboard', action='store_true', default=False)
parser.add_argument('--cpu', action='store_true', default=False)
args = parser.parse_args()
if not os.path.exists(args.output):
    os.makedirs(args.output)
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
# torch.backends.cudnn.benchmark = True

# Set seed
set_seed(args.seed)
print(f'Seed is {args.seed}')

curr_time = str(time.time()).replace(".", "")
run_name = f'model_genesis_{curr_time}'

writer = None
if args.tensorboard:
    # Create tensorboard writer
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    writer = SummaryWriter(os.path.join(args.output, run_name))

#Declare the Dice Loss
def torch_dice_coef_loss(y_true,y_pred, smooth=1.):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return 1. - ((2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth))

# prepare your own data
generator = DataGenerator(args)
data_loader = generator.brats_finetune()
train_generator = data_loader['train']
valid_generator = data_loader['eval']

# prepare the 3D model

model = unet3d.UNet3D(in_chann=4, n_class=3)

# Load pre-trained weights
weight_dir = args.weight
checkpoint = torch.load(weight_dir)
state_dict = checkpoint['state_dict']
# Unparallelize weights
model_dict = {}
for key in state_dict.keys():
    model_dict[key.replace("module.", "")] = state_dict[key]
# Adjust first conv and last conv for downstream dataset
# First conv weight
first_conv_weight = model_dict['down_tr64.ops.0.conv1.weight']
first_conv_weight = first_conv_weight.repeat((1, 4, 1, 1, 1))
model_dict['down_tr64.ops.0.conv1.weight'] = first_conv_weight
# Last conv weight
last_conv_weight = model_dict['out_tr.final_conv.weight']
last_conv_weight = last_conv_weight.repeat((3, 1, 1, 1, 1))
model_dict['out_tr.final_conv.weight'] = last_conv_weight
# Last conv bias
last_conv_bias = model_dict['out_tr.final_conv.bias']
last_conv_bias = last_conv_bias.repeat((3))
model_dict['out_tr.final_conv.bias'] = last_conv_bias
pretrain_dict = {k: v for k, v in model_dict.items() if
                    k in model_dict}  # load the encoder and decoder part
model.load_state_dict(model_dict)

model.cuda()
model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
criterion = brats_dice_loss
optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# Train the model

train_losses = []
valid_losses = []
avg_train_losses = []
avg_valid_losses = []
best_loss = 100000
num_epoch_no_improvement = 0

for epoch in range(0, args.epochs + 1):
    scheduler.step(epoch)
    model.train()
    for iteration, (image, gt) in enumerate(train_generator):
        image = image.cuda().float()
        gt = gt.cuda().float()
        pred = model(image)
        loss = criterion(pred, gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(round(loss.item(), 2))
        if (iteration + 1) % 5 == 0:
            print('Epoch [{}/{}], iteration {}, Loss:{:.6f}, {:.6f}'
                    .format(epoch + 1, args.epochs, iteration + 1, loss.item(), np.average(train_losses)))
            sys.stdout.flush()

    with torch.no_grad():
        model.eval()
        print("validating....")
        for i, (x, y) in enumerate(valid_generator):
            x = x.cuda()
            y = y.cuda().float()
            pred = model(x)
            loss = criterion(pred, y)
            valid_losses.append(loss.item())

    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)
    print("Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(epoch + 1, valid_loss,
                                                                                train_loss))
    train_losses = []
    valid_losses = []
    if valid_loss < best_loss:  # Saves only best epoch
        print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, valid_loss))
        best_loss = valid_loss
        num_epoch_no_improvement = 0
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict(),  # only save encoder
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(args.output, run_name + ".pt"))
        print("Saving model ", run_name + ".pt")
    else:
        print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss,
                                                                                                    num_epoch_no_improvement))
        num_epoch_no_improvement += 1
        if num_epoch_no_improvement == args.patience:
            print("Early Stopping")
            break

    if args.tensorboard:
        writer.add_scalar('loss/train', train_loss, epoch)  # Write train loss on tensorboard
        writer.add_scalar('loss/val', valid_loss, epoch)  # Write val loss on tensorboard
        if epoch == 0:  # Only on the first iteration, write model graph on tensorboard
            writer.add_graph(model, image)

    sys.stdout.flush()

if args.tensorboard:
    writer.close()
