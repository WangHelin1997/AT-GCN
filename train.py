import sys, os, os.path, time
import argparse
import numpy
import torch
import torch.nn as nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.autograd import Variable
from net import gcnNet, CNN10
from util_in import *
from util_out import *
from util_f1 import *
import numpy as np

torch.backends.cudnn.benchmark = True

# Parse input arguments
def mybool(s):
    return s.lower() in ['t', 'true', 'y', 'yes', '1']

def mixup_data(x, y, alpha=0.2):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(class_criterion, pred, y_a, y_b, lam):
    return lam * class_criterion(pred, y_a) + (1 - lam) * class_criterion(pred, y_b)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--ckpt_size', type = int, default = 2000)      # how many batches per checkpoint
parser.add_argument('--optimizer', type = str, default = 'adam', choices = ['adam', 'sgd'])
parser.add_argument('--init_lr', type = float, default = 3e-5)
parser.add_argument('--lr_patience', type = int, default = 3)
parser.add_argument('--lr_factor', type = float, default = 0.9)
parser.add_argument('--max_ckpt', type = int, default = 200)
parser.add_argument('--random_seed', type = int, default = 15213)
args = parser.parse_args()
numpy.random.seed(args.random_seed)

# Prepare log file and model directory
expid = 'CNN14_mixup_specaugment_drop'
WORKSPACE = os.path.join('../../workspace/audioset_GCN_8', expid)
MODEL_PATH = os.path.join(WORKSPACE, 'model')
if not os.path.exists(MODEL_PATH): os.makedirs(MODEL_PATH)
LOG_FILE = os.path.join(WORKSPACE, 'train.log')
with open(LOG_FILE, 'w'):
    pass

def write_log(s):
    timestamp = time.strftime('%m-%d %H:%M:%S')
    msg = '[' + timestamp + '] ' + s
    print(msg)
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')

# Load data
write_log('Loading data ...gcn_mixup_specaugment_drop_BCELoss')
train_gen = batch_generator(batch_size = args.batch_size, random_seed = args.random_seed)
gas_valid_x, gas_valid_y, _ = bulk_load('GAS_valid')
gas_eval_x, gas_eval_y, _ = bulk_load('GAS_eval')
# DCASE Task
# dcase_valid_x, dcase_valid_y, _ = bulk_load('DCASE_valid')
# dcase_test_x, dcase_test_y, _ = bulk_load('DCASE_test')
# dcase_test_frame_truth = load_dcase_test_frame_truth()
# DCASE_CLASS_IDS = [318, 324, 341, 321, 307, 310, 314, 397, 325, 326, 323, 319, 14, 342, 329, 331, 316]

# Build model
adj_pth = 'audioset_ba_adj.pkl'
inp_pth = 'audioset_glove_word2vec.pkl'
model_pth = '/home/cdd/code/cmu-thesis/workspace/audioset_GCN_7/CNN14_mixup_specaugment_drop/model/checkpoint199.pt'
fixed = False
model = gcnNet(args, adj=adj_pth, inp=inp_pth, pretrained=False, model_pth=model_pth, fixed=fixed).cuda()
model.load_state_dict(torch.load(model_pth))
# model = Net(args).cuda()
if args.optimizer == 'sgd':
    if fixed == True:
        optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = args.init_lr, momentum = 0.9, nesterov = True)
    else :
        optimizer = SGD(model.parameters(), lr = args.init_lr, momentum = 0.9, nesterov = True)
elif args.optimizer == 'adam':
    if fixed == True:
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = args.init_lr, betas=(0.9, 0.999), eps=1e-8)
    else :
        optimizer = Adam(model.parameters(), lr = args.init_lr, betas=(0.9, 0.999), eps=1e-8)
scheduler = ReduceLROnPlateau(optimizer, mode = 'max', factor = args.lr_factor, patience = args.lr_patience) if args.lr_factor < 1.0 else None
criterion = nn.BCELoss()
# criterion = nn.MultiLabelSoftMarginLoss()
# Train model
write_log('Training model ...')
write_log('                            ||       GAS_VALID       ||        GAS_EVAL       ')
write_log(" CKPT |    LR    |  Tr.LOSS ||  MAP  |  MAUC |   d'  ||  MAP  |  MAUC |   d'  ")
FORMAT  = ' %#4d | %8.0003g | %8.0006f || %5.3f | %5.3f |%6.03f || %5.3f | %5.3f |%6.03f '
SEP     = ''.join('+' if c == '|' else '-' for c in FORMAT)
write_log(SEP)
lr = args.init_lr

for checkpoint in range(1, args.max_ckpt + 1):
    # Train for args.ckpt_size batches
    model.train()
    train_loss = 0
    for batch in range(1, args.ckpt_size + 1):
        x, y = next(train_gen)
        optimizer.zero_grad()
        # with Mixup
        data, target_a, target_b, lam = mixup_data(x=x, y=y, alpha=0.2)
        global_prob,embedding = model(data)
        global_prob.clamp(min = 1e-7, max = 1 - 1e-7)
        loss = mixup_criterion(criterion, global_prob, target_a, target_b, lam)
        # without Mixup
        # global_prob,frame_prob = model(x)
        # global_prob.clamp(min = 1e-7, max = 1 - 1e-7)
        # loss = criterion(global_prob, y)
        train_loss += loss.item()
        if numpy.isnan(train_loss) or numpy.isinf(train_loss): break
        loss.backward()
        optimizer.step()
        sys.stderr.write('Checkpoint %d, Batch %d / %d, avg train loss = %f\r' % \
                         (checkpoint, batch, args.ckpt_size, train_loss / batch))
#         del x, y, global_prob, loss         # This line and next line: to save GPU memory
#         torch.cuda.empty_cache()            # I don't know if they're useful or not
    train_loss /= args.ckpt_size

    # Evaluate model
    model.eval()
    sys.stderr.write('Evaluating model on GAS_VALID ...\r')
    global_prob = model.predict(gas_valid_x, verbose = False)
    gv_map, gv_mauc, gv_dprime = gas_eval(global_prob, gas_valid_y)
    sys.stderr.write('Evaluating model on GAS_EVAL ... \r')
    global_prob = model.predict(gas_eval_x, verbose = False)
    ge_map, ge_mauc, ge_dprime = gas_eval(global_prob, gas_eval_y)
    
    # Write log
    write_log(FORMAT % (
        checkpoint, optimizer.param_groups[0]['lr'], train_loss,
        gv_map, gv_mauc, gv_dprime,
        ge_map, ge_mauc, ge_dprime
    ))

    # Abort if training has gone mad
    if numpy.isnan(train_loss) or numpy.isinf(train_loss):
        write_log('Aborted.')
        break

    # Save model.
    MODEL_FILE = os.path.join(MODEL_PATH, 'checkpoint%d.pt' % checkpoint)
    sys.stderr.write('Saving model to %s ...\r' % MODEL_FILE)
    torch.save(model.state_dict(), MODEL_FILE)

    # if checkpoint % 5 == 0 and checkpoint >= 45 :
    #     lr = lr * 0.8 
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr


    # Update learning rate
    if scheduler is not None:
        scheduler.step(gv_map)

write_log('DONE!')
