import numpy as np
import torch
import random
from tensorboardX import SummaryWriter
import torch.optim as optim
import os
from model.model import model_fn_decorator
from model.nets import my_model
from dataset.load_data import create_dataset
from tqdm import tqdm
from utils.loss_util import multi_VGGPerceptualLoss
from utils.common import CosineAnnealingWarmRestarts,mkdir, logging,get_LR_scheduler
import os.path as osp
from config.config import args,options_for_logger
import time
from datetime import timedelta


from utils.logger import get_root_logger
from utils.utility import get_time_str

def train_epoch(args, TrainImgLoader, model, model_fn, optimizer, epoch, iters, lr_scheduler):
    """
    Training Loop for each epoch
    """
    tbar = tqdm(TrainImgLoader)
    total_loss = 0
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    for batch_idx, data in enumerate(tbar):
        loss = model_fn(args, data, model, iters)
        
        TRAIN_FLAG = True
        #pass invalid data
        if args.DATA_TYPE == "ICBLN":
            TRAIN_FLAG = data['train_flag'].all()


        # if all data are valid ,then update loss
        if TRAIN_FLAG:
            # backward and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        iters += 1
        total_loss += loss.item()
        avg_train_loss = total_loss / (batch_idx+1)
        desc = 'Training  : Epoch %d, lr %.7f, Avg. Loss = %.5f' % (epoch, lr, avg_train_loss)
        tbar.set_description(desc)
        tbar.update()
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    # the learning rate is adjusted after each epoch
    lr_scheduler.step()

    return lr, avg_train_loss, iters

def init():
    # Make dirs
    args.LOGS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'logs')
    args.NETS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'net_checkpoints')
    args.VISUALS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'train_visual')
    mkdir(args.LOGS_DIR)
    mkdir(args.NETS_DIR)
    mkdir(args.VISUALS_DIR)

    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.GPU_ID
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # random seed
    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed_all(args.SEED)
    if args.SEED == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # summary writer
    logger = SummaryWriter(args.LOGS_DIR)
    
    return logger, device

def load_checkpoint(model, optimizer, load_epoch):
    load_dir = args.NETS_DIR + '/checkpoint' + '_' + '%06d' % load_epoch + '.tar'
    print('Loading pre-trained checkpoint %s' % load_dir)
    model_state_dict = torch.load(load_dir)['state_dict']
    model.load_state_dict(model_state_dict)
    optimizer_dict = torch.load(load_dir)['optimizer']
    optimizer.load_state_dict(optimizer_dict)
    learning_rate = torch.load(load_dir)['learning_rate']
    iters = torch.load(load_dir)['iters']
    
    log_file = get_resume_log_file(args.SAVE_PREFIX)
    base_logger = get_root_logger(logger_name='base', log_level=logging.INFO, log_file=log_file)
    base_logger.info('Learning rate recorded from the checkpoint: %s' % str(learning_rate))

    return learning_rate, iters

def get_resume_log_file(folder_path):
    import glob
    files = glob.glob(os.path.join(folder_path, '*.log'))
    # 获取最新的.log文件
    latest_file = max(files, key=os.path.getctime)
    # 输出最新的.log文件名
    #print(latest_file)
    new_file = os.path.splitext(latest_file)[0]+'_'+get_time_str()+'.log'

    return new_file

def main():
    logger, device = init()
    # create model
    model = my_model(en_feature_num=args.EN_FEATURE_NUM,
                     en_inter_num=args.EN_INTER_NUM,
                     de_feature_num=args.DE_FEATURE_NUM,
                     de_inter_num=args.DE_INTER_NUM,
                     dmm_number=args.DMM_NUMBER,
                     ).to(device)
    model._initialize_weights()

    # create optimizer
    optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': args.BASE_LR}], betas=(0.9, 0.999))
    learning_rate = args.BASE_LR
    iters = 0
    # resume training
    if args.LOAD_EPOCH:
        learning_rate, iters = load_checkpoint(model, optimizer, args.LOAD_EPOCH)
    # create loss function
    loss_fn = multi_VGGPerceptualLoss(lam=args.LAM, lam_p=args.LAM_P).to(device)
    # create learning rate scheduler
    if args.lr_policy == 'default':
        
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_MULT, eta_min=args.ETA_MIN,
                                               last_epoch=args.LOAD_EPOCH - 1)
    else:
        lr_scheduler = get_LR_scheduler(optimizer, args)
    
    # create training function
    model_fn = model_fn_decorator(loss_fn=loss_fn, device=device)
    # create dataset
    train_path = args.TRAIN_DATASET
    TrainImgLoader = create_dataset(args, data_path=train_path, mode='train')
        
    # logger, print net's structrue and number of parameters
    if not args.LOAD_EPOCH:
        log_file = osp.join(args.SAVE_PREFIX, f"train_{args.EXP_NAME}_{get_time_str()}.log")
        base_logger = get_root_logger(logger_name='base', log_level=logging.INFO, log_file=log_file)
        base_logger.info(options_for_logger)
        print_network(model)
        # start training
        base_logger.info("****start traininig!!!****")
    else:
        base_logger = get_root_logger()

    avg_train_loss = 0
    for epoch in range(args.LOAD_EPOCH + 1, args.EPOCHS + 1):
        # "mark start time"
        start_time = time.time()

        learning_rate, avg_train_loss, iters = train_epoch(args, TrainImgLoader, model, model_fn, optimizer, epoch,
                                                           iters, lr_scheduler)
        logger.add_scalar('Train/avg_loss', avg_train_loss, epoch)
        logger.add_scalar('Train/learning_rate', learning_rate, epoch)

        #mark end time
        end_time = time.time()
        # caculate the total time of running an epoch
        epoch_time_seconds  = end_time - start_time
        epoch_time_timedelta = timedelta(seconds=epoch_time_seconds)
        base_logger.info(f"the {epoch} th, lr {learning_rate:.8f}, AVG.Loss {avg_train_loss:.6f}, total running time: {epoch_time_timedelta} ({epoch_time_seconds:.2f} seconds)")
        
        # Save the network per ten epoch
        if epoch % args.SAVE_EPOCH == 0:
            savefilename = args.NETS_DIR + '/checkpoint' + '_' + '%06d' % epoch + '.tar'
            torch.save({
                'learning_rate': learning_rate,
                'iters': iters,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict()
            }, savefilename)

        # Save the latest model
        savefilename = args.NETS_DIR + '/checkpoint' + '_' + 'latest.tar'
        torch.save({
            'learning_rate': learning_rate,
            'iters': iters,
            'optimizer': optimizer.state_dict(),
            'state_dict': model.state_dict()
        }, savefilename)

def print_network(net):
        """Print the str and parameter number of a network.

        Args:
            net (nn.Module)
        """
 
        net_cls_str = f'{net.__class__.__name__}'

        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))
        
        base_logger = get_root_logger()
        base_logger.info(f'Network: {net_cls_str}, with parameters: {net_params:,d}')
        base_logger.info(net_str)

def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())

if __name__ == '__main__':
    main()
