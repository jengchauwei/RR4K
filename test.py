import datetime
import logging
import numpy as np
import torch
import random

import os
from model.model import model_fn_decorator
from model.nets import my_model
from dataset.load_data import *
from tqdm import tqdm
from utils.loss_util import multi_VGGPerceptualLoss
from utils.common import *
from config.config import args
import logging
import os.path as osp

def test(args, TestImgLoader, model, model_fn_test, save_path, compute_metrics):
    tbar = tqdm(TestImgLoader)
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    total_time = 0
    avg_val_time = 0
    for batch_idx, data in enumerate(tbar):
        model.eval()
        cur_psnr, cur_ssim, cur_lpips, cur_time = model_fn_test(args, data, model, save_path, compute_metrics)
        #number = data['number']
        if args.EVALUATION_METRIC:
            #logging.info(' : LPIPS is %.4f, PSNR is %.4f and SSIM is %.4f' % (cur_lpips, cur_psnr, cur_ssim))
            pass
        if args.EVALUATION_TIME:
            logging.info(' : TIME is %.4f' % ( cur_time))
        total_psnr += cur_psnr
        avg_val_psnr = total_psnr / (batch_idx+1)
        total_ssim += cur_ssim
        avg_val_ssim = total_ssim / (batch_idx+1)
        total_lpips += cur_lpips
        avg_val_lpips = total_lpips / (batch_idx+1)
        # skip calculation for first five samples to avoid warming-up cost
        if batch_idx > 5:
            total_time += cur_time
            avg_val_time = total_time / (batch_idx-5)
        if args.EVALUATION_METRIC:
            desc = 'Test %s : Avg. LPIPS = %.4f, Avg. PSNR = %.4f and SSIM = %.4f' % (args.cur_testset_name,avg_val_lpips, avg_val_psnr, avg_val_ssim)
        elif args.EVALUATION_TIME:
            desc = 'Avg. TIME is %.4f' % avg_val_time
        else:
            desc = 'Test without any evaluation'
        tbar.set_description(desc)
        tbar.update()
    if args.EVALUATION_METRIC:
        logging.warning('%s --> Avg. LPIPS is %.4f, PSNR is %.4f and SSIM is %.4f' % (args.cur_testset_name,avg_val_lpips, avg_val_psnr, avg_val_ssim))
    if args.EVALUATION_TIME:
        logging.warning('Avg. TIME is %.4f' % avg_val_time)

def init():
    # Make dirs
    args.TEST_RESULT_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'test_result')
    mkdir(args.TEST_RESULT_DIR)
    args.NETS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'net_checkpoints')
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

    return device

def load_checkpoint(model):
    if args.LOAD_PATH:
        load_path = args.LOAD_PATH
        save_path = args.TEST_RESULT_DIR + '/customer'
        log_path = args.TEST_RESULT_DIR + '/customer_result.log'

    else:
        load_epoch = args.TEST_EPOCH
        if load_epoch == 'auto':
            load_path = args.NETS_DIR + '/checkpoint_latest.tar'
            save_path = args.TEST_RESULT_DIR + '/latest'
            log_path = args.TEST_RESULT_DIR + '/latest_result.log'
        else:
            load_path = args.NETS_DIR + '/checkpoint' + '_' + '%06d' % load_epoch + '.tar'
            save_path = args.TEST_RESULT_DIR + '/' + '%04d' % load_epoch
            log_path = args.TEST_RESULT_DIR + '/%04d_' % load_epoch + 'result.log'

    mkdir(save_path)
    if load_path.endswith('.pth'):
        model_state_dict = torch.load(load_path)
    else:
        model_state_dict = torch.load(load_path)['state_dict']
    model.load_state_dict(model_state_dict)

    return load_path, save_path, log_path

def load_checkpoint_light(model):
    if args.LOAD_PATH:
        load_path = args.LOAD_PATH
        log_path = args.TEST_RESULT_DIR + '/customer_result.log'
       
    else:
        load_epoch = args.TEST_EPOCH
        if load_epoch == 'auto':
            load_path = args.NETS_DIR + '/checkpoint_latest.tar'
            log_path = args.TEST_RESULT_DIR + '/latest_result.log'

        else:
            load_path = args.NETS_DIR + '/checkpoint' + '_' + '%06d' % load_epoch + '.tar'
            log_path = args.TEST_RESULT_DIR + '/%04d_' % load_epoch + 'result.log'

    if load_path.endswith('.pth'):
        model_state_dict = torch.load(load_path)
    else:
        model_state_dict = torch.load(load_path)['state_dict']
    model.load_state_dict(model_state_dict)

    return load_path, log_path
def get_save_logs_path(args,dataset_name):

    load_epoch = args.TEST_EPOCH
    if load_epoch == 'auto':
        save_path = args.TEST_RESULT_DIR + '/'+dataset_name+'/latest'
        log_path = args.TEST_RESULT_DIR + '/'+dataset_name+ '/latest_result.log'
    else:
        save_path = args.TEST_RESULT_DIR + '/'+dataset_name+'/%04d' % load_epoch 
        log_path = args.TEST_RESULT_DIR + '/'+dataset_name+'/%04d_' % load_epoch + 'result.log'
    mkdir(save_path)
    return save_path, log_path

def set_logging(log_path):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('%(message)s')

    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

def main():
    device = init()
    # load model
    model = my_model(en_feature_num=args.EN_FEATURE_NUM,
                     en_inter_num=args.EN_INTER_NUM,
                     de_feature_num=args.DE_FEATURE_NUM,
                     de_inter_num=args.DE_INTER_NUM,
                     sam_number=args.SAM_NUMBER,
                     ).to(device)

    # load checkpoint
    load_path, save_path, log_path = load_checkpoint(model)

    # set logging for recording information or metrics
    set_logging(log_path)
    logging.warning(datetime.now())

    # computational cost for the model
    if args.EVALUATION_COST:
        calculate_cost(model, input_size=(1, 3, 2176, 3840))

    logging.warning('load model from %s' % load_path)
    logging.warning('save image results to %s' % save_path)
    logging.warning('save logger to %s' % log_path)

    compute_metrics = None
    if args.EVALUATION_TIME:
        # metric calculation may have negative impact on inference speed
        args.EVALUATION_METRIC = False
    if args.EVALUATION_METRIC:
        # load LPIPS metric
        from utils.metric import create_metrics
        compute_metrics = create_metrics(args, device=device)

    loss_fn = None
    model_fn_test = model_fn_decorator(loss_fn=loss_fn, device=device, mode='test')

    # Create dataset
    test_path = args.TEST_DATASET
    # Set test batch size to 1 for avoiding OOM
    args.batch_size = 1
    TestImgLoader = create_dataset(args, data_path=test_path, mode='test')

    test(args, TestImgLoader, model, model_fn_test, save_path, compute_metrics)


def test_various_datasets():
    device = init()
    # load model
    model = my_model(en_feature_num=args.EN_FEATURE_NUM,
                     en_inter_num=args.EN_INTER_NUM,
                     de_feature_num=args.DE_FEATURE_NUM,
                     de_inter_num=args.DE_INTER_NUM,
                     sam_number=args.SAM_NUMBER,
                     ).to(device)

    # load checkpoint
    load_path,log_path  = load_checkpoint_light(model)
    
    # computational cost for the model
    if args.EVALUATION_COST:
        calculate_cost(model, input_size=(1, 3, 2176, 3840))

    # set logging for recording information or metrics
    set_logging(log_path)
    logging.warning(datetime.now())
    logging.warning('load model from %s' % load_path)
    compute_metrics = None
    if args.EVALUATION_TIME:
        # metric calculation may have negative impact on inference speed
        args.EVALUATION_METRIC = False
    if args.EVALUATION_METRIC:
        # load LPIPS metric
        from utils.metric import create_metrics
        compute_metrics = create_metrics(args, device=device)
    
    loss_fn = None
    model_fn_test = model_fn_decorator(loss_fn=loss_fn, device=device, mode='test')

    # Create dataset
    test_path = args.TEST_DATASET
    # Set test batch size to 1 for avoiding OOM
    args.batch_size = 1
    
    dataset_list = args.test_dataset_list
    
    for ds in dataset_list:
        sub_test_path = osp.join(test_path,ds['dataset_name'])
          # set logging for recording information or metrics
        save_path, _ = get_save_logs_path(args, dataset_name=ds['dataset_name'])
        #logging.warning('save image results to %s' % save_path)
        #logging.warning('save logger to %s' % log_path)


        args.data_transform = False
        if ds['transform']:
            args.data_transform = True
        TestImgLoader = create_dataset(args,data_path= sub_test_path, mode='test')
        args.cur_testset_name = ds['dataset_name']
        test(args, TestImgLoader, model, model_fn_test, save_path, compute_metrics)
        

    #TestImgLoader = create_dataset(args, data_path=test_path, mode='test')

    
    

if __name__ == '__main__':
    wide_test = True
    gap = args.SAVE_EPOCH
    # test multi datasets
    args.test_dataset_list =[{'dataset_name':'Postcard','transform':False},
                        {'dataset_name':'Object','transform':False},
                        {'dataset_name':'Nature20','transform':False},
                        {'dataset_name':'Real20','transform':True}
                        ]
    
    args.test_dataset_list =[{'dataset_name':'Postcard','transform':False},
                        {'dataset_name':'Object','transform':False},
                        {'dataset_name':'Wild','transform':False},
                        {'dataset_name':'Nature20','transform':False},
                        {'dataset_name':'Real20','transform':True}    
                        ]
    args.test_dataset_list =[{'dataset_name':'RR4K','transform':False}]

    args.TEST_EPOCH = 200
    test_various_datasets()
    
    