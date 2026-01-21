import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import numpy as np
from tqdm import tqdm
from datetime import datetime
import utils.metrics as Measure
from utils.utils import set_gpu, structure_loss, clip_gradient
from models.EviNet import Network 
from data import get_dataloader


def train(train_loader, model, optimizer, epoch, save_path, writer):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, supp_feats, _) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            supp_feats = supp_feats.cuda()

            s3, s2, s1, s0,s1_refined, s0_refined,loss_prob = model(images, supp_feats, y=gts, training=True)
            
            loss_s1 = structure_loss(s1, gts)
            loss_s0 = structure_loss(s0, gts)
            loss_s = loss_s0 + loss_s1
            loss_refined0 = structure_loss(s0_refined, gts)
            loss_refined1 = structure_loss(s1_refined, gts)
            loss_refined = loss_refined1+loss_refined0
            total_loss = (loss_s + loss_refined) + loss_prob
            

            # 反向传播
            total_loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            epoch_step += 1
            loss_all += total_loss.data

            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss_all:{:.4f} '.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, total_loss.data))

        # 计算平均损失
        loss_all /= epoch_step
        
        # 记录所有损失到TensorBoard
        writer.add_scalar('Loss-epoch/Loss_Avg', loss_all, global_step=epoch)

        # 模型保存
        if epoch % 20 == 0:
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch
            }, os.path.join(save_path, 'Net_epoch_{}.pth'.format(epoch)))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save({
            'state_dict': model.state_dict(),
            'epoch': epoch
        }, os.path.join(save_path, 'FDUAT_Interrupt_epoch_{}.pth'.format(epoch + 1)))
        print('Save checkpoints successfully!')
        raise

def val(test_loader, model, epoch, save_path, writer):
    """
    validation function
    """
    global best_mae, best_epoch, best_score, best_score_epoch
    WFM = Measure.WeightedFmeasure()
    SM = Measure.Smeasure()
    EM = Measure.Emeasure()
    MAE = Measure.MAE()

    model.eval()
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as pbar:
            for (image, gt, sf, _) in test_loader:
                image = image.cuda()
                sf = sf.cuda()
                _, _, _, _,s0_refine = model(image, sf, y=gt, training=False)
                res = s0_refine
                
                gt = gt.numpy().astype(np.float32).squeeze()
                gt /= (gt.max() + 1e-8)  
                
                res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)

                WFM.step(pred=res*255, gt=gt*255)
                SM.step(pred=res*255, gt=gt*255)
                EM.step(pred=res*255, gt=gt*255)
                MAE.step(pred=res*255, gt=gt*255)
              
                pbar.update()

        sm1 = SM.get_results()['sm'].round(3)
        adpem1 = EM.get_results()['em']['adp'].round(3)
        wfm1 = WFM.get_results()['wfm'].round(3)
        mae1 = MAE.get_results()['mae'].round(3)

        writer.add_scalar('Metrics/Sm', torch.tensor(sm1), global_step=epoch)
        writer.add_scalar('Metrics/adaEm', torch.tensor(adpem1), global_step=epoch)
        writer.add_scalar('Metrics/wF', torch.tensor(wfm1), global_step=epoch)
        writer.add_scalar('Metrics/MAE', torch.tensor(mae1), global_step=epoch)

        if epoch == 0:
            best_mae = mae1
            best_score = sm1 + adpem1 + wfm1
        else:
            if mae1 < best_mae:
                best_mae = mae1
                best_epoch = epoch

                state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                torch.save({
                    'state_dict': state_dict,
                    'epoch': epoch
                }, os.path.join(save_path, 'Net_epoch_bestmae.pth'))
                print('Save state_dict successfully! Best MAE epoch:{}.'.format(epoch))
            
            score = sm1 + adpem1 + wfm1
            if score > best_score:
                best_score = score
                best_score_epoch = epoch

                state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                torch.save({
                    'state_dict': state_dict,
                    'epoch': epoch
                }, os.path.join(save_path, 'Net_epoch_bestscore.pth'))
                print('Save state_dict successfully! Best score epoch:{}.'.format(epoch))

        print(f'Epoch: {epoch}, ' f'MAE: {mae1}, ' f'Sm: {sm1}, ' f'adaEm: {adpem1}, ' f'wF: {wfm1}, '
          f'bestMAE: {best_mae}, 'f'bestMaeEpoch: {best_epoch},' f'bestScore: {best_score},'
          f'bestScoreEpoch: {best_score_epoch}.')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='EviRCOD')  
    parser.add_argument('--epoch', type=int, default=150, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--dim', type=int, default=64, help='dimension of our model')
    parser.add_argument('--imgsize', type=int, default=352, help='training image size')
    parser.add_argument('--shot', type=int, default=5, help='number of referring images') 
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--num_workers', type=int, default=8, help='the number of workers in dataloader')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--data_root', type=str, default='./datasets/R2C7K', help='the path to put dataset')
    parser.add_argument('--save_root', type=str, default='./output/', help='the path to save model params and log')
    parser.add_argument('--pvt_weights', type=str, default='./models/pvtv2/pvt_v2_b4.pth', help='PVT backbone weights path')
    opt = parser.parse_args()
    print(opt)

    save_path = opt.save_root + 'saved_models/' + opt.model_name + '/'
    save_logs_path = opt.save_root + 'logs/'
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_logs_path, exist_ok=True)

    set_gpu(opt.gpu_id)
    cudnn.benchmark = True

    model = Network(opt).cuda()

    base, body = [], []
    for name, param in model.named_parameters():
        if 'backbone' in name:  
            base.append(param)   
        else:
            body.append(param)

    params_dict = [{'params': base, 'lr': opt.lr * 0.1}, {'params': body, 'lr': opt.lr}]

    optimizer = torch.optim.Adam(params_dict)
    cosine_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=150)

    print('load data...')
    train_loader = get_dataloader(opt.data_root, opt.shot, opt.imgsize, opt.batchsize, opt.num_workers, mode='train')
    val_loader = get_dataloader(opt.data_root, opt.shot, opt.imgsize, opt.batchsize, opt.num_workers, mode='val')
    total_step = len(train_loader)

    writer = SummaryWriter(save_logs_path)

    step = 0

    best_mae = 1
    best_epoch = 0
    best_score = 0.
    best_score_epoch = 0

    print("Start train...")
    for epoch in range(0, opt.epoch + 1):

        cosine_schedule.step()
        writer.add_scalar('learning_rate/base', cosine_schedule.get_lr()[0], global_step=epoch)
        writer.add_scalar('learning_rate/body', cosine_schedule.get_lr()[1], global_step=epoch)

        train(train_loader, model, optimizer, epoch, save_path, writer)
        val(val_loader, model, epoch, save_path, writer)

