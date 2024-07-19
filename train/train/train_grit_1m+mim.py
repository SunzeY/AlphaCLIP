import torch
import alpha_clip
from utils import concat_all_gather, is_dist_avail_and_initialized, accuracy
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from dataset.imagenet_s_test import Imagenet_S
from dataset.mask_image_test import COCO_Masked_Test
from dataset.mask_image import ImageNet_Masked
from dataset.alpha_grit import Alpha_GRIT
from torch.utils.data.distributed import DistributedSampler
from scheduler import cosine_lr
import argparse
import os
import subprocess
import collections
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import loralib as lora
import numpy as np

simple_templates = [
    'a photo of a {}.'
]

def zeroshot_classifier(classnames, templates, model, local_rank=0):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames, disable=(local_rank != 0)):
            texts = [template.format(classname) for template in templates] #format with class
            texts = alpha_clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

class CLIP_Clean_Train():
    def __init__(self, local_rank=0, lr=4e-5, weigth_decay=0.02, log_scale=4.6052, lora_rank=-1, common_pair=0.0, para_gamma=0.01, exp_name="auto", warmup_length=200):
        self.local_rank = local_rank
        if lora_rank == -1:
            self.model, _ = alpha_clip.load("ViT-B/16", device='cpu', lora_adapt=False, rank=-1)
        else:
            self.model, _ = alpha_clip.load("ViT-B/16", device='cpu', lora_adapt=True, rank=lora_rank)
        torch.cuda.set_device(device=f'cuda:{local_rank}')
        self.model = self.model.float().cuda()
        self.batch_size = 64 * 8
        self.num_epoch = 10
        self.lr = lr
        if exp_name == "auto":
            self.logdir = f"runs_com_bf/grit_1m+mim46w/lr={lr}_wd={weigth_decay}_wl={warmup_length}_logs={log_scale}_RN50X16_lora={lora_rank}_cp={common_pair}_para_gamma={para_gamma}_4xb_e6"
        else:
            self.logdir = exp_name
        self.ckptdir = self.logdir + "/ckpt/"
        os.makedirs(self.ckptdir, exist_ok=True)
        self.writer = SummaryWriter(self.logdir)
        self.model.visual = torch.nn.parallel.DistributedDataParallel(self.model.visual, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        # logit scale
        self.model.logit_scale = torch.nn.Parameter(torch.ones([]) * log_scale)
        conv_opt_paras = []
        other_opt_paras = []
        if lora_rank != -1: # use lora
            lora.mark_only_lora_as_trainable(self.model)
            for k, v in self.model.named_parameters():
                if v.requires_grad:
                    other_opt_paras.append(v)
                elif "conv1_alpha" in k:
                    v.requires_grad_(True)
                    conv_opt_paras.append(v)
        else: # normal to not use lora
            for k, v in self.model.named_parameters():
                v.requires_grad_(False)
            for k, v in self.model.visual.named_parameters():
                v.requires_grad_(True)
                if "conv1_alpha" in k:
                    conv_opt_paras.append(v)
                else:
                    other_opt_paras.append(v) 
        self.optimizer = optim.AdamW(
            [
                {"params": conv_opt_paras, "lr": self.lr},
                {"params": other_opt_paras, "lr": self.lr * para_gamma}
            ],
        )
        
        self.para_gamma = para_gamma
        
        self.scaler = torch.cuda.amp.grad_scaler.GradScaler()

    def inference(self, images, masks, texts):
        image_features = self.model.visual(images, masks)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = self.model.encode_text(texts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        image_feat_all = concat_all_gather(image_features)
        text_feat_all = concat_all_gather(text_features)
        
        sim_i2t = torch.matmul(image_features, text_feat_all.T)
        sim_t2i = torch.matmul(image_feat_all, text_features.T)
        sim_t2i = sim_t2i.T

        sim_i2t = self.model.logit_scale.exp() * sim_i2t
        sim_t2i = self.model.logit_scale.exp() * sim_t2i

        if is_dist_avail_and_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
        bs = images.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            images.device
        )
        # DEBUG PRINT contrastive loss accross batch.
        # print(targets)
        # print(sim_i2t.shape)
        # print(sim_t2i.shape)
        loss_itc = (
                F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
                + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
            ) / 2
        return loss_itc
    
    def train_epoch(self, dataloader, test_loaders, epoch, start_iter=0, amp=False):
        running_loss = 0.0
        num_batches_per_epoch = len(dataloader)
        for i, (images, masks, texts) in enumerate(tqdm(dataloader, disable=(dist.get_rank() != 0))):
            step = num_batches_per_epoch * epoch + i
            if step < start_iter:
                continue
            self.optimizer.zero_grad()
            self.scheduler(step)
            images = images.cuda()
            masks = masks.cuda()
            texts = alpha_clip.tokenize(texts).cuda()
            if amp:
                with torch.cuda.amp.autocast():
                    loss = self.inference(images, masks, texts)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self.inference(images, masks, texts)
                loss.backward()
                self.optimizer.step()
            running_loss += loss.item()
            batch_num = i
            
            # average loss of different proc
            loss = running_loss / 400
            running_loss = 0.0
            loss = torch.tensor(loss).cuda()
            dist.all_reduce(loss)
            loss = loss.item() / torch.distributed.get_world_size()
            if step % 100 == 0:
                if dist.get_rank() == 0:
                    self.writer.add_scalar("hyper/lr", self.optimizer.param_groups[0]['lr'], step)
                    self.writer.add_scalar("logit_scale/train", self.model.logit_scale.item(), step)
                    print("=====================================")
                    print(f"train lr (alpha conv) step {step}: {self.optimizer.param_groups[0]['lr']}")
                    print(f"train lr (other layer) step {step}: {self.optimizer.param_groups[1]['lr']}")
                    print(f"train logit_scale step {step}: {self.model.logit_scale.item()}")
                    print(f"train loss step {step}: {loss}")
                    print("=====================================")
                    self.writer.add_scalar("Loss/train", loss, step)
                    if step % 100 == 0 and step != 0:
                        # MOD no checkpoint for now
                        # pass
                        torch.save(self.model.visual.state_dict(), self.ckptdir + f'iter_{step}.pth')
                    # torch.save(lora.lora_state_dict(self.model.visual), self.ckptdir + f'iter_{step}.pth')
                with torch.no_grad():
                    self.model.visual.eval()
                    for test_name, test_loader in test_loaders.items():
                        self.text_embeddings = zeroshot_classifier(test_loader.dataset.classes, simple_templates, self.model, self.local_rank)
                        temp_corr_dict = self.test_epoch(test_loader)
                        if is_dist_avail_and_initialized():
                            output = [None] * dist.get_world_size()
                            dist.all_gather_object(output, temp_corr_dict)
                        else:
                            output = [temp_corr_dict]
                        if dist.get_rank() == 0:
                            final_dict = dict()
                            for dic in output:
                                for k, v in dic.items():
                                    if k not in final_dict.keys():
                                        final_dict[k] = v
                                    else:
                                        final_dict[k][0] += v[0]
                                        final_dict[k][1] += v[1]
                                        final_dict[k][2] += v[2]
                            acc1 = 0.0
                            acc5 = 0.0
                            num_class = 0
                            for v in final_dict.values():
                                acc1 += v[1] / v[0]
                                acc5 += v[2] / v[0]
                                num_class += 1
                            acc1 = acc1 / num_class
                            acc5 = acc5 / num_class
                            # print(final_dict)
                            print("=====================================")
                            print(f"test {test_name} acc-1 step {step}: {acc1}")
                            print(f"test {test_name} acc-5 step {step}: {acc5}")
                            print("=====================================")
                            self.writer.add_scalar(f"{test_name}_Acc1/test", acc1, step)
                            self.writer.add_scalar(f"{test_name}_Acc5/test", acc5, step)
                    self.model.visual.train()
        return running_loss / batch_num

    @torch.no_grad()
    def test_epoch(self, dataloader):
        temp_corr_dict = dict()
        for i, (images, masks, target) in enumerate(tqdm(dataloader, disable=(dist.get_rank() != 0))):
            images = images.cuda()
            target = target.cuda()
            image_features = self.model.visual(images, masks)
            # image_features = self.model.visual(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            score = torch.matmul(image_features, self.text_embeddings)
            pred = score.topk(1, dim=1)[1].squeeze(dim=1)
            pred_5 = score.topk(5, dim=1)[1].squeeze(dim=1)
            for i in range(target.shape[0]):
                if target[i].item() not in temp_corr_dict:
                    temp_corr_dict[target[i].item()] = [0, 0, 0]
                temp_corr_dict[target[i].item()][0] += 1
                if target[i].item() == pred[i].item():
                    temp_corr_dict[target[i].item()][1] += 1
                if target[i].item() in pred_5[i].tolist():
                    temp_corr_dict[target[i].item()][2] += 1
        return temp_corr_dict
    
    def test(self, epoch=0):
        self.model.visual.eval()
        testset = Imagenet_S()
        self.text_embeddings = zeroshot_classifier(testset.classes, simple_templates, self.model, self.local_rank)
        sampler = DistributedSampler(dataset=testset, shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, sampler=sampler, num_workers=16, pin_memory=True)
        with torch.no_grad():
            temp_corr_dict = self.test_epoch(testloader)
            if is_dist_avail_and_initialized():
                output = [None] * dist.get_world_size()
                dist.all_gather_object(output, temp_corr_dict)
            else:
                output = [temp_corr_dict]
            if self.local_rank == 0:
                final_dict = dict()
                for dic in output:
                    for k, v in dic.items():
                        if k not in final_dict.keys():
                            final_dict[k] = v
                        else:
                            final_dict[k][0] += v[0]
                            final_dict[k][1] += v[1]
                            final_dict[k][2] += v[2]
                acc1 = 0.0
                acc5 = 0.0
                num_class = 0
                for v in final_dict.values():
                    acc1 += v[1] / v[0]
                    acc5 += v[2] / v[0]
                    num_class += 1
                acc1 = acc1 / num_class
                acc5 = acc5 / num_class
                print("=====================================")
                print(f"test mean of per class acc-1 step 0: {acc1}")
                print(f"test mean of per class acc-5 step 0: {acc5}")
                print("=====================================")
        return
    
    def train(self, common_pair=False, resume=False, amp=False, warmup_length=200):
        testset_image_s = Imagenet_S(hi_res=False)
        testset_coco = COCO_Masked_Test(hi_res=False)
        # trainset = Alpha_GRIT(common_pair=common_pair)
        trainset = torch.utils.data.ConcatDataset(datasets=[Alpha_GRIT(hi_res=False), ImageNet_Masked()])
        # trainset = Alpha_GRIT(ids_file='grit_20m_ids.pkl', root_pth='grit-20m/', common_pair=common_pair)
        test_loaders = dict()
        for name, testset in zip(['COCO', 'Imagenet-S'], [testset_coco, testset_image_s]):
            test_sampler = DistributedSampler(dataset=testset, shuffle=True)
            test_loader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, sampler=test_sampler, num_workers=16, pin_memory=True)
            test_loaders[name] = test_loader   
        train_sampler = DistributedSampler(dataset=trainset, shuffle=True)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, sampler=train_sampler, num_workers=32, pin_memory=True)
        # train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=16, pin_memory=True)
        # test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=16, pin_memory=True)
        self.scheduler = cosine_lr(self.optimizer, base_lr=self.lr, warmup_length=warmup_length, steps=self.num_epoch * len(train_loader), para_gamma=self.para_gamma)
        start_epoch = 0
        resume_iter = 0
        if resume == True and os.listdir(self.ckptdir) != []:
            resume_pth = os.listdir(self.ckptdir)[-1]
            resume_iter = int(resume_pth[5:-4])
            start_epoch = resume_iter // len(train_loader)
            map_location = {'cuda:0': f'cuda:{self.local_rank}'}
            self.model.visual.load_state_dict(torch.load(os.path.join(self.ckptdir, resume_pth), map_location=map_location))
            print(f"load resumed checkpoint: {resume_pth}")  
        for epoch in range(start_epoch, self.num_epoch):
            # train
            loss = self.train_epoch(train_loader, test_loaders, epoch, start_iter=resume_iter, amp=amp)

def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29511"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    return rank % num_gpus

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--lr', default=4e-5, type=float, help='lr.')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='wd.')
    parser.add_argument('--log_scale', default=4.6052, type=float, help='clip temperature log scale.')
    parser.add_argument('--lora_rank', default=-1, type=int, help='lora rank (-1 to not use lora).')
    parser.add_argument('--common_pair', default=0.0, type=float, help='propotion to use image with all 1 alpha and whole caption.')
    parser.add_argument('--para_gamma', default=0.01, type=float, help='para_gamma of other parameters')
    parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.")
    parser.add_argument("--amp", action="store_true", help="bf16 taining.")
    parser.add_argument("--exp_name", default="auto", type=str, help="specify experiment name.")
    parser.add_argument("--warmup_length", default=200, type=int, help="warmup_length.")
    args = parser.parse_args()
    local_rank = setup_distributed()
    trainer = CLIP_Clean_Train(
        local_rank=local_rank, 
        lr=args.lr, 
        weigth_decay=args.weight_decay, 
        log_scale=args.log_scale, 
        lora_rank=args.lora_rank, 
        common_pair=args.common_pair, 
        para_gamma=args.para_gamma,
        exp_name=args.exp_name,
        warmup_length=args.warmup_length,
        )
    trainer.train(common_pair=args.common_pair, resume=args.resume, amp=args.amp, warmup_length=args.warmup_length)
