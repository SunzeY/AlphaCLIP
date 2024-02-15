import torch
import alpha_clip
from tqdm import tqdm
from imagenet_s import Imagenet_S

model, preprocess = alpha_clip.load("ViT-L/14@336px", alpha_vision_ckpt_pth="../../clip_l14@336_grit_20m_4xe.pth")

def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = alpha_clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

dataset = Imagenet_S(hi_res=True)
loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=2)

imagenet_templates = [
    'a photo of a {}.'
]

zeroshot_weights = zeroshot_classifier(dataset.classes, imagenet_templates)
temp_corr_dict = dict()

with torch.no_grad():
    for i, (images, alpha, target) in enumerate(tqdm(loader)):
        images = images.cuda()
        alpha = alpha.cuda()
        target = target.cuda()
        # predict
        image_features = model.encode_image(images, alpha)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        score = 100. * image_features @ zeroshot_weights

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

acc1 = 0.0
acc5 = 0.0
num_class = 0
for v in temp_corr_dict.values():
    if v[0] == 0: continue
    acc1 += v[1] / v[0]
    acc5 += v[2] / v[0]
    num_class += 1
acc1 = acc1 / num_class * 100
acc5 = acc5 / num_class * 100

print(f"Top-1 accuracy: {acc1:.2f}")
print(f"Top-5 accuracy: {acc5:.2f}")
