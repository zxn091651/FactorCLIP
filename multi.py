import os
import glob
import random
import argparse
import warnings
import yaml
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from clip import clip
from dassl.metrics import compute_accuracy
from trainer.ours import *


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(42)
warnings.filterwarnings("default", category=UserWarning, message=".*deterministic.*")
if torch.cuda.is_available():
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
torch.use_deterministic_algorithms(True, warn_only=True)

class AvgMeter:
    """Computes and stores the average and current value."""
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"


class ImageFilter(nn.Module):
    """Filter images based on brightness threshold."""
    def __init__(self, brightness_threshold=0.01):
        super(ImageFilter, self).__init__()
        self.brightness_threshold = brightness_threshold

    def calculate_brightness(self, images):
        """Calculate average brightness of images."""
        grayscale_images = torch.mean(images, dim=1, keepdim=True)
        return grayscale_images.mean((2, 3))

    def forward(self, image_tensor):
        """Select images based on brightness criteria."""
        batch_size = image_tensor.size(0)
        brightness_values = self.calculate_brightness(image_tensor)
        
        indices_with_brightness = [
            i for i, value in enumerate(brightness_values) 
            if value >= self.brightness_threshold
        ]
        
        if len(indices_with_brightness) < batch_size:
            remaining_indices = [
                i for i in range(batch_size) 
                if i not in indices_with_brightness
            ]
            num_additional = batch_size - len(indices_with_brightness)
            additional_indices = random.sample(remaining_indices, 
                                             min(num_additional, len(remaining_indices)))
            return indices_with_brightness + additional_indices
        else:
            return random.sample(indices_with_brightness, batch_size)

class DataTrain(Dataset):
  def __init__(self,train_image_paths,train_domain,train_labels,train=True):
    self.image_path=train_image_paths
    self.domain=train_domain
    self.labels=train_labels
    self.train = train

  def __len__(self):
    return len(self.labels)

  def __getitem__(self,idx):
    if self.train:
        image = preprocess_train(Image.open(self.image_path[idx]))
    else:
        image = preprocess_val(Image.open(self.image_path[idx]))
    domain=self.domain[idx] 
    domain=torch.from_numpy(np.array(domain)) 
    label=self.labels[idx] 
    label=torch.from_numpy(np.array(label)) 
    # print("label",label)
    label_one_hot=F.one_hot(label,49)
  
    return image, domain, label, label_one_hot 

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device='cpu')
preprocess_train, preprocess_val = preprocess
with open('prompts/prompts_list_multi.txt', 'r') as file:
    prompt_list = file.readlines()

            
attri_embed = torch.from_numpy(np.load('./attributes/attribute_multi.npy')).to(device).to(torch.float32)
mask_embed = torch.from_numpy(np.load('./attributes/masks_multi.npy')).to(device).to(torch.bool)

# Remove any trailing newline characters
prompt_list = [line.strip() for line in prompt_list]
# random.shuffle(prompt_list)

repeat_transform = transforms.Compose([
    transforms.ToTensor(),
])


parser = argparse.ArgumentParser(description='Multi-dataset Domain Adaptation Training')
parser.add_argument('--source_domains', type=str, required=True,
                    help='Comma-separated source domains')
parser.add_argument('--target_domain', type=str, required=True,
                    help='Target domain')
parser.add_argument('--shots', type=int, default=1,
                    help='Number of shots per class')
parser.add_argument('--config', type=str,
                    help='Path to config file', default='configs/multi.yaml')
parser.add_argument('--data_root', type=str, default='./datasets',
                    help='Root containing office31, visda2017, stl10, and domainnet')
parser.add_argument('--output_dir', type=str, default='./experiments',
                    help='Output directory for results')
parser.add_argument('--degrees', type=int, default=5,
                    help='Degrees of rotation')
parser.add_argument('--project_dim', type=int, default=128,
                    help='Projection dimension for the model')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of training epochs')
parser.add_argument('--max_train_batches', type=int, default=0,
                    help='Limit training batches per epoch; 0 means no limit')
parser.add_argument('--max_test_batches', type=int, default=0,
                    help='Limit test batches per epoch; 0 means no limit')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

for key, value in vars(args).items():
    config[key] = value

source_domains = args.source_domains.split(',')
target_domain = args.target_domain
domains = source_domains + [target_domain]
target = domains[-1]
shots = args.shots
clip_model, preprocess = clip.load("ViT-B/32", device='cpu', degrees=args.degrees)
preprocess_train, preprocess_val = preprocess

multi_root = Path(args.data_root)
output_dir = args.output_dir

target_labels = [0, 1, 5, 6, 10, 11, 14, 17, 20, 26] + list(range(31, 37)) + list(range(39, 44)) + list(range(45, 47)) + list(range(48, 68))
known_index_dom = [0, 1, 5, 6, 10, 11, 14, 17, 20, 26] + list(range(31, 37)) + list(range(39, 44)) + list(range(45, 47))
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def image_paths(class_dir):
    paths = []
    for extension in IMAGE_EXTS:
        paths.extend(glob.glob(str(class_dir / f"**/*{extension}"), recursive=True))
        paths.extend(glob.glob(str(class_dir / f"**/*{extension.upper()}"), recursive=True))
    paths = sorted(set(paths))
    random.shuffle(paths)
    return paths


def sample_items(items, count):
    return random.sample(items, min(count, len(items)))


def source_class_dirs(source_root):
    return sorted([path for path in source_root.iterdir() if path.is_dir()], key=lambda path: path.name)


def add_source_domain(source_root, domain_id, start_label, class_names):
    domain_images = []
    domain_classes = []
    domain_labels = []
    next_label = start_label

    for class_dir in source_class_dirs(source_root):
        if next_label >= 48:
            break
        paths = image_paths(class_dir)
        if not paths:
            continue
        class_names[next_label] = class_dir.name
        for image_path in sample_items(paths, shots):
            domain_images.append(image_path)
            domain_classes.append(next_label)
            domain_labels.append(domain_id)
        next_label += 1

    return domain_images, domain_classes, domain_labels, next_label


office_amazon = multi_root / "office31" / "amazon"
visda_root = multi_root / "visda2017"
stl_root = multi_root / "stl10"
domainnet_root = multi_root / "domainnet"
test_domain_root = domainnet_root / target_domain

required_paths = [office_amazon, visda_root, stl_root, test_domain_root]
missing_paths = [str(path) for path in required_paths if not path.exists()]
if missing_paths:
    raise FileNotFoundError(
        "Missing multi-dataset directories under "
        f"{multi_root}: {', '.join(missing_paths)}"
    )

class_names4 = {}
all_classes = {}
next_label = 0
image_path_dom1, label_class_dom1, label_dom1, next_label = add_source_domain(
    office_amazon, 0, next_label, class_names4
)
image_path_dom2, label_class_dom2, label_dom2, next_label = add_source_domain(
    visda_root, 1, next_label, class_names4
)
image_path_dom3, label_class_dom3, label_dom3, next_label = add_source_domain(
    stl_root, 2, next_label, class_names4
)

if next_label < 48:
    raise RuntimeError(f"Multi-dataset sources produced {next_label} known classes; expected 48")

image_path_final = image_path_dom1 + image_path_dom2 + image_path_dom3
label_class_final = label_class_dom1 + label_class_dom2 + label_class_dom3
label_dom_final = label_dom1 + label_dom2 + label_dom3
domain_names = ["amazon", "synthetic 2D renderings", "photo"]
print("domain_names", domain_names)

test_dirs_dom = source_class_dirs(test_domain_root)
if len(test_dirs_dom) <= max(target_labels):
    raise RuntimeError(
        f"{test_domain_root} has {len(test_dirs_dom)} classes, "
        f"expected at least {max(target_labels) + 1}"
    )

test_image_path_final = []
test_label_class_final = []
for index in target_labels:
    class_dir = test_dirs_dom[index]
    if index in known_index_dom and index < 48:
        class_names4[index] = class_dir.name
    if index >= 48:
        all_classes[index] = class_dir.name
    for image_path in image_paths(class_dir):
        test_image_path_final.append(image_path)
        test_label_class_final.append(index if index <= 47 else 48)

test_label_dom_final = [3 for _ in test_image_path_final]
test_domain_names = [target_domain, target_domain, target_domain]

known_class_names = [class_names4[index] for index in range(48)]
known_classes = ",".join(known_class_names)
unknown_class_names = [all_classes[index] for index in range(48, 68)]
print(unknown_class_names)
train_prev_classnames = known_classes.split(",")
print("known_classes: ", known_classes)
class_to_attri_idx = {name: idx for idx, name in enumerate(train_prev_classnames)}

batchsize = config["batch_size"] #9
train_prev_ds=DataTrain(image_path_final,label_dom_final,label_class_final)
print(f'length of train_prev_ds: {len(train_prev_ds)}')

train_dl=DataLoader(train_prev_ds,batch_size=batchsize, num_workers=2, shuffle=True)
img_prev, domain_prev, label_prev, label_prev_one_hot = next(iter(train_dl))

domain_prev = domain_prev.to(device)

# train_prev_classnames = class_names[:54]
image_filter = ImageFilter(brightness_threshold=0.01)
W_DOMAIN = 0.33
W_ALIGN = 1.0
W_SEMANTIC_CLS = 1.0
W_REP = 0.5
W_COH = 0.2

def train_epoch(model,params, dynamic_unknown_generator, domainnames, train_loader, optimizer, lr_scheduler, step,epoch):
    loss_meter = AvgMeter()
    accuracy_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    dynamic_unknown_generator.reset_epoch_stats()

    for batch_idx, (img_prev, domain_prev, label_prev, label_one_hot_prev) in enumerate(tqdm_object):
        if args.max_train_batches and batch_idx >= args.max_train_batches:
            break
        img_prev = img_prev.to(device)
        domain_prev = domain_prev.to(device)

        random_int = epoch %3
        label_prev = label_prev.to(device)
        label_one_hot_prev = label_one_hot_prev.to(device)
        generated_unknown_images1, generation_modes, guide = dynamic_unknown_generator.generate(
            model=model,
            known_images=img_prev,
            known_labels=label_prev,
            domain_name=domainnames[random_int],
            attri_embed=attri_embed,
            mask_embed=mask_embed,
        )

        unknown_label_rank = len(train_prev_classnames)
        unknown_label = torch.full((generated_unknown_images1.shape[0],), unknown_label_rank).to(device)
        
        unknown_domain1 = torch.full((generated_unknown_images1.shape[0],), random_int).to(device)
        generated_unknown_images = generated_unknown_images1
        unknown_domains = unknown_domain1
        random_indices = image_filter(generated_unknown_images) 
        selected_images = generated_unknown_images[random_indices]
        selected_labels = unknown_label[random_indices]
        selected_domains = unknown_domains[random_indices]
        selected_modes = [generation_modes[i] if i < len(generation_modes) else "fallback" for i in random_indices]
        
        img = torch.cat((img_prev, selected_images), dim=0)
        img = img.to(device)

        label = torch.cat((label_prev, selected_labels), dim=0)
        label = label.to(device)

        domain = torch.cat((domain_prev, selected_domains), dim=0)
        domain = domain.to(device)

        (
            output,
            loss_sty,
            invariant,
            feat,
            layer_loss,
            align_loss,
            semantic_cls_loss,
            rep_loss,
            coh_loss,
        ) = model(
            img, attri_embed, mask_embed, label, domain, len(random_indices)
        )

        crossentropy_loss = (
            W_DOMAIN * loss_sty
            + W_ALIGN * (1 - F.cosine_similarity(invariant, feat, dim=1)).mean()
            + W_SEMANTIC_CLS * semantic_cls_loss
            + W_REP * rep_loss
            + W_COH * coh_loss
        )

        loss = crossentropy_loss 

        optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        count = img.size(0)
        loss_meter.update(loss.item(), count)

        acc = compute_accuracy(output, label)[0].item()
        accuracy_meter.update(acc, count)

        dynamic_unknown_generator.update_distance_stats(
            model=model,
            selected_images=selected_images,
            selected_modes=selected_modes,
            guide=guide,
        )

    dynamic_unknown_generator.log_epoch_stats(epoch)

    return loss_meter, accuracy_meter.avg

unknown_image_generator = GenerateUnknownImages(
    semantic_heads=attri_embed.shape[1],
    semantic_alpha=0.3,
).to(device)
dynamic_unknown_generator = DynamicPseudoUnknownGenerator(
    unknown_image_generator=unknown_image_generator,
    prompt_pool=prompt_list,
    known_class_names=train_prev_classnames,
    known_classes_text=known_classes,
    dynamic_batch_size=3,
    near_far_ratio=(2, 1),
    enable_dynamic=True,
    attri_embed=attri_embed,
    mask_embed=mask_embed,
    class_to_attri_idx=class_to_attri_idx,
    num_semantic_heads=attri_embed.shape[1],
    offline_attri_alpha=0.2,
)

train_classnames = train_prev_classnames + ['unknown']
print(f'length of train_classnames : {len(train_classnames)}')
domains_open = ["image",'2D rendering','grayscale view']
train_model = CustomCLIP(train_classnames, domains_open, clip_model,config,project=True)

for param in train_model.parameters():
            param.requires_grad_(False)
for p in train_model.cross_attention.parameters():
    p.requires_grad= True
train_model.projector.requires_grad = True
for p in train_model.class_semantic_builder.parameters():
    p.requires_grad = True
train_model.unknown_prompt_ctx.requires_grad = True
params = [
            {"params": train_model.projector.parameters(),'lr' : config["projector_lr"]},
            {"params": train_model.cross_attention.parameters(),'lr' : config["cross_attention_lr"]},
            {"params": train_model.class_semantic_builder.parameters(), 'lr': config["prompt_lr"]},
            {"params": [train_model.unknown_prompt_ctx], 'lr': config["prompt_lr"]},
        ]
optimizer = torch.optim.AdamW(params,  weight_decay=config["weight_decay"])

warmup_epochs = 1

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=1, factor=0.8
        )
num_epochs = args.epochs
warmup_period = 1
num_steps = len(train_dl) * num_epochs - warmup_period

all_params=[]
for group in optimizer.param_groups:
        all_params += group['params']
scaler = GradScaler() 

test_ds=DataTrain(test_image_path_final,test_label_dom_final,test_label_class_final, train=False)
print(len(test_ds))
test_dl=DataLoader(test_ds,batch_size=32, num_workers=4, shuffle=True)
test_img, test_domain, test_label, test_label_one_hot = next(iter(test_dl))

step = "epoch"
best_acc = 0
best_closed_set_acc = 0
best_open_set_acc = 0
best_avg_acc = 0
accuracy_file_path = f"{output_dir}/{domains[-1]}/{target}_{shots}.txt"  
accuracy_dir = os.path.dirname(accuracy_file_path)
if not os.path.exists(accuracy_dir):
    os.makedirs(accuracy_dir)
accuracy_file = open(accuracy_file_path, "w")
torch.autograd.set_detect_anomaly(True)

test_model = CustomCLIP(train_classnames, test_domain_names, clip_model,config,project=True).to(device)
train_model = train_model.to(device)
for epoch in range(num_epochs):
    closed_set_features = []    
    closed_set_labels = []  # To store labels of closed-set samples
    open_set_features = []
    print(f"Epoch: {epoch + 1}")
    train_model.train()
    train_loss, train_acc = train_epoch(
        train_model,
        all_params,
        dynamic_unknown_generator,
        domain_names,
        train_dl,
        optimizer,
        lr_scheduler,
        step,
        epoch,
    )
    print(f"epoch {epoch+1} : training accuracy: {train_acc}")

    save_path = f"{output_dir}/{domains[-1]}/{target}_{shots}_temp.pth"

    torch.save(obj=train_model.state_dict(), f=save_path)
    
    test_model.load_state_dict(torch.load(save_path))

    with torch.no_grad():
        test_probs_all = torch.empty(0).to(device)
        test_labels_all = torch.empty(0).to(device)
        test_class_all = torch.empty(0).to(device)
        test_tqdm_object = tqdm(test_dl, total=len(test_dl))

        total_correct_a = 0
        total_samples_a = 0
        total_correct_b = 0
        total_samples_b = 0
        
        for batch_idx, (test_img, test_domain, test_label, test_label_one_hot) in enumerate(test_tqdm_object):
            if args.max_test_batches and batch_idx >= args.max_test_batches:
                break
            test_img = test_img.to(device)
            test_domain =test_domain.to(device)
            test_label = test_label.to(device)
            test_label_one_hot = test_label_one_hot.to(device)
            
            # with profile(with_flops=True) as prof:
            test_output,_ = test_model(
                test_img.to(device),
                attri_embed,
                mask_embed,
                test_label,
                test_domain,
            )

            predictions = torch.argmax(test_output, dim=1)
            class_a_mask = (test_label <= 47) 
            class_b_mask = (test_label > 47)

            correct_predictions_a = (predictions[class_a_mask] == test_label[class_a_mask]).sum().item()
            correct_predictions_b = (predictions[class_b_mask] == test_label[class_b_mask]).sum().item()
            
            total_correct_a += correct_predictions_a
            total_samples_a += class_a_mask.sum().item()
            
            total_correct_b += correct_predictions_b
            total_samples_b += class_b_mask.sum().item()

        closed_set_accuracy = total_correct_a / total_samples_a if total_samples_a > 0 else 0.0
        closed_set_acc = closed_set_accuracy*100
        open_set_accuracy = total_correct_b / total_samples_b if total_samples_b > 0 else 0.0
        open_set_acc = open_set_accuracy*100

        average_acc = (
            (2 * closed_set_acc * open_set_acc) / (closed_set_acc + open_set_acc)
            if (closed_set_acc + open_set_acc) > 0
            else 0.0
        )

        print(f"Closed Set Accuracy: {closed_set_acc:.2f}%")
        print(f"Open Set Accuracy: {open_set_acc:.2f}%")
        print(f"Harmonic Score: {average_acc:.2f}%")
        
        # Write results
        accuracy_file.write(f"Epoch {epoch+1}\n")
        accuracy_file.write(f"Closed Set Accuracy: {closed_set_acc:.2f}%\n")
        accuracy_file.write(f"Open Set Accuracy: {open_set_acc:.2f}%\n") 
        accuracy_file.write(f"Harmonic Score: {average_acc:.2f}%\n")
        accuracy_file.write("-" * 40 + "\n")
        accuracy_file.flush()

        if average_acc > best_avg_acc:
            best_closed_set_acc = closed_set_acc
            best_open_set_acc = open_set_acc
            best_avg_acc = average_acc
            TEST_MODEL_PATH = Path(f"{output_dir}")
            TEST_MODEL_PATH.mkdir(parents=True, exist_ok=True)
            TEST_MODEL_NAME = f"{target}ours.pth"
            TEST_MODEL_SAVE_PATH = TEST_MODEL_PATH / TEST_MODEL_NAME
            print(f"Saving test_model with best harmonic score to: {TEST_MODEL_SAVE_PATH}")
            torch.save(obj=test_model.state_dict(), f=TEST_MODEL_SAVE_PATH) 
            
            print(f"New best harmonic score: {best_avg_acc:.2f}%")
accuracy_file.write(f"End Closed Score: {best_closed_set_acc:.2f}%\n")
accuracy_file.write(f"End Harmonic Score: {best_avg_acc:.2f}%\n")

print(f"\nTraining completed. Best harmonic score: {best_avg_acc:.2f}%")
print(f"Results saved to: {accuracy_file_path}")
accuracy_file.close()
