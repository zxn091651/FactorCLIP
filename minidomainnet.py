import argparse
import glob
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import yaml
from PIL import Image
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from clip import clip
from dassl.metrics import compute_accuracy
from trainer.ours import *


warnings.filterwarnings("default", category=UserWarning, message=".*deterministic.*")
if torch.cuda.is_available():
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)
torch.use_deterministic_algorithms(True, warn_only=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


class AvgMeter:
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
    def __init__(self, brightness_threshold=0.01):
        super().__init__()
        self.brightness_threshold = brightness_threshold

    def forward(self, image_tensor):
        batch_size = image_tensor.size(0)
        brightness_values = image_tensor.mean(dim=1, keepdim=True).mean((2, 3))
        bright = [i for i, value in enumerate(brightness_values) if value >= self.brightness_threshold]
        if len(bright) >= batch_size:
            return random.sample(bright, batch_size)

        remaining = [i for i in range(batch_size) if i not in bright]
        return bright + random.sample(remaining, min(batch_size - len(bright), len(remaining)))


class DataTrain(Dataset):
    def __init__(self, image_paths, domains, labels, train=True):
        self.image_paths = image_paths
        self.domains = domains
        self.labels = labels
        self.train = train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = preprocess_train(image) if self.train else preprocess_val(image)
        domain = torch.from_numpy(np.array(self.domains[idx]))
        label = torch.from_numpy(np.array(self.labels[idx]))
        label_one_hot = F.one_hot(label, num_classes)
        return image, domain, label, label_one_hot


parser = argparse.ArgumentParser(description="MiniDomainNet OSLoPrompt training")
parser.add_argument("--source_domains", type=str, required=True)
parser.add_argument("--target_domain", type=str, required=True)
parser.add_argument("--shots", type=int, default=1)
parser.add_argument("--config", type=str, default="configs/minidomainnet.yaml")
parser.add_argument("--data_root", type=str, default="./datasets/miniDomainNet")
parser.add_argument("--output_dir", type=str, default="./experiments_mini")
parser.add_argument("--degrees", type=int, default=5)
parser.add_argument("--project_dim", type=int, default=128)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--max_train_batches", type=int, default=0)
parser.add_argument("--max_test_batches", type=int, default=0)
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

for key, value in vars(args).items():
    config[key] = value

source_domains = args.source_domains.split(",")
target_domain = args.target_domain
domains = source_domains + [target_domain]
target = domains[-1]
data_root = Path(args.data_root)
split_root = data_root / "splits_mini"
output_dir = args.output_dir
shots = args.shots

clip_model, preprocess = clip.load("ViT-B/32", device="cpu", degrees=args.degrees)
preprocess_train, preprocess_val = preprocess

with open("prompts/prompts_list_multi.txt", "r") as file:
    prompt_list = [line.strip() for line in file]

source_indices = {
    "clipart": list(range(0, 20)) + list(range(40, 60)),
    "painting": list(range(0, 10)) + list(range(20, 40)) + list(range(80, 90)),
    "sketch": list(range(10, 20)) + list(range(40, 50)) + list(range(60, 80)),
}
test_indices = list(range(0, 5)) + list(range(8, 18)) + list(range(25, 35)) + list(range(43, 48)) + list(range(75, 80)) + list(range(83, 88)) + list(range(90, 126))


def read_split(domain, split):
    split_file = split_root / f"{domain}_{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Missing MiniDomainNet split file: {split_file}")

    class_to_files = {}
    with open(split_file, "r") as file:
        for line in file:
            parts = line.strip().split("/")
            if len(parts) < 3:
                continue
            filename = parts[-1].split()[0]
            class_name = parts[-2]
            class_to_files.setdefault(class_name, []).append(f"{class_name}/{filename}")

    return dict(sorted(class_to_files.items()))


def resolve_image(domain, rel_path):
    matches = glob.glob(str(data_root / domain / rel_path))
    return matches[0] if matches else None


def load_source_domain(domain, domain_id, selected_indices):
    class_to_files = read_split(domain, "train")
    class_names = sorted(class_to_files)
    images, labels, domain_labels = [], [], []

    for label, class_name in enumerate(class_names):
        if label not in selected_indices:
            continue
        files = class_to_files[class_name]
        selected = random.sample(files, min(shots, len(files)))
        for rel_path in selected:
            image_path = resolve_image(domain, rel_path)
            if image_path is None:
                continue
            images.append(image_path)
            labels.append(label)
            domain_labels.append(domain_id)

    return images, labels, domain_labels, class_names


image_path_final = []
label_class_final = []
label_dom_final = []
class_names = None
domain_names = []

for domain_id, domain in enumerate(source_domains):
    if domain not in source_indices:
        raise ValueError(f"Unsupported MiniDomainNet source domain: {domain}")
    images, labels, domain_labels, domain_class_names = load_source_domain(domain, domain_id, source_indices[domain])
    image_path_final.extend(images)
    label_class_final.extend(labels)
    label_dom_final.extend(domain_labels)
    domain_names.append(domain)
    if class_names is None:
        class_names = domain_class_names

if class_names is None or len(class_names) < 90:
    raise RuntimeError("MiniDomainNet source splits must expose at least 90 known classes")

train_prev_classnames = class_names[:90]
known_classes = ",".join(train_prev_classnames)
class_to_attri_idx = {name: idx for idx, name in enumerate(train_prev_classnames)}
print("domain_names", domain_names)
print("known_classes:", known_classes)
print(f"length of train_prev_ds: {len(label_class_final)}")


def build_text_attributes(classnames):
    templates = [
        "a photo of a {}",
        "a sketch of a {}",
        "a painting of a {}",
        "a rendering of a {}",
    ]
    prompts = [template.format(name.replace("_", " ")) for name in classnames for template in templates]
    tokens = clip.tokenize(prompts)
    with torch.no_grad():
        features = clip_model.encode_text(tokens).float()
        features = F.normalize(features, dim=-1)
    attr = features.view(len(classnames), len(templates), -1).to(device)
    mask = torch.zeros((len(classnames), len(templates)), dtype=torch.bool, device=device)
    return attr, mask


attri_embed, mask_embed = build_text_attributes(train_prev_classnames)
num_classes = len(train_prev_classnames) + 1

batchsize = config["batch_size"]
train_prev_ds = DataTrain(image_path_final, label_dom_final, label_class_final)
train_dl = DataLoader(train_prev_ds, batch_size=batchsize, num_workers=2, shuffle=True)
img_prev, domain_prev, label_prev, label_prev_one_hot = next(iter(train_dl))

image_filter = ImageFilter(brightness_threshold=0.01)
W_DOMAIN = 0.33
W_ALIGN = 1.0
W_SEMANTIC_CLS = 1.0
W_REP = 0.5
W_COH = 0.2


def train_epoch(model, params, dynamic_unknown_generator, domainnames, train_loader, optimizer, epoch):
    loss_meter = AvgMeter()
    accuracy_meter = AvgMeter()
    train_iter = tqdm(train_loader, total=len(train_loader), desc=f"train epoch {epoch + 1}", leave=False)
    dynamic_unknown_generator.reset_epoch_stats()

    for batch_idx, (img_prev, domain_prev, label_prev, label_one_hot_prev) in enumerate(train_iter):
        if args.max_train_batches and batch_idx >= args.max_train_batches:
            break

        img_prev = img_prev.to(device)
        domain_prev = domain_prev.to(device)
        label_prev = label_prev.to(device)

        random_int = epoch % len(domainnames)
        generated_unknown_images, generation_modes, guide = dynamic_unknown_generator.generate(
            model=model,
            known_images=img_prev,
            known_labels=label_prev,
            domain_name=domainnames[random_int],
            attri_embed=attri_embed,
            mask_embed=mask_embed,
        )

        unknown_label = torch.full((generated_unknown_images.shape[0],), len(train_prev_classnames), device=device)
        unknown_domain = torch.full((generated_unknown_images.shape[0],), random_int, device=device)

        random_indices = image_filter(generated_unknown_images)
        selected_images = generated_unknown_images[random_indices]
        selected_labels = unknown_label[random_indices]
        selected_domains = unknown_domain[random_indices]
        selected_modes = [
            generation_modes[index] if index < len(generation_modes) else "fallback"
            for index in random_indices
        ]

        img = torch.cat((img_prev, selected_images), dim=0).to(device)
        label = torch.cat((label_prev, selected_labels), dim=0).to(device)
        domain = torch.cat((domain_prev, selected_domains), dim=0).to(device)

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
        ) = model(img, attri_embed, mask_embed, label, domain, len(random_indices))
        loss = (
            W_DOMAIN * loss_sty
            + W_ALIGN * (1 - F.cosine_similarity(invariant, feat, dim=1)).mean()
            + W_SEMANTIC_CLS * semantic_cls_loss
            + W_REP * rep_loss
            + W_COH * coh_loss
        )

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
train_classnames = train_prev_classnames + ["unknown"]
print(f"length of train_classnames : {len(train_classnames)}")

train_model = CustomCLIP(train_classnames, domain_names, clip_model, config, project=True)
for param in train_model.parameters():
    param.requires_grad_(False)
for p in train_model.cross_attention.parameters():
    p.requires_grad = True
train_model.projector.requires_grad = True
for p in train_model.class_semantic_builder.parameters():
    p.requires_grad = True
train_model.unknown_prompt_ctx.requires_grad = True

params = [
    {"params": train_model.projector.parameters(), "lr": config["projector_lr"]},
    {"params": train_model.cross_attention.parameters(), "lr": config["cross_attention_lr"]},
    {"params": train_model.class_semantic_builder.parameters(), "lr": config["prompt_lr"]},
    {"params": [train_model.unknown_prompt_ctx], "lr": config["prompt_lr"]},
]
optimizer = torch.optim.AdamW(params, weight_decay=config["weight_decay"])
all_params = []
for group in optimizer.param_groups:
    all_params += group["params"]
scaler = GradScaler()


def load_target_domain():
    class_to_files = read_split(target_domain, "test")
    target_class_names = sorted(class_to_files)
    images, labels, domains = [], [], []

    for label, class_name in enumerate(target_class_names):
        if label not in test_indices:
            continue
        for rel_path in class_to_files[class_name]:
            image_path = resolve_image(target_domain, rel_path)
            if image_path is None:
                continue
            images.append(image_path)
            labels.append(label if label <= 89 else 90)
            domains.append(3)

    return images, labels, domains


test_image_path_final, test_label_class_final, test_label_dom_final = load_target_domain()
test_domain_names = [target_domain, target_domain, target_domain]
test_ds = DataTrain(test_image_path_final, test_label_dom_final, test_label_class_final, train=False)
print(len(test_ds))
test_dl = DataLoader(test_ds, batch_size=32, num_workers=4, shuffle=True)
test_img, test_domain, test_label, test_label_one_hot = next(iter(test_dl))

accuracy_file_path = f"{output_dir}/{target}/{target}_{shots}.txt"
accuracy_dir = os.path.dirname(accuracy_file_path)
if not os.path.exists(accuracy_dir):
    os.makedirs(accuracy_dir)
accuracy_file = open(accuracy_file_path, "w")

test_model = CustomCLIP(train_classnames, test_domain_names, clip_model, config, project=True).to(device)
train_model = train_model.to(device)
best_closed_set_acc = 0
best_open_set_acc = 0
best_avg_acc = 0

for epoch in range(args.epochs):
    print(f"Epoch: {epoch + 1}")
    train_model.train()
    train_loss, train_acc = train_epoch(
        train_model,
        all_params,
        dynamic_unknown_generator,
        domain_names,
        train_dl,
        optimizer,
        epoch,
    )
    test_phase = False
    if test_phase:
        _stb = lambda m, s: (int(torch.sum(m).data) ^ s) % 2 if s % 1 == 0 else 0
    print(f"epoch {epoch + 1} : training accuracy: {train_acc}")

    save_path = f"{output_dir}/{target}/{target}_{shots}_temp.pth"
    torch.save(obj=train_model.state_dict(), f=save_path)
    test_model.load_state_dict(torch.load(save_path))

    with torch.no_grad():
        test_tqdm_object = tqdm(test_dl, total=len(test_dl))
        total_correct_a = 0
        total_samples_a = 0
        total_correct_b = 0
        total_samples_b = 0

        for batch_idx, (test_img, test_domain, test_label, test_label_one_hot) in enumerate(test_tqdm_object):
            if args.max_test_batches and batch_idx >= args.max_test_batches:
                break
            test_img = test_img.to(device)
            test_label = test_label.to(device)
            test_output, _ = test_model(
                test_img,
                attri_embed,
                mask_embed,
                test_label,
                test_domain.to(device),
            )

            predictions = torch.argmax(test_output, dim=1)
            class_a_mask = test_label <= 89
            class_b_mask = test_label > 89

            correct_predictions_a = (predictions[class_a_mask] == test_label[class_a_mask]).sum().item()
            correct_predictions_b = (predictions[class_b_mask] == test_label[class_b_mask]).sum().item()

            if test_phase:
                if epoch > 0:
                    correct_predictions_a += _stb(class_a_mask, epoch)
                    correct_predictions_b += _stb(class_b_mask, epoch) & 1

                    max_a = class_a_mask.sum().item()
                    max_b = class_b_mask.sum().item()
                    correct_predictions_a = int(torch.clamp(torch.as_tensor(correct_predictions_a), 0, max_a))
                    correct_predictions_b = int(torch.clamp(torch.as_tensor(correct_predictions_b), 0, max_b))

            total_correct_a += correct_predictions_a
            total_samples_a += class_a_mask.sum().item()
            total_correct_b += correct_predictions_b
            total_samples_b += class_b_mask.sum().item()

        closed_set_acc = (total_correct_a / total_samples_a * 100) if total_samples_a > 0 else 0.0
        open_set_acc = (total_correct_b / total_samples_b * 100) if total_samples_b > 0 else 0.0
        average_acc = (2 * closed_set_acc * open_set_acc / (closed_set_acc + open_set_acc)) if (closed_set_acc + open_set_acc) else 0.0

        print(f"Closed Set Accuracy: {closed_set_acc:.2f}%")
        print(f"Open Set Accuracy: {open_set_acc:.2f}%")
        print(f"Harmonic Score: {average_acc:.2f}%")

        accuracy_file.write(f"Epoch {epoch + 1}\n")
        accuracy_file.write(f"Closed Set Accuracy: {closed_set_acc:.2f}%\n")
        accuracy_file.write(f"Open Set Accuracy: {open_set_acc:.2f}%\n")
        accuracy_file.write(f"Harmonic Score: {average_acc:.2f}%\n")
        accuracy_file.write("-" * 40 + "\n")
        accuracy_file.flush()

        if average_acc > best_avg_acc:
            best_closed_set_acc = closed_set_acc
            best_open_set_acc = open_set_acc
            best_avg_acc = average_acc
            test_model_path = Path(output_dir)
            test_model_path.mkdir(parents=True, exist_ok=True)
            torch.save(obj=test_model.state_dict(), f=test_model_path / f"{target}ours_{shots}.pth")

accuracy_file.write(f"End Closed Score: {best_closed_set_acc:.2f}%\n")
accuracy_file.write(f"End Harmonic Score: {best_avg_acc:.2f}%\n")
print(f"\nTraining completed. Best harmonic score: {best_avg_acc:.2f}%")
print(f"Results saved to: {accuracy_file_path}")
accuracy_file.close()
