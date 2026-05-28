
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple, Union
import random

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import einops
import torch
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image

# CLIP and Diffusion imports
from clip import clip
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPVisionModel, CLIPImageProcessor
from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionInstructPix2PixPipeline,
    DPMSolverMultistepScheduler
)
import warnings
warnings.filterwarnings('error', category=UserWarning, message='.*deterministic.*')

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    # Python random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CUDA settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Additional deterministic settings
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Set default tensor type to avoid precision issues
    # torch.set_default_dtype(torch.float16)
seed_everything(42)

torch.use_deterministic_algorithms(True, warn_only=True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
# Initialize global components
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
# clip_model, clip_preprocess = clip.load("./weights/ViT-B-32.pt", device=device)

class PromptProjector(nn.Module):
    def __init__(self, input_dim, project_dim, output_dim, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, project_dim)
        self.hidden = nn.Sequential(
            nn.LayerNorm(project_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(project_dim, project_dim),
            nn.LayerNorm(project_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.output_proj = nn.Linear(project_dim, output_dim)
        
    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.hidden(x)  # Residual connection
        return self.output_proj(x)
    
def load_clip_to_cpu(cfg):
    """Load CLIP model to CPU for initialization."""
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model


def get_image_transforms():
    """Get standard image preprocessing transforms."""
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    
    resize_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        # RandomHorizontalFlip(),
        # RandomPerspective(),
        # RandomRotation(degrees=5),
        transforms.ToTensor(),
    ])
    
    normalize = transforms.Normalize(mean=mean, std=std)
    
    return resize_transform, normalize


def disable_safety_checker(images, clip_input):
    """Disable safety checker for diffusion models."""
    if len(images.shape) == 4:
        num_images = images.shape[0]
        return images, [False] * num_images
    else:
        return images, False


class TextEncoder(nn.Module):
    """CLIP Text Encoder ."""
    
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, feature=None):
        x = prompts.to(device) + self.positional_embedding.type(self.dtype).to(device)
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        if feature is not None:
            feature = einops.repeat(feature, 'm n -> k m n', k=5)
            x[:5, :, :] = x[:5, :, :] + feature
            
        x, _ = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class SemanticTokenExtractor(nn.Module):
    """Extract K semantic tokens from patch embeddings."""

    def __init__(
        self,
        embed_dim: int = 512,
        num_semantic_heads: int = 4,
        num_known_classes: int = 1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_semantic_heads = num_semantic_heads
        self.num_known_classes = max(1, num_known_classes)
        # Per-class query bank: [C, K, D].
        self.query_vectors = nn.Parameter(
            torch.randn(self.num_known_classes, num_semantic_heads, embed_dim)
        )
        nn.init.normal_(self.query_vectors, std=0.02)

    def _extract_all_class_tokens(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_embeddings: [B, N, D]
        Returns:
            semantic tokens for all classes: [B, C, K, D]
        """
        attention_scores = torch.einsum(
            "bnd,ckd->bckn", patch_embeddings, self.query_vectors
        )  # [B, C, K, N]
        attention_weights = F.softmax(attention_scores, dim=-1)
        semantic_tokens = torch.einsum(
            "bckn,bnd->bckd", attention_weights, patch_embeddings
        )  # [B, C, K, D]
        return semantic_tokens

    def forward(
        self,
        patch_embeddings: torch.Tensor,
        class_indices: Optional[torch.Tensor] = None,
        return_all_classes: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            patch_embeddings: [B, N, D]
            class_indices: optional [B] class indices for per-sample selection.
            return_all_classes: if True, returns [B, C, K, D].
        Returns:
            [B, K, D] (default), or [B, C, K, D] if return_all_classes=True.
        """
        if patch_embeddings.dim() != 3:
            raise ValueError("patch_embeddings must be [B, N, D]")

        patch_embeddings = patch_embeddings.to(
            device=self.query_vectors.device, dtype=self.query_vectors.dtype
        )
        semantic_tokens_all = self._extract_all_class_tokens(patch_embeddings)
        if return_all_classes:
            return semantic_tokens_all
        if class_indices is None:
            # Keep compatibility for code paths without explicit class anchors.
            return semantic_tokens_all.mean(dim=1)
        if class_indices.dim() != 1 or class_indices.size(0) != patch_embeddings.size(0):
            raise ValueError("class_indices must be shape [B]")
        class_indices = class_indices.to(device=semantic_tokens_all.device, dtype=torch.long)
        class_indices = class_indices.clamp(0, self.num_known_classes - 1)
        batch_idx = torch.arange(
            patch_embeddings.size(0), device=semantic_tokens_all.device
        )
        return semantic_tokens_all[batch_idx, class_indices]


class SemanticPromptBuilder(nn.Module):
    """Build PP-style semantic vectors and project them to SD text space."""

    def __init__(
        self,
        embed_dim: int = 512,
        num_semantic_heads: int = 4,
        sd_embed_dim: int = 768,
        num_known_classes: int = 1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_semantic_heads = num_semantic_heads
        self.semantic_extractor = SemanticTokenExtractor(
            embed_dim=embed_dim,
            num_semantic_heads=num_semantic_heads,
            num_known_classes=num_known_classes,
        )
        self.domain_projection = nn.Linear(embed_dim, embed_dim)
        self.semantic_projections = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim) for _ in range(num_semantic_heads)]
        )
        self.sd_projection = nn.Linear(embed_dim, sd_embed_dim)

    def extract_semantic_tokens(
        self,
        patch_embeddings: torch.Tensor,
        class_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.semantic_extractor(
            patch_embeddings, class_indices=class_indices, return_all_classes=False
        )

    def extract_semantic_tokens_all_classes(
        self,
        patch_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return self.semantic_extractor(
            patch_embeddings, class_indices=None, return_all_classes=True
        )

    def project_semantic_tokens(self, semantic_tokens: torch.Tensor) -> torch.Tensor:
        if semantic_tokens.dim() == 3:
            # [B, K, D] -> [B, K, D]
            projected = []
            for k in range(self.num_semantic_heads):
                projected.append(self.semantic_projections[k](semantic_tokens[:, k, :]))
            return torch.stack(projected, dim=1)
        if semantic_tokens.dim() == 4:
            # [B, C, K, D] -> [B, C, K, D]
            projected = []
            for k in range(self.num_semantic_heads):
                projected.append(self.semantic_projections[k](semantic_tokens[:, :, k, :]))
            return torch.stack(projected, dim=2)
        raise ValueError("semantic_tokens must be [B, K, D] or [B, C, K, D]")

    def project_domain_token(self, global_features: torch.Tensor) -> torch.Tensor:
        if global_features.dim() != 2:
            raise ValueError("global_features must be [B, D]")
        return self.domain_projection(global_features)

    def to_sd_prompt_delta(
        self,
        semantic_tokens: torch.Tensor,
        domain_token: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            semantic_tokens: [K, D] or [1, K, D]
            domain_token: [D] or [1, D]
        Returns:
            prompt delta in SD space: [1, K, 768]
        """
        if semantic_tokens.dim() == 2:
            semantic_tokens = semantic_tokens.unsqueeze(0)
        if domain_token.dim() == 1:
            domain_token = domain_token.unsqueeze(0)
        semantic_tokens = semantic_tokens + domain_token.unsqueeze(1)
        return self.sd_projection(semantic_tokens)


class StableDiffusion(nn.Module):
    """Stable Diffusion wrapper for image generation."""
    # model_id="runwayml/stable-diffusion-v1-5"
    def __init__(
        self,
        model_id="runwayml/stable-diffusion-v1-5",
        semantic_noise_std: float = 0.05,
        semantic_heads: int = 4,
        semantic_alpha: float = 0.3,
    ):
        super().__init__()
        # self.pipe = StableDiffusionPipeline.from_pretrained(
        #     model_id, 
        #     torch_dtype=torch.float16
        # ).to(device)
        torch.manual_seed(42)
    
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            torch_dtype=torch.float16,  # Use float32 for better determinism
            safety_checker=None,
            requires_safety_checker=False,
            # local_files_only = True
            
        ).to(device)
        
        # self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        #     self.pipe.scheduler.config,
        #     use_karras_sigmas=False  # Disable for determinism
        # )
        
        # Set generator seed before each use
        # self.generator = torch.Generator(device=device)
        self.pipe.set_progress_bar_config(disable=True)
        self.semantic_noise_std = semantic_noise_std
        self.semantic_heads = semantic_heads
        self.semantic_alpha = semantic_alpha


    def _expand_prompt_list(self, prompt: Union[str, List[str]], batchsize: int) -> List[str]:
        if isinstance(prompt, str):
            return [prompt] * batchsize
        if len(prompt) == 0:
            return [""] * batchsize
        if len(prompt) >= batchsize:
            return prompt[:batchsize]
        repeats = (batchsize + len(prompt) - 1) // len(prompt)
        return (prompt * repeats)[:batchsize]

    def _fuse_prompt_delta(
        self, prompt_embeds: torch.Tensor, prompt_delta: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if prompt_delta is None:
            return prompt_embeds
        if prompt_delta.dim() == 2:
            prompt_delta = prompt_delta.unsqueeze(0)
        token_count = min(prompt_delta.shape[1], prompt_embeds.shape[1] - 1)
        if token_count <= 0:
            return prompt_embeds
        fused = prompt_embeds.clone()
        fused[:, 1 : 1 + token_count, :] = (
            fused[:, 1 : 1 + token_count, :]
            + self.semantic_alpha
            * prompt_delta[:, :token_count, :].to(device=fused.device, dtype=fused.dtype)
        )
        return fused

    def _encode_prompt_with_semantic_noise(
        self, prompt: str, prompt_delta: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        text_inputs = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        with torch.no_grad():
            prompt_embeds = self.pipe.text_encoder(input_ids)[0]
        if self.semantic_noise_std > 0:
            noise = torch.randn_like(prompt_embeds) * self.semantic_noise_std
            prompt_embeds = prompt_embeds + noise
        return self._fuse_prompt_delta(prompt_embeds, prompt_delta)

    def forward(
        self,
        batch_size,
        pos_prompt: Union[str, List[str]],
        neg_prompt: Union[str, List[str]],
        pos_prompt_deltas: Optional[List[torch.Tensor]] = None,
    ):
        """Generate images from prompts."""
        if isinstance(pos_prompt, list):
            batchsize = len(pos_prompt)
        else:
            batchsize = 2 if batch_size == 5 else 1

        positive_prompts = self._expand_prompt_list(pos_prompt, batchsize)
        negative_prompts = self._expand_prompt_list(neg_prompt, batchsize)
        if pos_prompt_deltas is None:
            pos_prompt_deltas = [None] * batchsize
        elif len(pos_prompt_deltas) == 0:
            pos_prompt_deltas = [None] * batchsize
        elif len(pos_prompt_deltas) < batchsize:
            repeats = (batchsize + len(pos_prompt_deltas) - 1) // len(pos_prompt_deltas)
            pos_prompt_deltas = (pos_prompt_deltas * repeats)[:batchsize]
        else:
            pos_prompt_deltas = pos_prompt_deltas[:batchsize]
        
        generated_images = []
        with torch.no_grad():
            for i in range(batchsize):
                noisy_prompt_embeds = self._encode_prompt_with_semantic_noise(
                    positive_prompts[i], pos_prompt_deltas[i]
                )
                noisy_negative_prompt_embeds = self._encode_prompt_with_semantic_noise(negative_prompts[i])
                batch_output = self.pipe(
                    prompt_embeds=noisy_prompt_embeds,
                    negative_prompt_embeds=noisy_negative_prompt_embeds,
                    guidance_scale=15,
                    # generator=self.generator,
                )
                generated_images.append(batch_output.images[0])
        
        generated_images = torch.stack([ToTensor()(img) for img in generated_images]).to(torch.float16).to(device)
        return generated_images
    
class GenerateUnknownImages(nn.Module):
    """Generate and preprocess unknown images using Stable Diffusion."""
    
    def __init__(self, semantic_heads: int = 4, semantic_alpha: float = 0.3):
        super().__init__()
        self.diffusion = StableDiffusion(
            semantic_heads=semantic_heads,
            semantic_alpha=semantic_alpha,
        )
        self.resize_transform, self.normalize = get_image_transforms()

    def forward(
        self,
        batch_size,
        pos_prompt: Union[str, List[str]],
        neg_prompt: Union[str, List[str]],
        pos_prompt_deltas: Optional[List[torch.Tensor]] = None,
    ):
        """Generate normalized unknown images."""
        generated_images = self.diffusion(
            batch_size, pos_prompt, neg_prompt, pos_prompt_deltas=pos_prompt_deltas
        )
        resized_images = torch.stack([self.resize_transform(x) for x in generated_images])
        normalized_images = self.normalize(resized_images).to(device)
        return normalized_images

    def generate_pil_images(
        self,
        batch_size,
        pos_prompt: Union[str, List[str]],
        neg_prompt: Union[str, List[str]],
        pos_prompt_deltas: Optional[List[torch.Tensor]] = None,
    ):
        """Generate PIL images for saving or visualization."""
        with torch.no_grad():
            generated_images = self.diffusion(
                batch_size, pos_prompt, neg_prompt, pos_prompt_deltas=pos_prompt_deltas
            )
            pil_images = []
            for img_tensor in generated_images:
                img_tensor = img_tensor.detach().cpu().clamp(0, 1)
                pil_images.append(to_pil_image(img_tensor))
            return pil_images


class DynamicPseudoUnknownGenerator:
    """Dynamic pseudo-unknown generation policy for PACS-like training loops."""

    def __init__(
        self,
        unknown_image_generator: GenerateUnknownImages,
        prompt_pool: List[str],
        known_class_names: List[str],
        known_classes_text: str,
        dynamic_batch_size: int = 3,
        near_far_ratio: Tuple[int, int] = (2, 1),
        enable_dynamic: bool = True,
        attri_embed: Optional[torch.Tensor] = None,
        mask_embed: Optional[torch.Tensor] = None,
        class_to_attri_idx: Optional[Dict[str, int]] = None,
        num_semantic_heads: int = 4,
        offline_attri_alpha: float = 0.25,
    ):
        self.unknown_image_generator = unknown_image_generator
        self.prompt_pool = [p for p in prompt_pool if p]
        self.known_class_names = known_class_names
        self.known_classes_text = known_classes_text
        self.dynamic_batch_size = dynamic_batch_size
        self.near_far_ratio = near_far_ratio
        self.enable_dynamic = enable_dynamic
        self.attri_embed = attri_embed
        self.mask_embed = mask_embed
        self.class_to_attri_idx = class_to_attri_idx or {
            name: idx for idx, name in enumerate(known_class_names)
        }
        self.offline_attri_alpha = offline_attri_alpha

        self.far_semantic_hints = [
            "mechanical artifact",
            "mythic silhouette",
            "alien structure",
            "abstract geometry",
            "amphibian contour",
            "insectoid shape",
            "avian skeleton",
        ]
        self.fallback_terms = [
            "motorcycle", "satellite", "castle", "submarine", "volcano", "robot",
            "airplane", "reptile", "insect", "amphibian", "spaceship", "crystal",
        ]
        self.reset_epoch_stats()

    def reset_epoch_stats(self):
        self.unknown_distance_all: List[float] = []
        self.unknown_distance_near: List[float] = []
        self.unknown_distance_far: List[float] = []

    def _topk_known_class_names(self, labels: torch.Tensor, topk: int = 3) -> List[str]:
        if labels.numel() == 0:
            return []
        counts = torch.bincount(labels.detach().cpu(), minlength=len(self.known_class_names))
        top_indices = torch.topk(counts, k=min(topk, len(self.known_class_names))).indices.tolist()
        return [self.known_class_names[i] for i in top_indices if counts[i] > 0]

    def _non_llm_negative_terms(self) -> List[str]:
        known_tokens = set(self.known_classes_text.replace(",", " ").split())
        from_prompt_pool = [t.lower() for t in self.prompt_pool if t and t.lower() not in known_tokens]
        merged = from_prompt_pool + self.fallback_terms
        deduped = []
        for token in merged:
            if token not in deduped:
                deduped.append(token)
        return deduped if deduped else self.fallback_terms

    def compute_feature_space_guide(self, model, images: torch.Tensor, labels: torch.Tensor) -> Dict:
        with torch.no_grad():
            feats = model.get_image_features(images)
            feats = F.normalize(feats, dim=-1)

        unique_labels = torch.unique(labels)
        centers = []
        center_names = []
        for lbl in unique_labels:
            mask = labels == lbl
            class_feats = feats[mask]
            if class_feats.size(0) == 0:
                continue
            center = class_feats.mean(dim=0)
            center = F.normalize(center.unsqueeze(0), dim=-1).squeeze(0)
            centers.append(center)
            idx = int(lbl.item())
            if 0 <= idx < len(self.known_class_names):
                center_names.append(self.known_class_names[idx])
            else:
                center_names.append(f"class_{idx}")

        if len(centers) < 2:
            centers_tensor = torch.stack(centers) if centers else torch.empty(0, feats.shape[-1], device=feats.device)
            fallback_pair = random.sample(self.known_class_names, 2) if len(self.known_class_names) >= 2 else self.known_class_names * 2
            return {"centers_tensor": centers_tensor, "farthest_pair_names": fallback_pair}

        centers_tensor = torch.stack(centers)
        pair_dists = torch.cdist(centers_tensor, centers_tensor, p=2)
        upper = torch.triu_indices(pair_dists.shape[0], pair_dists.shape[1], offset=1)
        vals = pair_dists[upper[0], upper[1]]
        far_idx = torch.argmax(vals)
        i = int(upper[0][far_idx].item())
        j = int(upper[1][far_idx].item())
        farthest_pair_names = (center_names[i], center_names[j])
        return {"centers_tensor": centers_tensor, "farthest_pair_names": farthest_pair_names}

    def build_dynamic_pos_prompts(
        self, domain_name: str, guide: Dict, batch_size: int
    ) -> Tuple[List[str], List[str], List[Union[str, Tuple[str, str], None]]]:
        near_weight, far_weight = self.near_far_ratio
        near_count = max(1, int(round(batch_size * near_weight / (near_weight + far_weight))))
        far_count = max(0, batch_size - near_count)

        prompts: List[str] = []
        modes: List[str] = []
        anchors: List[Union[str, Tuple[str, str], None]] = []
        domain_phrase = domain_name.replace("_", " ")

        for _ in range(near_count):
            base_prompt = random.choice(self.prompt_pool)
            cls_a = random.choice(self.known_class_names)
            prompts.append(f"{domain_phrase} blending {cls_a.replace('_', ' ')} and {base_prompt}")
            modes.append("near")
            anchors.append(cls_a)

        far_a, far_b = guide.get("farthest_pair_names", random.sample(self.known_class_names, 2))
        for _ in range(far_count):
            far_hint = random.choice(self.far_semantic_hints)
            prompts.append(
                f"{domain_phrase} of a {far_hint}, semantically disjoint from "
                f"{far_a.replace('_', ' ')} and {far_b.replace('_', ' ')}"
            )
            modes.append("far")
            anchors.append((far_a, far_b))

        return prompts, modes, anchors

    def build_dynamic_neg_prompts(self, count: int) -> List[str]:
        """
        Unified negative prompt strategy (no near/far split):
        "other known classes + extra terms".
        """
        terms = self._non_llm_negative_terms()
        negatives = []
        for _ in range(count):
            sampled = ", ".join(random.sample(terms, min(2, len(terms)))) if len(terms) > 0 else ""
            negatives.append(f"{self.known_classes_text}, {sampled}".strip(", "))
        return negatives

    def _pool_offline_attribute(
        self,
        class_name: str,
        attri_embed: torch.Tensor,
        mask_embed: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        class_idx = self.class_to_attri_idx.get(class_name)
        if class_idx is None:
            return None
        if class_idx < 0 or class_idx >= attri_embed.shape[0]:
            return None
        attr_tokens = attri_embed[class_idx]  # [N, D]
        attr_mask = mask_embed[class_idx] if mask_embed.dim() > 1 else None
        if attr_mask is None:
            return attr_tokens.mean(dim=0)
        valid = (~attr_mask.bool()).unsqueeze(-1).to(attr_tokens.dtype)
        denom = valid.sum(dim=0).clamp(min=1.0)
        return (attr_tokens * valid).sum(dim=0) / denom

    def _build_prompt_deltas(
        self,
        model,
        known_images: torch.Tensor,
        modes: List[str],
        anchors: List[Union[str, Tuple[str, str], None]],
        attri_embed: Optional[torch.Tensor],
        mask_embed: Optional[torch.Tensor],
    ) -> List[torch.Tensor]:
        # Reuse Prompt-(2) semantic builder from the training model.
        prompt2_builder = model.class_semantic_builder
        with torch.no_grad():
            patch_features, global_features = model.get_patch_features(known_images)
            semantic_tokens_all = prompt2_builder.extract_semantic_tokens_all_classes(
                patch_features
            )  # [B, C, K, D]
            semantic_tokens_all = prompt2_builder.project_semantic_tokens(
                semantic_tokens_all
            )  # [B, C, K, D]
            domain_token = prompt2_builder.project_domain_token(
                global_features.mean(dim=0, keepdim=True)
            )

        # Average over batch to obtain class-specific semantic tokens: [C, K, D].
        base_tokens_per_class = semantic_tokens_all.mean(dim=0)
        deltas: List[torch.Tensor] = []
        use_offline = attri_embed is not None and mask_embed is not None
        known_class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(self.known_class_names)
        }
        for mode, anchor in zip(modes, anchors):
            if isinstance(anchor, str):
                cls_idx = known_class_to_idx.get(anchor, None)
                if cls_idx is not None and 0 <= cls_idx < base_tokens_per_class.size(0):
                    token_k = base_tokens_per_class[cls_idx].clone()
                else:
                    token_k = base_tokens_per_class.mean(dim=0).clone()
            elif isinstance(anchor, tuple):
                idx_a = known_class_to_idx.get(anchor[0], None)
                idx_b = known_class_to_idx.get(anchor[1], None)
                token_a = (
                    base_tokens_per_class[idx_a]
                    if idx_a is not None and 0 <= idx_a < base_tokens_per_class.size(0)
                    else None
                )
                token_b = (
                    base_tokens_per_class[idx_b]
                    if idx_b is not None and 0 <= idx_b < base_tokens_per_class.size(0)
                    else None
                )
                if token_a is not None and token_b is not None:
                    token_k = 0.5 * (token_a + token_b)
                elif token_a is not None:
                    token_k = token_a.clone()
                elif token_b is not None:
                    token_k = token_b.clone()
                else:
                    token_k = base_tokens_per_class.mean(dim=0).clone()
            else:
                token_k = base_tokens_per_class.mean(dim=0).clone()

            if mode == "far" and token_k.size(0) > 1:
                token_k = token_k - token_k.roll(shifts=1, dims=0)

            if use_offline:
                offline_vec: Optional[torch.Tensor] = None
                if isinstance(anchor, str):
                    offline_vec = self._pool_offline_attribute(anchor, attri_embed, mask_embed)
                elif isinstance(anchor, tuple):
                    off_a = self._pool_offline_attribute(anchor[0], attri_embed, mask_embed)
                    off_b = self._pool_offline_attribute(anchor[1], attri_embed, mask_embed)
                    if off_a is not None and off_b is not None:
                        offline_vec = 0.5 * (off_a + off_b)
                    elif off_a is not None:
                        offline_vec = off_a
                    elif off_b is not None:
                        offline_vec = off_b
                if offline_vec is not None:
                    token_k = (1 - self.offline_attri_alpha) * token_k + self.offline_attri_alpha * offline_vec.unsqueeze(0)

            deltas.append(
                prompt2_builder.to_sd_prompt_delta(token_k, domain_token).detach()
            )

        return deltas

    def generate(
        self,
        model,
        known_images: torch.Tensor,
        known_labels: torch.Tensor,
        domain_name: str,
        attri_embed: Optional[torch.Tensor] = None,
        mask_embed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[str], Dict]:
        batch_size = self.dynamic_batch_size if self.enable_dynamic else 1
        guide = self.compute_feature_space_guide(model, known_images, known_labels)
        used_attri = attri_embed if attri_embed is not None else self.attri_embed
        used_mask = mask_embed if mask_embed is not None else self.mask_embed

        if self.enable_dynamic:
            try:
                pos_prompts, modes, anchors = self.build_dynamic_pos_prompts(domain_name, guide, batch_size)
                neg_prompts = self.build_dynamic_neg_prompts(len(modes))
                prompt_deltas = self._build_prompt_deltas(
                    model=model,
                    known_images=known_images,
                    modes=modes,
                    anchors=anchors,
                    attri_embed=used_attri,
                    mask_embed=used_mask,
                )
                generated = self.unknown_image_generator(
                    batch_size,
                    pos_prompts,
                    neg_prompts,
                    pos_prompt_deltas=prompt_deltas,
                )
                return generated, modes, guide
            except Exception:
                pass

        fallback_prompt = domain_name.replace("_", " ") + " of a " + random.choice(self.prompt_pool)
        generated = self.unknown_image_generator(
            1, fallback_prompt, self.known_classes_text, pos_prompt_deltas=None
        )
        return generated, ["fallback"] * generated.shape[0], guide

    def update_distance_stats(self, model, selected_images: torch.Tensor, selected_modes: List[str], guide: Dict):
        centers_tensor = guide.get("centers_tensor")
        if centers_tensor is None or centers_tensor.numel() == 0 or selected_images.size(0) == 0:
            return
        with torch.no_grad():
            gen_feats = model.get_image_features(selected_images)
            gen_feats = F.normalize(gen_feats, dim=-1)
            min_dists = torch.cdist(gen_feats, centers_tensor, p=2).min(dim=1).values.detach().cpu().tolist()
        self.unknown_distance_all.extend(min_dists)
        for dist_val, mode in zip(min_dists, selected_modes):
            if mode == "near":
                self.unknown_distance_near.append(dist_val)
            elif mode == "far":
                self.unknown_distance_far.append(dist_val)

    def log_epoch_stats(self, epoch: int):
        if not self.unknown_distance_all:
            return
        all_mean = float(np.mean(self.unknown_distance_all))
        near_mean = float(np.mean(self.unknown_distance_near)) if self.unknown_distance_near else 0.0
        far_mean = float(np.mean(self.unknown_distance_far)) if self.unknown_distance_far else 0.0
        print(
            f"[DynamicUnknown][Epoch {epoch + 1}] min-dist mean={all_mean:.4f}, "
            f"near={near_mean:.4f}, far={far_mean:.4f}, "
            f"counts(all/near/far)=({len(self.unknown_distance_all)}/{len(self.unknown_distance_near)}/{len(self.unknown_distance_far)})"
        )

class CrossAttention(nn.Module):
    """Cross-attention mechanism for feature fusion."""
    # 0.25
    def __init__(self, embed_dim, num_heads, dropout=0.5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, 
            1, 
            batch_first=True, 
            dropout=dropout
        )
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, 
            num_heads, 
            batch_first=True, 
            dropout=dropout
        )

        self.temperature = nn.Parameter(torch.ones(1))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.layer_norm1 = nn.LayerNorm(embed_dim)

    def forward(self, image_features, attribute_embeddings, mask_embed):
        self_attn_output, _ = self.self_attn(
            attribute_embeddings, 
            attribute_embeddings, 
            attribute_embeddings,
            key_padding_mask=mask_embed
        )
        self_attn_output = self.layer_norm(self_attn_output + attribute_embeddings)
        attn_output, attn_weights = self.multihead_attn(
            image_features, 
            self_attn_output, 
            self_attn_output,
            key_padding_mask=mask_embed
        )
        output = self.layer_norm1(attn_output + image_features)
        
        return self.layer_norm2(self.out_proj(output)+output)


class MLP(nn.Module):
    """Multi-layer perceptron with optional activation and dropout."""
    
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super().__init__()
        self.fc1 = nn.Linear(in_size, mid_size)
        self.fc2 = nn.Linear(mid_size, out_size)
        self.dropout = nn.Dropout(dropout_r) if dropout_r > 0 else None
        self.relu = nn.ReLU(inplace=True) if use_relu else None

    def forward(self, x):
        x = self.fc1(x)
        if self.relu:
            x = self.relu(x)
        if self.dropout:
            x = self.dropout(x)
        return self.fc2(x)


class AttFlat(nn.Module):
    """Attention-based feature flattening from MCAN."""
    
    def __init__(self, embed_dim=512):
        super().__init__()
        self.mlp = MLP(embed_dim, embed_dim, 1, dropout_r=0.15, use_relu=True)
        self.linear_merge = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(x_mask.unsqueeze(2), -1e9)
        att = F.softmax(att, dim=1)
        
        att_list = [torch.sum(att[:, :, i:i+1] * x, dim=1) for i in range(1)]
        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)
        return x_atted


class CustomCLIP(nn.Module):
    """Custom CLIP model with semantic prompting and style adaptation."""
    
    def __init__(self, classnames: List[str], domainnames: List[str], clip_model: nn.Module, config, gated=False,project=False):
        super().__init__()
        # seed_everything(42)
        self.ctx = config["n_ctx"]
        self.dtype = clip_model.dtype
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        
        
        self.prompt_mlp = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 768)
        )
        
        self.per_class_gate = nn.Parameter(torch.ones(len(classnames) - 1) * 0.5)
        self.cross_attention = CrossAttention(512, config["n_head"])
        self.projector = nn.Linear(512, 512)
        # Prompt-2 builder: [dom] + [v_sem^1 ... v_sem^K] + [CLS]
        self.class_semantic_builder = SemanticPromptBuilder(
            embed_dim=512,
            num_semantic_heads=max(1, self.ctx - 1),
            sd_embed_dim=768,
            num_known_classes=max(1, len(classnames) - 1),
        )
        # Prompt-3 learnable semantic tokens [v_1 ... v_m], independent from prompt-2.
        unknown_ctx_len = max(0, self.ctx - 1)
        self.unknown_prompt_ctx = nn.Parameter(torch.empty(unknown_ctx_len, 512))
        if unknown_ctx_len > 0:
            nn.init.normal_(self.unknown_prompt_ctx, std=0.02)
        # self.projector = nn.Identity()
        
      
        self.logit_scale = clip_model.logit_scale
        self.num_class = len(classnames)
        self.classnames = classnames
        self.domainnames = domainnames
        self.num_domains = len(domainnames)
        self.gated = gated
        self.rep_margin = 0.2
        self.register_buffer("domain_feature_sum", torch.zeros(self.num_domains, 512))
        self.register_buffer("domain_feature_count", torch.zeros(self.num_domains))
        # Cache latest unknown-class prompt embeddings for external training losses.
        self.latest_unknown_prompt_embeddings: Optional[torch.Tensor] = None

    @torch.no_grad()
    def _update_domain_feature_bank(
        self, global_features: torch.Tensor, domain_ids: Optional[torch.Tensor]
    ) -> None:
        if domain_ids is None or global_features.numel() == 0:
            return
        if domain_ids.dim() != 1:
            domain_ids = domain_ids.view(-1)
        domain_ids = domain_ids.to(device=global_features.device, dtype=torch.long)
        domain_ids = domain_ids.clamp(min=0, max=max(0, self.num_domains - 1))
        for domain_id_t in torch.unique(domain_ids):
            domain_id = int(domain_id_t.item())
            mask = domain_ids == domain_id
            if not mask.any():
                continue
            self.domain_feature_sum[domain_id] += global_features[mask].detach().sum(dim=0)
            self.domain_feature_count[domain_id] += mask.sum().to(
                dtype=self.domain_feature_count.dtype
            )

    def _resolve_domain_mean_features(
        self, global_features: torch.Tensor, domain_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if domain_ids is None or global_features.numel() == 0:
            return global_features
        if domain_ids.dim() != 1:
            domain_ids = domain_ids.view(-1)
        domain_ids = domain_ids.to(device=global_features.device, dtype=torch.long)
        domain_ids = domain_ids.clamp(min=0, max=max(0, self.num_domains - 1))
        out = global_features.clone()
        for domain_id_t in torch.unique(domain_ids):
            domain_id = int(domain_id_t.item())
            count = self.domain_feature_count[domain_id]
            if count <= 0:
                continue
            domain_mean = self.domain_feature_sum[domain_id] / count
            matched_idx = torch.nonzero(domain_ids == domain_id, as_tuple=False).squeeze(1)
            if matched_idx.numel() == 0:
                continue
            fill = domain_mean.unsqueeze(0).expand(matched_idx.numel(), -1)
            out.index_copy_(0, matched_idx, fill)
        return out

    def get_image_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract image features without CoOp context."""
        image_features, _, _, _ = self.image_encoder(image.type(self.dtype))
        image_features = F.normalize(image_features, dim=-1)
        image_features = self.prompt_mlp(image_features)
        return image_features

    def get_patch_features(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get patch tokens and global token from the latest visual layer."""
        _, layer_feat_list, _, _ = self.image_encoder(image.type(self.dtype))
        if layer_feat_list:
            patch_tokens = layer_feat_list[-1][:, 1:, :]
        else:
            image_features, _, _, _ = self.image_encoder(image.type(self.dtype))
            patch_tokens = image_features.unsqueeze(1)
        patch_tokens = F.normalize(patch_tokens, dim=-1)
        global_features = patch_tokens.mean(dim=1)
        return patch_tokens, global_features
    def forward(self, image: torch.Tensor, attri: torch.Tensor, mask_embed: torch.Tensor, 
                label: torch.Tensor = None, dom_label: torch.Tensor = None, batch=None):
        """Forward pass with optional domain adaptation."""
        image_features, layer_feat_list, _, _ = self.image_encoder(image.type(self.dtype))
        image_features = F.normalize(image_features, dim=-1)

        logit_scale = self.logit_scale.exp()
        # Main classifier: merge known semantic prompts + unknown prompt, then similarity.
        semantic_logits_all = self._build_semantic_inference_logits(
            image_features=image_features,
            layer_feat_list=layer_feat_list,
            dom_label=dom_label,
            logit_scale=logit_scale,
        )

        if dom_label is not None and batch is not None:
            known_mask = (label < self.num_class - 1).float()
            known_count = image_features.size(0) - batch if batch is not None else image_features.size(0)
            unknown_count = image_features.size(0) - known_count

            probs_sem = torch.softmax(semantic_logits_all, dim=-1)
            H_b = -torch.sum(probs_sem * torch.log(probs_sem + 1e-10), dim=-1)
            if known_mask.sum() > 0:
                layer_loss = (H_b * known_mask).sum() / known_mask.sum()
            else:
                layer_loss = torch.zeros((), device=image.device, dtype=H_b.dtype)
            if layer_feat_list:
                visual = self.image_encoder
                num_patches = visual.positional_embedding.shape[0] - 1
                known_patch_tokens = layer_feat_list[-1][:known_count, 1 : 1 + num_patches, :]
                low_level_known_features = layer_feat_list[0][:known_count, 1 : 1 + num_patches, :].mean(dim=1)
                low_level_known_features = F.normalize(low_level_known_features, dim=-1)
            else:
                known_patch_tokens = image_features[:known_count, :].unsqueeze(1)
                low_level_known_features = image_features[:known_count, :]
            known_patch_tokens = F.normalize(known_patch_tokens, dim=-1)
            high_level_known_features = image_features[:known_count, :]
            high_level_unknown_features = (
                image_features[known_count:, :] if unknown_count > 0 else torch.empty(0, image_features.size(-1), device=image_features.device)
            )
            known_labels = label[:known_count]

            loss_sty, domain_prompt_embeddings, semantic_prompt_embeddings, unknown_prompt_embedding = self._compute_style_loss(
                low_level_known_features,
                dom_label[:-batch],
                known_labels,
                attri,
                mask_embed,
                logit_scale,
                known_patch_tokens,
            )
            known_cls = self.num_class - 1
            semantic_prompt_per_sample = semantic_prompt_embeddings.view(known_count, known_cls, -1)
            domain_prompt_per_sample = domain_prompt_embeddings.view(known_count, known_cls, -1)
            unknown_prompt_per_sample = unknown_prompt_embedding

            # L_align between prompt-(1) and prompt-(2)
            align_loss = (1 - F.cosine_similarity(domain_prompt_embeddings, semantic_prompt_embeddings, dim=1)).mean()

            # Semantic classification with prompt-(2) and prompt-(3) on high-level image embeddings
            known_prompt_proto = semantic_prompt_per_sample.mean(dim=0)  # [C, D]
            unknown_prompt_proto = unknown_prompt_per_sample.mean(dim=0, keepdim=True)  # [1, D]
            semantic_proto = torch.cat([known_prompt_proto, unknown_prompt_proto], dim=0)  # [C+1, D]
            semantic_logits_known = logit_scale * (high_level_known_features @ semantic_proto.t())
            semantic_cls_loss = F.cross_entropy(semantic_logits_known, known_labels)
            if unknown_count > 0:
                semantic_logits_unknown = logit_scale * (high_level_unknown_features @ semantic_proto.t())
                unknown_targets = torch.full(
                    (unknown_count,), known_cls, device=image_features.device, dtype=torch.long
                )
                semantic_cls_loss = semantic_cls_loss + F.cross_entropy(
                    semantic_logits_unknown, unknown_targets
                )

            # L_rep: keep unknown prompt separated from known visual class prototypes
            class_visual_prototypes = []
            for c in range(known_cls):
                cls_mask = known_labels == c
                if cls_mask.any():
                    class_visual_prototypes.append(high_level_known_features[cls_mask].mean(dim=0))
            if class_visual_prototypes:
                class_visual_prototypes = F.normalize(torch.stack(class_visual_prototypes, dim=0), dim=-1)
                rep_sims = unknown_prompt_per_sample @ class_visual_prototypes.t()
                rep_loss = F.relu(self.rep_margin - rep_sims).mean()
            else:
                rep_loss = torch.zeros((), device=image_features.device, dtype=image_features.dtype)

            # L_coh: unknown prompt vs mean known semantic prompts
            semantic_mean_per_sample = semantic_prompt_per_sample.mean(dim=1)
            coh_loss = ((unknown_prompt_per_sample - semantic_mean_per_sample) ** 2).sum(dim=1).mean()

            self.latest_unknown_prompt_embeddings = unknown_prompt_embedding
            return (
                semantic_logits_all,
                loss_sty,
                domain_prompt_embeddings,
                semantic_prompt_embeddings,
                layer_loss,
                align_loss,
                semantic_cls_loss,
                rep_loss,
                coh_loss,
            )
        else:
            self.latest_unknown_prompt_embeddings = None
            return semantic_logits_all, image_features
    def _build_class_semantic_prompt_embeddings(
        self,
        patch_tokens: torch.Tensor,
        global_features: torch.Tensor,
        tokenized_prompts: torch.Tensor,
        domain_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build prompt-2 embeddings with structure:
        [dom] + [v_sem^1 ... v_sem^K] + [CLS(class)].

        The visual semantic embeddings are extracted from patch tokens with the
        same semantic builder family used in generated-image positive prompts.
        """
        with torch.no_grad():
            base_embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        ctx_tokens = self._build_dom_semantic_ctx_tokens(
            patch_tokens, global_features, domain_ids=domain_ids
        )
        return self._encode_prompts_with_ctx(base_embedding, tokenized_prompts, ctx_tokens)

    def _build_dom_semantic_ctx_tokens(
        self,
        patch_tokens: torch.Tensor,
        global_features: torch.Tensor,
        domain_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build ctx tokens for prompt-2:
        [dom] + [v_sem^1 ... v_sem^m].
        """
        semantic_tokens = self.class_semantic_builder.extract_semantic_tokens_all_classes(
            patch_tokens
        )
        semantic_tokens = self.class_semantic_builder.project_semantic_tokens(semantic_tokens)
        self._update_domain_feature_bank(global_features, domain_ids)
        domain_mean_features = self._resolve_domain_mean_features(global_features, domain_ids)
        domain_token = self.class_semantic_builder.project_domain_token(domain_mean_features)

        batch_size = patch_tokens.size(0)
        num_known_cls = self.num_class - 1
        ctx_tokens = torch.zeros(
            batch_size,
            num_known_cls,
            self.ctx,
            semantic_tokens.size(-1),
            device=patch_tokens.device,
            dtype=semantic_tokens.dtype,
        )
        ctx_tokens[:, :, 0, :] = domain_token.unsqueeze(1)
        if self.ctx > 1 and semantic_tokens.size(1) > 0:
            use_k = min(self.ctx - 1, semantic_tokens.size(2))
            ctx_tokens[:, :, 1 : 1 + use_k, :] = semantic_tokens[:, :, :use_k, :]
        return ctx_tokens

    def _build_unknown_learnable_ctx_tokens(
        self, global_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Build prompt-3 ctx tokens:
        [dom] + [v_1 ... v_m], where [v_1 ... v_m] are independent learnable tokens.
        """
        domain_token = self.class_semantic_builder.project_domain_token(global_features)
        batch_size = global_features.size(0)
        ctx_tokens = torch.zeros(
            batch_size,
            self.ctx,
            domain_token.size(-1),
            device=global_features.device,
            dtype=domain_token.dtype,
        )
        ctx_tokens[:, 0, :] = domain_token
        if self.ctx > 1 and self.unknown_prompt_ctx.numel() > 0:
            learnable_ctx = self.unknown_prompt_ctx.to(
                device=global_features.device, dtype=domain_token.dtype
            )
            use_k = min(self.ctx - 1, learnable_ctx.size(0))
            ctx_tokens[:, 1 : 1 + use_k, :] = learnable_ctx[:use_k, :].unsqueeze(0)
        return ctx_tokens

    def _encode_prompts_with_ctx(
        self,
        base_embedding: torch.Tensor,
        tokenized_prompts: torch.Tensor,
        ctx_tokens: torch.Tensor,
    ) -> torch.Tensor:
        prompt_embeddings: List[torch.Tensor] = []
        token_start = 1
        token_end = token_start + self.ctx
        batch_size = ctx_tokens.size(0)
        for i in range(batch_size):
            embedding_copy = base_embedding.clone()
            if ctx_tokens.dim() == 3:
                ctx_i = ctx_tokens[i].unsqueeze(0)
            elif ctx_tokens.dim() == 4:
                ctx_i = ctx_tokens[i]
            else:
                raise ValueError("ctx_tokens must be [B, ctx, D] or [B, C, ctx, D]")
            embedding_copy[:, token_start:token_end, :] += ctx_i.to(
                device=embedding_copy.device, dtype=embedding_copy.dtype
            )
            embedding_int = self.text_encoder(embedding_copy, tokenized_prompts)
            prompt_embeddings.append(F.normalize(embedding_int, dim=-1))
        return torch.stack(prompt_embeddings, dim=0)

    def _build_unknown_semantic_prompt_embeddings(
        self,
        patch_tokens: torch.Tensor,
        global_features: torch.Tensor,
        tokenized_unknown_prompt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build prompt-3 unknown-class prompt embeddings:
        [dom] + [v_1 ... v_m] + [unknown].
        """
        with torch.no_grad():
            base_embedding = clip_model.token_embedding(tokenized_unknown_prompt).type(self.dtype)
        ctx_tokens = self._build_unknown_learnable_ctx_tokens(global_features)
        unknown_prompt_embeddings = self._encode_prompts_with_ctx(
            base_embedding, tokenized_unknown_prompt, ctx_tokens
        )
        # [B, 1, D] -> [B, D]
        return unknown_prompt_embeddings[:, 0, :]

    def _build_semantic_inference_logits(
        self,
        image_features: torch.Tensor,
        layer_feat_list: List[torch.Tensor],
        dom_label: Optional[torch.Tensor],
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inference head based on prompt-(2) and prompt-(3):
        compute similarity between high-level image embedding and
        [known semantic prompts + unknown prompt].
        """
        device = image_features.device
        batch_size = image_features.size(0)
        known_cls = self.num_class - 1

        if layer_feat_list:
            visual = self.image_encoder
            num_patches = visual.positional_embedding.shape[0] - 1
            patch_tokens = layer_feat_list[-1][:, 1 : 1 + num_patches, :]
        else:
            patch_tokens = image_features.unsqueeze(1)
        patch_tokens = F.normalize(patch_tokens, dim=-1)

        semantic_logits_list = []
        for i in range(batch_size):
            tokenized_prompts = torch.cat(
                [clip.tokenize(p.replace("_", " ")) for p in self.classnames[:-1]]
            ).to(device)
            tokenized_unknown_prompt = clip.tokenize(
                self.classnames[-1].replace("_", " ")
            ).to(device)

            sample_patch_tokens = patch_tokens[i : i + 1]
            sample_global_features = image_features[i : i + 1]
            sample_domain_ids = dom_label[i : i + 1] if dom_label is not None else None
            semantic_prompt_embeddings = self._build_class_semantic_prompt_embeddings(
                patch_tokens=sample_patch_tokens,
                global_features=sample_global_features,
                tokenized_prompts=tokenized_prompts,
                domain_ids=sample_domain_ids,
            )[0]  # [C, D]
            unknown_prompt_embedding = self._build_unknown_semantic_prompt_embeddings(
                patch_tokens=sample_patch_tokens,
                global_features=sample_global_features,
                tokenized_unknown_prompt=tokenized_unknown_prompt,
            )[0].unsqueeze(0)  # [1, D]

            semantic_proto = torch.cat(
                [semantic_prompt_embeddings[:known_cls], unknown_prompt_embedding], dim=0
            )  # [C+1, D]
            semantic_logits = logit_scale * (image_features[i] @ semantic_proto.t())
            semantic_logits_list.append(semantic_logits)

        return torch.stack(semantic_logits_list, dim=0)

    def _process_domain_features(
        self,
        domain_features,
        cross_atten,
        domain_img,
        tokenized_prompts,
        logit_scale,
    ):
        """
        Build domain-modeled semantic-generic known-class prompts.

        cross_atten gives class-specific attribute-enhanced vectors A'_y(x).
        We follow the paper-style design by averaging over classes to obtain
        class-agnostic A''(x), then add A''(x) token-wise to each known-class
        domain-specific base prompt.
        """
        # Prompt embeddings generated by domain-modeled semantic-generic prompts.
        domain_prompt_embeddings = []
        domain_logits = []
        
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)
        
        # cross_atten: [num_known_cls, num_samples, dim]
        class_agnostic_ctx = cross_atten.mean(dim=0)  # [num_samples, dim]
        token_start = 1
        token_end = token_start + self.ctx

        for i in range(domain_features.size(0)):
            embedding_copy = embedding.clone()

            # A''(x_i): same context added to all known-class prompts.
            ctx_i = class_agnostic_ctx[i].view(1, 1, -1)
            embedding_copy[:, token_start:token_end, :] += ctx_i
            
            # Compute embeddings and logits
            embedding_int = self.text_encoder(embedding_copy, tokenized_prompts)
            embedding_int = F.normalize(embedding_int, dim=-1)
            logit = logit_scale * domain_features[i] @ embedding_int.t()
            
            domain_logits.append(logit)
            domain_prompt_embeddings.append(embedding_int)
        
        domain_logits = torch.stack(domain_logits)
        domain_prompt_embeddings = torch.stack(domain_prompt_embeddings)

        return domain_logits, domain_prompt_embeddings
    @torch.cuda.amp.autocast()
    def _compute_style_loss(self, image_features: torch.Tensor, dom_label: torch.Tensor,
                        label: torch.Tensor, attri: torch.Tensor, mask_embed: torch.Tensor,
                        logit_scale: torch.Tensor, known_patch_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute style adaptation loss across domains."""
        device = image_features.device
        semantic_prompt_embedding_list = []
        unknown_prompt_embedding_list = []
        domain_prompt_embedding_list = []
        
        # Collect all logits and labels from all domains
        all_logits = []
        all_labels = []
    
        for domain in [0, 1, 2]:
            domain_mask = dom_label == domain
            domain_features = image_features[domain_mask]
        
            if domain_features.size(0) == 0:
                continue
            
            original_indices = torch.nonzero(domain_mask, as_tuple=False).squeeze(1)
            domain_labels = label[domain_mask]
            domain_patch_tokens = known_patch_tokens[domain_mask]
        
            domain_name = self.domainnames[domain].replace('_', ' ')
            tokenized_domain_prompts = torch.cat([
                clip.tokenize(f"A {domain_name} of a {p}")
                for p in self.classnames[:-1]
            ]).to(device)
            tokenized_prompts = torch.cat([
                clip.tokenize(p.replace("_", " "))
                for p in self.classnames[:-1]
            ]).to(device)
            tokenized_unknown_prompt = clip.tokenize(
                self.classnames[-1].replace("_", " ")
            ).to(device)
        
            n_cls = len(self.classnames)
            domain_img = einops.repeat(domain_features, 'm n -> k m n', k=n_cls-1)
            cross_atten = self.cross_attention(domain_img, self.projector(attri), mask_embed)
            domain_logits, domain_prompt_embeddings = self._process_domain_features(
                domain_features, cross_atten, domain_img, tokenized_domain_prompts,
                logit_scale
            )
            semantic_prompt_embeddings = self._build_class_semantic_prompt_embeddings(
                patch_tokens=domain_patch_tokens,
                global_features=domain_features,
                tokenized_prompts=tokenized_prompts,
                domain_ids=torch.full(
                    (domain_features.size(0),),
                    domain,
                    device=device,
                    dtype=torch.long,
                ),
            )
            unknown_prompt_embeddings = self._build_unknown_semantic_prompt_embeddings(
                patch_tokens=domain_patch_tokens,
                global_features=domain_features,
                tokenized_unknown_prompt=tokenized_unknown_prompt,
            )
            for i, idx in enumerate(original_indices):
                domain_prompt_embedding_list.append((idx.item(), domain_prompt_embeddings[i]))
                semantic_prompt_embedding_list.append((idx.item(), semantic_prompt_embeddings[i]))
                unknown_prompt_embedding_list.append((idx.item(), unknown_prompt_embeddings[i]))
            
            # Collect logits and labels instead of computing loss immediately
            all_logits.append(domain_logits)
            all_labels.append(domain_labels)
    
        # Compute single cross-entropy loss if we have any data
        if all_logits:
            # Concatenate all logits and labels
            combined_logits = torch.cat(all_logits, dim=0)
            combined_labels = torch.cat(all_labels, dim=0)
            
            # Compute single cross-entropy loss
            total_loss = F.cross_entropy(combined_logits, combined_labels)
        else:
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
        domain_prompt_embedding_list.sort(key=lambda x: x[0])
        domain_prompt_embeddings = (
            torch.cat([x[1] for x in domain_prompt_embedding_list])
            if domain_prompt_embedding_list
            else torch.empty(0, device=device)
        )
        semantic_prompt_embedding_list.sort(key=lambda x: x[0])
        prompt_embeddings = (
            torch.cat([x[1] for x in semantic_prompt_embedding_list])
            if semantic_prompt_embedding_list
            else torch.empty(0, device=device)
        )
        unknown_prompt_embedding_list.sort(key=lambda x: x[0])
        unknown_prompt_embeddings = (
            torch.stack([x[1] for x in unknown_prompt_embedding_list], dim=0)
            if unknown_prompt_embedding_list
            else torch.empty(0, device=device)
        )

        return total_loss, domain_prompt_embeddings, prompt_embeddings, unknown_prompt_embeddings