"""Dataset helpers for loading inpainted images and their masks.

Provides:
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode as IM

ImageTensor = torch.Tensor
MaskTensor = torch.Tensor


def _list_image_mask_pairs(images_dir: str, masks_dir: str) -> List[Tuple[str, str]]:
	"""Return list of (image_path, mask_path) pairs with matching stems.

	Only pairs where both an image and a mask exist are returned. Supported image
	extensions are .png, .jpg, .jpeg.
	"""
	img_dir = Path(images_dir)
	mask_dir = Path(masks_dir)
	exts = {".png", ".jpg", ".jpeg"}

	# Map mask stems to their path (choose first matching ext)
	mask_map = {}
	for p in mask_dir.iterdir() if mask_dir.exists() else []:
		if p.suffix.lower() in exts and p.is_file():
			mask_map[p.stem] = str(p)

	pairs: List[Tuple[str, str]] = []
	for p in img_dir.iterdir() if img_dir.exists() else []:
		if p.suffix.lower() in exts and p.is_file():
			stem = p.stem
			mask_path = mask_map.get(stem)
			if mask_path:
				pairs.append((str(p), mask_path))

	pairs.sort()  # deterministic order
	return pairs


def _list_images(images_dir: str) -> List[str]:
	"""Return sorted list of image file paths in a folder.

	Supported extensions: .png, .jpg, .jpeg
	"""
	img_dir = Path(images_dir)
	exts = {".png", ".jpg", ".jpeg"}
	imgs: List[str] = []
	for p in img_dir.iterdir() if img_dir.exists() else []:
		if p.suffix.lower() in exts and p.is_file():
			imgs.append(str(p))
	imgs.sort()
	return imgs


class ImageMaskDataset(Dataset):
	"""Standard indexable Dataset that returns (img_tensor, mask_tensor).

	Args:
		images_dir: folder containing inpainted images.
		masks_dir: folder containing corresponding masks.
		transform_img: optional transform applied to PIL image -> tensor.
		transform_mask: optional transform applied to PIL mask -> tensor.
	"""

	def __init__(
		self,
		images_dir: str = "./data/COCO_inpainted",
		masks_dir: str = "./data/masks",
		image_size: int = 256,
		transform_img: Optional[Callable[[Image.Image], ImageTensor]] = None,
		transform_mask: Optional[Callable[[Image.Image], MaskTensor]] = None,
	): 
		self.pairs = _list_image_mask_pairs(images_dir, masks_dir)
		self.image_size = image_size
		if not self.pairs:
			raise RuntimeError(
				f"No image/mask pairs found in {images_dir} and {masks_dir}"
			)

		# Default transforms: resize/center-crop to image_size then to tensor
		if transform_img is not None:
			self.transform_img = transform_img
		else:
			self.transform_img = T.Compose([
				T.Resize((image_size, image_size), interpolation=IM.BILINEAR),
				T.CenterCrop((image_size, image_size)),
				T.ToTensor(),
			])

		# For masks: single-channel, resize with nearest neighbor to keep labels
		if transform_mask is not None:
			self.transform_mask = transform_mask
		else:
			self.transform_mask = T.Compose([
				T.Resize((image_size, image_size), interpolation=IM.NEAREST),
				T.CenterCrop((image_size, image_size)),
				T.Grayscale(num_output_channels=1),
				T.ToTensor(),
			])

	def __len__(self) -> int:
		return len(self.pairs)

	def __getitem__(self, idx: int) -> Tuple[ImageTensor, MaskTensor, ImageTensor]:
		img_path, mask_path = self.pairs[idx]
		img = Image.open(img_path).convert("RGB")
		mask = Image.open(mask_path).convert("L")

		img_t = self.transform_img(img)
		mask_t = self.transform_mask(mask)

		# Also return an un-normalized resized original image. We resize to the
		# same (image_size, image_size) but DO NOT apply ToTensor()/normalization
		# so pixel values remain in 0..255. Represent as a uint8 torch tensor
		# with shape (C, H, W) to be compatible with other tensors.

		# Simpler: derive target size from first Resize call parameters if possible
		# Fallback: use center crop size if present
		try:
			# Attempt to get explicit size from transform sequence
			for t in self.transform_img.transforms:
				if isinstance(t, T.Resize):
					target_size = t.size
					break
			else:
				target_size = None
		except Exception:
			target_size = None

		if target_size is None:
			# use provided image_size by reading the CenterCrop or fallback to 256
			try:
				for t in self.transform_img.transforms:
					if isinstance(t, T.CenterCrop):
						target_size = t.size
						break
			except Exception:
				pass
		if target_size is None:
			# final fallback: square of length of first Resize argument if given,
			# otherwise use 256
			try:
				# some Resize instances use a tuple
				for t in self.transform_img.transforms:
					if isinstance(t, T.Resize):
						if isinstance(t.size, int):
							target_size = (t.size, t.size)
						else:
							target_size = t.size
						break
			except Exception:
				pass
		if target_size is None:
			target_size = (self.image_size, self.image_size)
		# normalize int -> tuple
		if isinstance(target_size, int):
			target_size = (target_size, target_size)

		orig_resized = img.resize((target_size[0], target_size[1]), resample=Image.BILINEAR)
		orig_arr = np.array(orig_resized)
		# Ensure shape (H, W, C)
		if orig_arr.ndim == 2:
			# grayscale -> replicate channels
			orig_arr = np.stack([orig_arr] * 3, axis=-1)
		# Convert to torch tensor, channel-first, uint8
		orig_t = torch.from_numpy(orig_arr).permute(2, 0, 1).contiguous()

		return img_t, mask_t, orig_t


class BatchedImageMaskIterable(IterableDataset):
	"""IterableDataset that yields batches of (imgs, masks).

	This dataset reads image/mask files and yields pre-batched tensors so
	iteration returns (batch_imgs, batch_masks). Use with DataLoader(...,
	batch_size=None) or just iterate over the dataset directly.

	Args:
		images_dir, masks_dir: folders with files named [id].png
		batch_size: number of samples per yielded batch
		shuffle: whether to shuffle samples each epoch
		drop_last: whether to drop the final incomplete batch
		transform_img, transform_mask: transforms applied per sample
	"""

	def __init__(
		self,
		images_dir: str = "./data/COCO_inpainted",
		masks_dir: str = "./data/masks",
		image_size: int = 256,
		batch_size: int = 8,
		shuffle: bool = False,
		drop_last: bool = False,
		transform_img: Optional[Callable[[Image.Image], ImageTensor]] = None,
		transform_mask: Optional[Callable[[Image.Image], MaskTensor]] = None,
	):
		super().__init__()
		self.pairs = _list_image_mask_pairs(images_dir, masks_dir)
		if not self.pairs:
			raise RuntimeError(
				f"No image/mask pairs found in {images_dir} and {masks_dir}"
			)
		self.image_size = image_size
		self.batch_size = int(batch_size)
		self.shuffle = bool(shuffle)
		self.drop_last = bool(drop_last)

		# reuse ImageMaskDataset defaults if transforms not provided; include resize
		if transform_img is not None:
			self.transform_img = transform_img
		else:
			self.transform_img = T.Compose([
				T.Resize((image_size, image_size), interpolation=IM.BILINEAR),
				T.CenterCrop((image_size, image_size)),
				T.ToTensor(),
			])

		if transform_mask is not None:
			self.transform_mask = transform_mask
		else:
			self.transform_mask = T.Compose([
				T.Resize((image_size, image_size), interpolation=IM.NEAREST),
				T.CenterCrop((image_size, image_size)),
				T.Grayscale(num_output_channels=1),
				T.ToTensor(),
			])

	def __iter__(self):
		# Support DataLoader worker splitting
		worker_info = get_worker_info()

		# Build indices for this epoch
		indices = list(range(len(self.pairs)))
		if self.shuffle:
			random.shuffle(indices)

		if worker_info is not None:
			# Split indices between workers evenly
			per_worker = int(math.ceil(len(indices) / float(worker_info.num_workers)))
			start = worker_info.id * per_worker
			end = min(start + per_worker, len(indices))
			indices = indices[start:end]

		batch_imgs: List[ImageTensor] = []
		batch_masks: List[MaskTensor] = []
		batch_origs: List[ImageTensor] = []

		for i in indices:
			img_path, mask_path = self.pairs[i]
			img = Image.open(img_path).convert("RGB")
			mask = Image.open(mask_path).convert("L")

			img_t = self.transform_img(img)
			mask_t = self.transform_mask(mask)

			# produce un-normalized original resized image (uint8 torch tensor)
			# derive target size same as transform_img
			try:
				for t in self.transform_img.transforms:
					if isinstance(t, T.Resize):
						target_size = t.size
						break
			except Exception:
				target_size = None
			if target_size is None:
				# try CenterCrop
				for t in self.transform_img.transforms:
					if isinstance(t, T.CenterCrop):
						target_size = t.size
						break
			if target_size is None:
				target_size = (self.image_size, self.image_size)
			# normalize int -> tuple
			if isinstance(target_size, int):
				target_size = (target_size, target_size)

			orig_resized = img.resize((target_size[0], target_size[1]), resample=Image.BILINEAR)
			orig_arr = np.array(orig_resized)
			if orig_arr.ndim == 2:
				orig_arr = np.stack([orig_arr] * 3, axis=-1)
			orig_t = torch.from_numpy(orig_arr).permute(2, 0, 1).contiguous()

			batch_imgs.append(img_t)
			batch_masks.append(mask_t)
			batch_origs.append(orig_t)

			if len(batch_imgs) == self.batch_size:
				yield torch.stack(batch_imgs, dim=0), torch.stack(batch_masks, dim=0), torch.stack(batch_origs, dim=0)
				batch_imgs = []
				batch_masks = []
				batch_origs = []

		# yield last partial batch
		if batch_imgs and not self.drop_last:
			yield torch.stack(batch_imgs, dim=0), torch.stack(batch_masks, dim=0), torch.stack(batch_origs, dim=0)


def get_dataloader(
	images_dir: str = "./data/COCO_inpainted",
	masks_dir: str = "./data/masks",
	batch_size: int = 8,
	shuffle: bool = False,
	num_workers: int = 0,
	drop_last: bool = False,
	transform_img: Optional[Callable[[Image.Image], ImageTensor]] = None,
	transform_mask: Optional[Callable[[Image.Image], MaskTensor]] = None,
	model: Optional[torch.nn.Module] = None,
	image_size: Optional[int] = None,
) -> DataLoader:
	"""Return a DataLoader that produces (batch_imgs, batch_masks) tuples.

	Note: the returned DataLoader must be used with batch_size=None because the
	dataset yields pre-batched items.
	"""
	# If user passed a model, try to infer the required image_size from it.
	chosen_size = image_size
	if chosen_size is None and model is not None:
		chosen_size = _infer_image_size_from_model(model)
	if chosen_size is None:
		chosen_size = 256

	ds = BatchedImageMaskIterable(
		images_dir=images_dir,
		masks_dir=masks_dir,
		image_size=chosen_size,
		batch_size=batch_size,
		shuffle=shuffle,
		drop_last=drop_last,
		transform_img=transform_img,
		transform_mask=transform_mask,
	)

	return DataLoader(ds, batch_size=None, num_workers=num_workers)


class BatchedImageIterable(IterableDataset):
	"""IterableDataset that yields batches of images (no masks).

	Yields either (batch_imgs, batch_origs) where each is a torch tensor with
	shape (B, C, H, W). Use DataLoader(..., batch_size=None) or iterate over the
	dataset directly.

	Args mirror BatchedImageMaskIterable except there's no masks_dir and no
	transform_mask.
	"""

	def __init__(
		self,
		images_dir: str = "./data/COCO_inpainted",
		image_size: int = 256,
		batch_size: int = 8,
		shuffle: bool = False,
		drop_last: bool = False,
		transform_img: Optional[Callable[[Image.Image], ImageTensor]] = None,
	):
		super().__init__()
		self.images = _list_images(images_dir)
		if not self.images:
			raise RuntimeError(f"No images found in {images_dir}")
		self.image_size = image_size
		self.batch_size = int(batch_size)
		self.shuffle = bool(shuffle)
		self.drop_last = bool(drop_last)

		if transform_img is not None:
			self.transform_img = transform_img
		else:
			self.transform_img = T.Compose([
				T.Resize((image_size, image_size), interpolation=IM.BILINEAR),
				T.CenterCrop((image_size, image_size)),
				T.ToTensor(),
			])

	def __iter__(self):
		worker_info = get_worker_info()
		indices = list(range(len(self.images)))
		if self.shuffle:
			random.shuffle(indices)

		if worker_info is not None:
			per_worker = int(math.ceil(len(indices) / float(worker_info.num_workers)))
			start = worker_info.id * per_worker
			end = min(start + per_worker, len(indices))
			indices = indices[start:end]

		batch_imgs: List[ImageTensor] = []
		batch_origs: List[ImageTensor] = []

		for i in indices:
			img_path = self.images[i]
			img = Image.open(img_path).convert("RGB")

			img_t = self.transform_img(img)

			# derive target_size same as transform_img (reuse logic from above)
			try:
				for t in self.transform_img.transforms:
					if isinstance(t, T.Resize):
						target_size = t.size
						break
			except Exception:
				target_size = None
			if target_size is None:
				for t in self.transform_img.transforms:
					if isinstance(t, T.CenterCrop):
						target_size = t.size
						break
			if target_size is None:
				target_size = (self.image_size, self.image_size)
			if isinstance(target_size, int):
				target_size = (target_size, target_size)

			orig_resized = img.resize((target_size[0], target_size[1]), resample=Image.BILINEAR)
			orig_arr = np.array(orig_resized)
			if orig_arr.ndim == 2:
				orig_arr = np.stack([orig_arr] * 3, axis=-1)
			orig_t = torch.from_numpy(orig_arr).permute(2, 0, 1).contiguous()

			batch_imgs.append(img_t)
			batch_origs.append(orig_t)

			if len(batch_imgs) == self.batch_size:
				yield torch.stack(batch_imgs, dim=0), torch.stack(batch_origs, dim=0)
				batch_imgs = []
				batch_origs = []

		if batch_imgs and not self.drop_last:
			yield torch.stack(batch_imgs, dim=0), torch.stack(batch_origs, dim=0)


__all__ = ["ImageMaskDataset", "BatchedImageMaskIterable", "BatchedImageIterable", "get_dataloader"]


def _infer_image_size_from_model(model: torch.nn.Module, prefer: int = 256) -> int:
	"""Try to infer the model's expected input image size from its positional embeddings

	Strategy:
	- Find a parameter or buffer whose name contains 'positional_embedding' or 'pos_embed'.
	- Compute grid = sqrt(n_patches) where n_patches = embedding_len - 1.
	- Try to detect patch size from submodules (attribute 'patch_size' or conv kernel size).
	- If patch size cannot be detected, try common patch sizes and pick the image size
	  closest to `prefer`.
	"""
	# Search named parameters and buffers for positional embedding
	n_patches = None
	for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
		lname = name.lower()
		if "positional_embedding" in lname or "pos_embed" in lname or "positionalemb" in lname:
			# tensor shape is (1, n_patches+1, dim) typically
			try:
				n = tensor.shape[1]
			except Exception:
				continue
			if n >= 2:
				n_patches = n - 1
				break

	if n_patches is None:
		# try to find common CLIP-like attribute
		for nm, mod in model.named_modules():
			if hasattr(mod, "positional_embedding"):
				tensor = getattr(mod, "positional_embedding")
				try:
					n = tensor.shape[1]
					if n >= 2:
						n_patches = n - 1
						break
				except Exception:
					continue

	if n_patches is None:
		return prefer

	grid = int(math.sqrt(n_patches))
	if grid * grid != n_patches:
		# Not a perfect square; fallback
		return prefer

	# Try to detect patch size
	patch_size = None
	for nm, mod in model.named_modules():
		if hasattr(mod, "patch_size"):
			ps = getattr(mod, "patch_size")
			if isinstance(ps, (tuple, list)):
				patch_size = int(ps[0])
			else:
				patch_size = int(ps)
			break
		# look for conv kernel sizes commonly used for patch embedding
		if hasattr(mod, "kernel_size"):
			ks = getattr(mod, "kernel_size")
			if isinstance(ks, tuple) and len(ks) == 2:
				# Heuristic: if kernel_size is a typical patch size
				if ks[0] in (8, 14, 16, 32):
					patch_size = int(ks[0])
					break

	if patch_size is not None:
		return grid * patch_size

	# Otherwise try common patch sizes and pick closest to prefer
	candidates = {}
	for ps in (14, 16, 32, 8):
		candidates[ps] = grid * ps

	best_ps = min(candidates.items(), key=lambda kv: abs(kv[1] - prefer))[0]
	return candidates[best_ps]
