
img_dir = "./data/COCO_inpainted"

#get all the filenames inside the img_dir
import os
file_names = os.listdir(img_dir)
ids = set(fn.split(".")[0] for fn in file_names)

mask_dir = "./data/masks_and_bbox/mask"
mask_fnames = os.listdir(mask_dir)
for fn in mask_fnames:
    id_ = fn.split(".")[0]
    if id_ not in ids:
        os.remove(os.path.join(mask_dir, fn))
        print(f"Removed mask file: {fn}")
    