from utils.dataset import get_dataloader
from utils.vulnerability_map import get_vulnerability_map, visualize_vulnerability_map, plot_triple_res
from utils.nets import load_network
import matplotlib.pyplot as plt
from pprint import pprint
if __name__ == '__main__':
    device = "mps"

    weights_dir = './weights/weights_AdversarialRobustnessCLIP'
    detector = load_network('OJHA_latent_clip', weights_dir).to(device)

    # Create dataloader AFTER loading the model so we can infer image size
    images_dir: str = "./data/COCO_inpainted"
    masks_dir: str = "./data/masks"
    dataloader = get_dataloader(
        images_dir,
        masks_dir,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        model=detector,
        transform_img=detector.preprocess
    )
    
    is_resnet = type(detector).__name__ == 'ResNet'
    print(f"is_resnet={is_resnet}")

    for i, (image, mask, orig) in enumerate(dataloader):
        # use the un-normalized original resized image (`orig`) instead of the
        # normalized/processed `image` when visualizing or computing vulnerability
        # maps. `orig` is a uint8 tensor with shape (B, 3, H, W).
        print(f"BATCH {i}: processed_image.shape={image.shape}, mask.shape={mask.shape}, orig.shape={orig.shape}")
        pprint(orig)

        # If `get_vulnerability_map` expects normalized floats, but you want to
        # feed the original image, ensure the function can accept uint8 input.
        # Here we pass `orig` as requested by the user.
        vuln_map, res = get_vulnerability_map(image, mask, detector, is_resnet=is_resnet, device=device)
        vis = visualize_vulnerability_map(vuln_map, orig)
        plot_triple_res(orig, vis, mask)

