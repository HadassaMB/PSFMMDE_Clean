from skimage.metrics import structural_similarity as ssim
import lpips
import wandb
import numpy as np
import torch
import gc
import math
import matplotlib.pyplot as plt
# Metric functions
def compute_metrics(depth_pred, depth_gt, rgb_pred, rgb_gt):
    # Depth metrics
    depth_rmse = np.sqrt(np.mean((depth_pred - depth_gt) ** 2))
    delta = np.maximum(depth_pred / (depth_gt + 1e-8), depth_gt / (depth_pred + 1e-8))
    depth_delta1 = np.mean(delta < 1.25); depth_delta2 = np.mean(delta < 1.25**2); depth_delta3 = np.mean(delta < 1.25**3)
    depth_log10 = np.mean(np.abs(np.log10(depth_pred + 1e-8) - np.log10(depth_gt + 1e-8)))
    depth_AbsRel = np.mean(np.abs(depth_pred - depth_gt) / (depth_gt + 1e-8))
    depth_metrics = {"Depth RMSE": depth_rmse,
                     "Depth Delta1": depth_delta1,
                     "Depth Delta2": depth_delta2,
                     "Depth Delta3": depth_delta3,
                     "Depth Log10": depth_log10,
                     "Depth AbsRel": depth_AbsRel}
    # RGB metrics
    rgb_mse = np.mean((rgb_pred - rgb_gt) ** 2)
    rgb_psnr = 10 * np.log10((255 ** 2) / (rgb_mse + 1e-8))
    rgb_ssim = ssim(rgb_pred, rgb_gt, data_range=rgb_pred.max() - rgb_pred.min(), channel_axis=-1)
    # rgb_pred_t = torch.tensor(rgb_pred).permute(2,0,1).unsqueeze(0).float()*2-1
    # rgb_gt_t   = torch.tensor(rgb_gt).permute(2,0,1).unsqueeze(0).float()*2-1
    # lpips_model = lpips.LPIPS(net='alex')  # or 'vgg'
    # rgb_lpips = lpips_model(rgb_pred_t, rgb_gt_t).item()
    rgb_metrics = {"RGB MSE": rgb_mse,
                   "RGB PSNR": rgb_psnr,
                   "RGB SSIM": rgb_ssim,
                   # "RGB LPIPS": rgb_lpips
    }
    return depth_metrics, rgb_metrics

# Logging function
def log_pred_gt(rgb_pred, rgb_gt, depth_pred, depth_gt, step_info):
    # Convert to numpy
    rgb_pred_np = (rgb_pred.detach().permute(1,2,0).cpu().numpy()* 255).astype(np.uint8)
    rgb_gt_np = (rgb_gt.detach().permute(1,2,0).cpu().numpy()* 255).astype(np.uint8)
    depth_pred_np = (depth_pred.detach().squeeze(0).cpu().numpy()* 65535).astype(np.uint16)
    depth_gt_np = (depth_gt.detach().squeeze(0).cpu().numpy()* 65535).astype(np.uint16)

    # Stack horizontally: RGB pred | RGB GT | Depth pred | Depth GT
    depth_pred_rgb = np.stack([depth_pred_np]*3, axis=-1)
    depth_gt_rgb = np.stack([depth_gt_np]*3, axis=-1)
    combined_img = np.hstack([rgb_pred_np, rgb_gt_np, depth_pred_rgb, depth_gt_rgb])

    # Compute metrics
    depth_metrics, rgb_metrics = compute_metrics(depth_pred_np, depth_gt_np, rgb_pred_np, rgb_gt_np)

    # Log to wandb
    wandb.log({
        "Prediction vs GT": [wandb.Image(combined_img, caption="RGB pred | RGB GT | Depth pred | Depth GT")],
        **depth_metrics,
        **rgb_metrics,
        **step_info
    })

# Logging function
def log_pred_gt_V2(rgb_pred, rgb_gt, depth_pred, depth_gt, step_info):
    # Convert to numpy
    rgb_pred_np = rgb_pred.permute(1,2,0).cpu().numpy() # ).astype(np.uint8)
    rgb_gt_np = rgb_gt.permute(1,2,0).cpu().numpy() # ).astype(np.uint8)
    depth_pred_np = depth_pred.squeeze(0).cpu().numpy() # ).astype(np.uint16)
    depth_gt_np = depth_gt.squeeze(0).cpu().numpy() # ).astype(np.uint16)

    # Stack horizontally: RGB pred | RGB GT | Depth pred | Depth GT
    # depth_pred_rgb = np.stack([depth_pred_np]*3, axis=-1)
    # depth_gt_rgb = np.stack([depth_gt_np]*3, axis=-1)
    # rgb_diff = np.abs(rgb_pred_np - rgb_gt_np)
    # depth_diff = np.abs(depth_pred_rgb - depth_gt_rgb)
    # combined_img = np.round(255*np.hstack([rgb_pred_np, rgb_gt_np, rgb_diff, depth_pred_rgb, depth_gt_rgb, depth_diff])).astype(np.uint8)
    # Compute metrics
    depth_metrics, rgb_metrics = compute_metrics(depth_pred_np, depth_gt_np, rgb_pred_np, rgb_gt_np)

    # Log to wandb
    wandb.log({
        # "Prediction vs GT": [wandb.Image(combined_img, caption="RGB pred | RGB GT | RGB diff | Depth pred | Depth GT | Depth diff")],
        **depth_metrics,
        **rgb_metrics,
        **step_info
    })
    del rgb_pred_np, rgb_gt_np, depth_pred_np, depth_gt_np