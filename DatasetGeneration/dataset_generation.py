import numpy as np
import torch
import time, os
from typing import Tuple, List, Iterable, Optional, Any
import cv2
import scipy.io as sio
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt

def nn_quantization(depth_map: np.ndarray, depths_sampled: np.ndarray, rgb_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour quantization of a depth map using provided stack depths.

    Returns (imset_quant, quantized_map) where `imset_quant` is an array of
    per-level RGB cuts with shape (D, H, W, 3) and `quantized_map` has the
    same shape as `depth_map` with the chosen sample depth for each pixel.
    """
    idx = np.searchsorted(depths_sampled, depth_map)
    idx = np.clip(idx, 1, len(depths_sampled) - 1)
    l = depths_sampled[idx - 1]
    r = depths_sampled[idx]
    idx = np.where(np.abs(depth_map - l) <= np.abs(depth_map - r), idx - 1, idx)
    quantized = depths_sampled[idx]
    levels = np.unique(quantized)
    imset_quant = np.array([
        np.transpose([(quantized == level) * rgb_image[:, :, c] for c in range(3)], (1, 2, 0))
        for level in levels
    ])
    return imset_quant, quantized

def propagate_ASM(Uin, lambda_um, z_um, dx_um):
    [Ny, Nx] = Uin.shape
    k = 2*np.pi/lambda_um
    fx = np.linspace(-np.floor(Nx/2), np.ceil(Nx/2)-1, Nx)/(Nx*dx_um)
    fy = np.linspace(-np.floor(Ny/2), np.ceil(Ny/2)-1, Ny)/(Ny*dx_um)
    [FX, FY] = np.meshgrid(fx, fy)

    arg = 1 - (lambda_um*FX)**2 - (lambda_um*FY)**2
    arg[arg<0] = 0
    kz = k * np.sqrt(arg)
    H = np.exp(1j * z_um * kz)
    Uin_f = np.fft.fft2(Uin)
    Uout  = np.fft.ifft2(Uin_f*H)
    return Uout 

# Interfaces and implementations for depth quantization
class DepthQuantizer:
    """
    Interface for depth quantization. Implementations should produce an image-set tensor and a quantized depth map.
    """
    def quantize(self, im: np.ndarray, depth: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        raise NotImplementedError

class LogSpaceDepthQuantizer(DepthQuantizer):
    def __init__(self, min_depth: float, max_depth: float, zero_pos: float, device: torch.device, N_depths: Optional[int] = 10, quantization_method: str = "logspace", MaxLloyd_V: Optional[str] = "V3"):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.zero_pos = zero_pos
        self.device = device
        self.N_depths = N_depths
        self.quantization_method = quantization_method # from ["logspace", "MaxLloyd"]
        self.MaxLloyd_V = MaxLloyd_V
        self._last_quantized = None  # Cache for visualization

    def quantize(self, im: np.ndarray, depth: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """Quantize `depth` into discrete levels and produce an image-set tensor.

        - For `StackPSFGenerator` uses nearest-neighbour to stack depths.
        - Otherwise uses local MaxLloyd implementation.
        
        Caches the quantized depth map in self._last_quantized for visualization.
        """
        depths_sampled = np.logspace(np.log10(self.min_depth), np.log10(self.max_depth), self.N_depths) * 100
        imset_quant_all, quantized = nn_quantization(depth, depths_sampled, im)
        self._last_quantized = quantized.copy()

        unique_depths_all = np.unique(quantized) * 0.01 - self.zero_pos
        mask = (unique_depths_all >= (self.min_depth - self.zero_pos)) & (unique_depths_all <= (self.max_depth - self.zero_pos))
        imset_quant = imset_quant_all[mask].copy()
        unique_depths = unique_depths_all[mask].copy()

        imset_tensor = torch.stack([torch.tensor(im_i, dtype=torch.float32).permute(2, 0, 1) / 255.0 for im_i in imset_quant]).to(self.device)
        return imset_tensor, unique_depths

# Interfaces and implementations for PSF generation
class PSFGeneratorInterface:
    """
    Interface for PSF generators. Implementations should return a tensor of PSFs with shape (D, Hk, Wk).
    """

    def get_PSFs(self, unique_depths: np.ndarray) -> torch.Tensor:
        raise NotImplementedError

class ModelPSFGenerator(PSFGeneratorInterface):
    def __init__(self, cam_params: dict, device: torch.device):
        self.mask_path = cam_params['mask_path']
        self.Iris = cam_params['Iris']
        self.wavelength = cam_params['wavelength']
        self.p_camera = cam_params['p_camera']
        self.ROI_camera = cam_params['ROI_camera']
        self.f_lens = cam_params['f_lens']
        self.D = cam_params['D']
        self.zero_pos = cam_params['zero_pos'] # [m]
        self.device = device

    def _get_z_defocus(self) -> float:
        return 1. / (1. / self.f_lens - 1. / (self.zero_pos*1e6)) - self.f_lens 
    
    def extract_psf_model(self) -> Any:
        R_mask = self.D / 2
        Mask_Load = sio.loadmat(self.mask_path)  # expects: d_mask, Xbfp, Ybfp, lambda_mask, dn_mask
        d_mask = Mask_Load['d_mask']; Xbfp = Mask_Load['Xbfp']; Ybfp = Mask_Load['Ybfp']
        dx_bfp = Xbfp[0, 1] - Xbfp[0, 0]
        #################### Conserve grid before and after Fourier #####################
        N0 = d_mask.shape[0]
        N_min_pupil = np.ceil((2 * N0) / dx_bfp)  # + 2 * 32 # Pathological case to avoid
        N_cam_target = np.ceil(self.wavelength * self.f_lens / (dx_bfp * self.p_camera))
        N_target = max(N_min_pupil, N_cam_target)
        if N_target < N0:
            N_target = N_target * np.ceil(N0 / N_target)
        pv = int(max(0, np.ceil((N_target - N0) / 2)))
        #######################################################################
        xobj = Xbfp[0, :]; p_obj = (xobj[1] - xobj[0])
        xbfp = np.linspace(0, N_target - 1, int(N_target)) * p_obj
        ybfp = xbfp.copy()
        xbfp = xbfp - (abs(xbfp[-1]) - abs(xbfp[0])) / 2
        ybfp = ybfp - (abs(ybfp[-1]) - abs(ybfp[0])) / 2
        dx_bfp = xbfp[1] - xbfp[0]
        [Xbfp, Ybfp] = np.meshgrid(xbfp, ybfp)
        mask_aper_cond = (Xbfp ** 2 + Ybfp ** 2) > R_mask ** 2
        iris_aper_cond = (Xbfp ** 2 + Ybfp ** 2) > (self.Iris / 2) ** 2
        Mask_amplitude = np.ones(Xbfp.shape)  # Ebfp
        Mask_amplitude[mask_aper_cond] = 0
        Mask_amplitude[iris_aper_cond] = 0
        Mask_phase = np.pad(d_mask, pad_width=((pv, pv), (pv, pv)), mode='constant', constant_values=0)
        if Mask_phase.shape[0] > N_target:
            Mask_phase = np.delete(Mask_phase, -1, axis=0)
            Mask_phase = np.delete(Mask_phase, -1, axis=1)
        Mask_function = Mask_amplitude * np.exp(1j * Mask_phase)
        return Xbfp, Ybfp, Mask_function, pv, dx_bfp
    
    def get_PSFs(self, unique_depths: np.ndarray) -> torch.Tensor:
        Xbfp, Ybfp, Mask_function, pv, dx_bfp = self.extract_psf_model()
        PSFs_stack = []
        for z_obj in unique_depths:
            z_obj_um = z_obj * 1e6
            phase_bfp = 2 * np.pi / self.wavelength / (2 * z_obj_um) * (Xbfp**2 + Ybfp**2)

            Ebfp = np.pad(Mask_function*np.exp(1j * phase_bfp), pad_width=((pv, pv), (pv, pv)),mode='constant',constant_values=0)

            dx_img = self.wavelength *self.f_lens / (Ebfp.shape[1] * dx_bfp)
            Eimg = propagate_ASM(np.fft.fftshift(np.fft.fft2( Ebfp )), self.wavelength , self._get_z_defocus(), dx_img)

            Iimg = np.abs(Eimg)**2
            scale = dx_img / self.p_camera
            I_cam = cv2.resize(Iimg, None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)

            a1 = int(np.floor(I_cam.shape[0]/2) - np.floor(self.ROI_camera/2))
            a2 = int(np.floor(I_cam.shape[0]/2) + np.ceil(self.ROI_camera/2))
            b1 = int(np.floor(I_cam.shape[1]/2) - np.floor(self.ROI_camera/2))
            b2 = int(np.floor(I_cam.shape[1]/2) + np.ceil(self.ROI_camera/2))
            
            efbp_clean = Ebfp.copy(); efbp_clean[np.abs(np.real(efbp_clean)) < 1e-10] = 0
            """
            plt.subplot(1,3,1); plt.imshow(np.angle(efbp_clean)); plt.title(f'Phase at BFP [rad], z={z_obj_m}[m]')
            plt.subplot(1,3,2); plt.imshow(Mask_phase); plt.title(f'Mask phase without unwrapping [rad]')
            plt.subplot(1,3,3); plt.imshow(I_cam[a1:a2, b1:b2]); plt.title(f'PSF, z={z_obj_m}[m]'); plt.show()
            """
            PSFs_stack.append(I_cam[a1:a2, b1:b2])
        psfs = torch.tensor(PSFs_stack, device=self.device).float()
        return psfs / psfs.sum(dim=(1, 2), keepdim=True)
    
class DepthAwareImageEncoder:
    def __init__(self, up_factor: int = 1, device: Optional[torch.device] = 'cpu'):
        self.up_factor = up_factor
        self.device = device

    def encode_slow(self, imset_tensor: torch.Tensor, PSFs) -> Tuple[torch.Tensor, np.ndarray]:
        """Apply depth-varying PSF convolution.
        Returns (encoded_up_tensor, encoded_down_numpy).
        """
        # Efficient grouped convolution:
        # - imset_tensor: (D, 3, H, W)
        # - PSFs: tensor (D, Hk, Wk)
        imset_up = F.interpolate(imset_tensor, scale_factor=self.up_factor, mode="nearest")
        device = self.device

        D, C, H, W = imset_up.shape

        # Merge depth and color channels: (1, D*C, H, W)
        im_group = imset_up.view(1, D * C, H, W)

        # Prepare kernels: for each depth, repeat the PSF per color channel
        # PSFs: (D, Hk, Wk) -> kernels: (D*C, 1, Hk, Wk)
        psf_kernels = torch.stack([psf.to(device) for psf in PSFs])  # (D, Hk, Wk)
        psf_kernels = psf_kernels.unsqueeze(1).repeat(1, C, 1, 1)  # (D, C, Hk, Wk)
        kernels = psf_kernels.view(D * C, 1, psf_kernels.shape[2], psf_kernels.shape[3]).to(device)

        padding = kernels.shape[-1] // 2
        out = F.conv2d(im_group, kernels, padding=padding, groups=D * C)

        # out shape: (1, D*C, H, W) -> reshape to (D, C, H, W)
        out = out.view(D, C, H, W)

        # Sum across depths to get final encoded image (C, H, W)
        im_final = out.sum(dim=0)

        encoded_up = im_final.clamp(0.0, 1.0)

        # Downsample back to original size
        encoded_down_t = F.interpolate(im_final.unsqueeze(0), scale_factor=1 / self.up_factor, mode="nearest").squeeze(0)
        encoded_down = encoded_down_t.permute(1, 2, 0).clamp(0.0, 1.0)
        return encoded_up, encoded_down.cpu().numpy()
    
    def encode(self, imset_tensor: torch.Tensor, PSFs: list):
        imset_up = F.interpolate(imset_tensor, scale_factor=self.up_factor, mode='nearest')  # Nx3xHxW
        im_final = None
        for im_d, PSF_d in zip(imset_up, PSFs):
            im_d = im_d.unsqueeze(0)
            PSF_kernel = PSF_d.unsqueeze(0).unsqueeze(0).repeat(3,1,1,1)
            im_conv = F.conv2d(im_d, PSF_kernel, padding=PSF_d.shape[0]//2, groups=3).squeeze(0)/255
            if im_final is None:
                im_final = im_conv
            else:
                im_final += im_conv
        encoded_up = im_final.clamp(0,1)
        # Downsample back to original size
        encoded_down = F.interpolate(im_final.unsqueeze(0), scale_factor=1/self.up_factor, mode='nearest').squeeze(0).permute(1,2,0).clamp(0,1)
        return encoded_up, encoded_down
    
class DepthAwareConvolver:
    """Orchestrates depth-aware PSF convolution simulation.

    Combines PSF generation, depth quantization, and depth-aware encoding
    to produce realistic image simulations with depth-dependent blur.
    """

    def __init__(self, psf_generator: PSFGeneratorInterface, depth_quantizer: DepthQuantizer, encoder: DepthAwareImageEncoder):
        self.psf_generator = psf_generator
        self.depth_quantizer = depth_quantizer
        self.encoder = encoder

        # Cached results from last simulation
        self.imset_tensor: Optional[torch.Tensor] = None
        self.unique_depths: Optional[np.ndarray] = None
        self.quantized_depth: Optional[np.ndarray] = None
        self.encoded_im_up: Optional[torch.Tensor] = None
        self.encoded_im: Optional[np.ndarray] = None
        self.last_elapsed_time: float = 0.0
        self.last_image_shape: Optional[Tuple] = None

    def simulate_image(self, im: np.ndarray, depth: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """Simulate a single image with depth-aware PSF convolution.

        Args:
            im: RGB image (H, W, 3) in uint8 or float [0, 255].
            depth: Depth map (H, W) in cm or consistent depth units.

        Returns:
            (encoded_up, encoded_down): upsampled tensor (3, H_up, W_up) and
            downsampled numpy array (H, W, 3) in [0, 1].
        """
        t_start = time.time()
        self.last_image_shape = im.shape

        imset_tensor, unique_depths = self.depth_quantizer.quantize(im, depth)
        
        # Cache quantized depth from quantizer (now always available)
        self.quantized_depth = self.depth_quantizer._last_quantized
        
        PSFs = self.psf_generator.get_PSFs(unique_depths)
        encoded_up, encoded_down = self.encoder.encode(imset_tensor, PSFs)

        t_end = time.time()
        self.last_elapsed_time = t_end - t_start

        # Cache results for inspection
        self.imset_tensor = imset_tensor
        self.unique_depths = unique_depths
        self.encoded_im_up = encoded_up
        self.encoded_im = encoded_down

        return encoded_up, encoded_down

    def simulate_batch(self, images: Iterable[np.ndarray], depths: Iterable[np.ndarray], verbose: bool = True) -> List[Tuple[torch.Tensor, np.ndarray]]:
        """Simulate a batch of images with progress logging.

        Args:
            images: Iterable of RGB images (H, W, 3).
            depths: Iterable of depth maps (H, W).
            verbose: Log progress if True.

        Returns:
            List of (encoded_up, encoded_down, metadata) tuples where metadata
            contains 'time_sec', 'size', 'unique_depths'.
        """
        results = []
        for idx, (im, depth) in enumerate(zip(images, depths)):
            t_start = time.time()
            result_up, result_down = self.simulate_image(im, depth)
            elapsed = time.time() - t_start
            
            metadata = {
                "time_sec": elapsed,
                "size": im.shape,
                "unique_depths": self.unique_depths.tolist() if self.unique_depths is not None else [],
            }
            results.append((result_up, result_down, metadata))
            print(f"Image {idx + 1}: {im.shape} processed in {elapsed:.2f}s ")
        print(f"Batch complete: {len(results)} images processed")
        return results

    def simulate_batch_to_disk(
        self,
        images: Iterable[np.ndarray],
        depths: Iterable[np.ndarray],
        output_dir: str,
        prefix: str = "encoded",
    ) -> List[str]:
        """Simulate batch and write outputs to disk as PNG.

        Args:
            images: Iterable of RGB images.
            depths: Iterable of depth maps.
            output_dir: Directory to write results.
            prefix: Filename prefix (e.g., "encoded_0.png", "encoded_1.png").

        Returns:
            List of written file paths.
        """

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        written = []
        results = self.simulate_batch(images, depths, verbose=True)
        for idx, (_, encoded_down, metadata) in enumerate(results):
            out_path = output_dir / f"{prefix}_{idx:06d}.png"
            # Convert [0, 1] to [0, 255] and BGR for cv2
            bgr = cv2.cvtColor(
                (np.clip(encoded_down, 0.0, 1.0) * 255).astype(np.uint8),
                cv2.COLOR_RGB2BGR,
            )
            cv2.imwrite(str(out_path), bgr)
            written.append(str(out_path))
            print(f"Saved: {out_path} (processed in {metadata['time_sec']:.2f}s)")
        return written
    
if __name__ == "__main__":
    camera_properties = {
                            "mask_path": r"C:\Users\hadassa-m\Desktop\MSc\PSFMMDE-Local\DatasetGeneration\DH_Leonid.mat",
                            "wavelength": 530e-3,
                            "z_defocus": 145.13788098693476,
                            "p_camera": 4.3,
                            "D": 32e3,
                            "f_lens": 100e3,
                            "ROI_camera": 101,
                            "Iris": 26e3,
                            "min_depth": 2.0,
                            "max_depth": 100.0,
                            "zero_pos": 69.0
                        }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    psf_generator = ModelPSFGenerator(cam_params=camera_properties, device=device)
    depth_quantizer = LogSpaceDepthQuantizer(min_depth=0.1, max_depth=100.0, zero_pos=0.0, device=device, N_depths=10)
    encoder = DepthAwareImageEncoder(up_factor=1, device=device)
    convolver = DepthAwareConvolver(psf_generator, depth_quantizer, encoder)
    # Load example image and depth (replace with actual paths)
    im = cv2.imread(r"C:\Users\hadassa-m\Desktop\MSc\PSFMMDE-Local\Datasets\vkittiv2\rgbs\rgb_00000.jpg")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(r"C:\Users\hadassa-m\Desktop\MSc\PSFMMDE-Local\Datasets\vkittiv2\depths\depth_00000.png", cv2.IMREAD_ANYDEPTH)

    encoded_up, encoded_down = convolver.simulate_image(im, depth)

    plt.subplot(1, 2, 1); plt.imshow(encoded_up.permute(1, 2, 0).cpu().numpy()); plt.title('Encoded Up (Tensor)')
    plt.subplot(1, 2, 2); plt.imshow(encoded_down); plt.title('Encoded Down (Numpy)'); plt.show()