import cv2
from copy import deepcopy
import random
import numpy as np
import time

class WeatherAug:
    """
    A custom data augmentation class for YOLO Detection using OpenCV operations.
    
    This class provides a set of data augmentation techniques commonly used in object detection tasks.
    It includes different types of blurring, compression, and geometric transformations.
    
    Attributes:
        transforms (List[Callable]): A list of callable data augmentation functions.
        hyp (dict): Hyperparameters for augmentations.
    """
    def __init__(self, hyp=None,realtime=True):
        """
        Initializes the Cv2Aug class.

        Args:
            hyp (dict): Hyperparameters dictionary containing augmentation settings.
                Expected format:
                {
                    'blur_p': 0.5,  # Probability of applying blur
                    'blur_k': {3: 50, 5: 30, 7: 20},  # Kernel sizes and their probabilities
                    'gaussian_blur_p': 0.5,
                    'gaussian_blur_sigma': {1: 50, 2: 30, 3: 20},
                    'median_blur_p': 0.5,
                    'median_blur_k': {3: 50, 5: 30, 7: 20},
                    'motion_blur_p': 0.5,
                    'motion_blur_k': {7: 50, 9: 30, 11: 20},
                    'jpeg_quality_p': 0.5,
                    'jpeg_quality': {50: 50, 70: 30, 90: 20}
                }
        """
        self.hyp = hyp or {}
        self.transforms_realtime = [
            self.blur,
            self.median_blur,
            self.gaussian_blur,
            self.motion_blur, 
            self.salt_pepper_noise,
            self.brightness_contrast,
            self.sharpen,
            self.denoise,
            self.adaptive_sharpen
        ]
        self.transforms_prepro = [
            self.jpeg_compression,
            self.wave_transform,
            self.fourier_noise,
            self.speckle_noise,
            self.poisson_noise,
            self.bilateral_filter,
            self.defocus_blur,
            self.anisotropic_diffusion,
        ]
        self.isrealtime = realtime
    
    def _get_param(self, param_dict):
        """
        Helper function to get a random parameter based on a probability distribution.

        Args:
            param_dict (Union[dict, int, float]): A dictionary of values and their probabilities,
                                                  or a fixed value.

        Returns:
            The selected parameter value.
        """
        if isinstance(param_dict, (int, float)):
            return param_dict
        if isinstance(param_dict,(list,tuple)):  
            return abs(random.random() * (param_dict[1] - param_dict[0]) + param_dict[0])
        if isinstance(param_dict, dict):
            items = list(param_dict.items())
            values, weights = zip(*items)
            weights = [w/sum(weights) for w in weights]
            return random.choices(values, weights=weights)[0]
        print("param dict must be a dict, int, float, list, or tuple")
        print(param_dict)
        print(type(param_dict))
        return None

def __call__(self, img):
    """
    Applies data augmentation transformations based on realtime flag.

    Args:
        img: Input image to be augmented

    Returns:
        tuple: (augmented_image, transform_names)
    """
    transform_names = ''
    
    # Select transforms based on realtime flag
    transforms_to_use = self.transforms_realtime if self.isrealtime else self.transforms_prepro
    
    # Dictionary mapping transform names to their parameters
    transform_params = {
        'denoise': 'denoise_p',
        'blur': 'blur_p', 
        'gaussian_blur': 'gaussian_blur_p',
        'motion_blur': 'motion_blur_p',
        'median_blur': 'median_blur_p',
        'jpeg_compression': 'jpeg_quality_p',
        'wave_transform': 'wave_transform_p',
        'fourier_noise': 'fourier_noise_p',
        'salt_pepper_noise': 'salt_pepper_noise_p',
        'brightness_contrast': 'brightness_contrast_p',
        'speckle_noise': 'speckle_p',
        'poisson_noise': 'poisson_p',
        'bilateral_filter': 'bilateral_p',
        'radial_blur': 'radial_p',
        'defocus_blur': 'defocus_p',
        'anisotropic_diffusion': 'aniso_p',
        'sharpen': 'sharpen_p',
        'adaptive_sharpen': 'adaptive_sharpen_p',
    }
    
    # Apply transforms based on selected list
    for transform in transforms_to_use:
        transform_name = transform.__name__
        param = transform_params.get(transform_name)
        
        if random.random() < self.hyp.get(param, 0):
            t0 = time.time()
            img = transform(img)
            transform_names += transform_name[0]
            t1 = time.time()
            print(f"Time taken for {transform_name}: {t1 - t0:.2f} seconds")
    
    # Resize the image to the desired size
    img = cv2.resize(img, (self.hyp['imgsz'], self.hyp['imgsz']))
    
    return img, transform_names


    def blur(self, img):
        """Average blurring"""
        k = self._get_param(self.hyp.get('blur_k', {3: 50, 5: 30, 7: 20}))
        k = int(k)
        # print("k:",k)
        return cv2.blur(img, (k, k))

    def median_blur(self, img):
        """Median blurring"""
        k = self._get_param(self.hyp.get('median_blur_k', {3: 50, 5: 30, 7: 20}))
        k = int(k)
        k = k if k % 2 == 1 else k + 1  # Ensure k is odd
        # print("k:",k)
        return cv2.medianBlur(img, k)

    def gaussian_blur(self, img,):
        """Gaussian blurring"""
        k = self._get_param(self.hyp.get('gaussian_blur_k', 7))
        k = int(k)
        k = k if k % 2 == 1 else k + 1  # Ensure k is odd
                
        sigma = self._get_param(self.hyp.get('gaussian_blur_sigma', 1))
        # print("k:",k),print("sigma:",sigma)
        return cv2.GaussianBlur(img, (k, k), sigma)

    def motion_blur(self, img):
        """Motion blur effect"""
        k = self._get_param(self.hyp.get('motion_blur_k',(5,15)))
        k = int(k)
        k = k if k % 2 == 1 else k + 1  # Ensure k is odd
        # print("k:",k)
        kernel = np.zeros((k, k))
        kernel[int((k-1)/2), :] = np.ones(k)
        kernel = kernel / k
        return cv2.filter2D(img, -1, kernel)

    def jpeg_compression(self, img):
        """JPEG compression artifacts"""
        quality = self._get_param(self.hyp.get('jpeg_quality', (25,100)))
        quality = int(quality)
        # print("quality: ",quality)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        return cv2.imdecode(encimg, 1)

    def wave_transform(self, img):
        """
        Apply wave distortion to the image and optionally transform bounding boxes.
        """
        # Fetch parameters
        amplitude = self._get_param(self.hyp.get('wave_amplitude', {3: 50, 5: 25, 2: 25}))
        wavelength = self._get_param(self.hyp.get('wave_wavelength', {80: 50, 60: 25, 100: 25}))

        rows, cols = img.shape[:2]

        # Generate transformation map
        x_indices, y_indices = np.meshgrid(np.arange(cols), np.arange(rows))
        x_offsets = (amplitude * np.sin(2 * np.pi * y_indices / wavelength)).astype(np.float32)
        y_offsets = (amplitude * np.cos(2 * np.pi * x_indices / wavelength)).astype(np.float32)

        x_new = np.clip(x_indices + x_offsets, 0, cols - 1).astype(np.float32)
        y_new = np.clip(y_indices + y_offsets, 0, rows - 1).astype(np.float32)

        # Apply wave distortion to the image
        transformed_img = cv2.remap(img, x_new, y_new, interpolation=cv2.INTER_LINEAR)
        bboxes = self.bboxes
        # If no bounding boxes provided, return transformed image only
        if bboxes is None:
            return transformed_img

        # Transform bounding boxes
        transformed_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox

            # Get the four corners of the bounding box
            corners = np.array([
                [x1, y1],  # Top-left
                [x2, y1],  # Top-right
                [x1, y2],  # Bottom-left
                [x2, y2]   # Bottom-right
            ], dtype=np.float32)

            # Apply wave transformation to each corner
            transformed_corners = []
            for x, y in corners:
                dx = amplitude * np.sin(2 * np.pi * y / wavelength)
                dy = amplitude * np.cos(2 * np.pi * x / wavelength)

                new_x = np.clip(x + dx, 0, cols - 1)
                new_y = np.clip(y + dy, 0, rows - 1)

                transformed_corners.append([new_x, new_y])

            transformed_corners = np.array(transformed_corners)

            # Calculate new bounding box
            new_x1 = np.min(transformed_corners[:, 0])
            new_y1 = np.min(transformed_corners[:, 1])
            new_x2 = np.max(transformed_corners[:, 0])
            new_y2 = np.max(transformed_corners[:, 1])

            transformed_bboxes.append([int(new_x1), int(new_y1), int(new_x2), int(new_y2)])

        return transformed_img, transformed_bboxes

    def fourier_noise(self, img):
        """Add noise in frequency domain"""
        strength = self._get_param(self.hyp.get('fourier_strength', (0.1,0.8)))
        # print('strength:',strength)
        # Apply FFT
        f_transform = np.fft.fft2(img.astype(np.float32))
        f_transform = np.fft.fftshift(f_transform)
        
        # Create noise
        rows, cols = img.shape[:2]
        noise = np.random.normal(0, strength, (rows, cols, *img.shape[2:]))
        f_transform += noise
        
        # Inverse FFT
        img_back = np.fft.ifft2(np.fft.ifftshift(f_transform))
        img_back = np.abs(img_back).clip(0, 255).astype(np.uint8)
        
        return img_back

    def salt_pepper_noise(self, img):
        """Add salt and pepper noise"""
        density = self._get_param(self.hyp.get('sp_density',(0.003,0.02)))
        # print(density)
        noisy = np.copy(img)
        h, w = img.shape[:2]
        n_pixels = int(density * h * w)
        
        # Salt
        salt_coords = (np.random.randint(0, h, n_pixels),np.random.randint(0, w, n_pixels))
        noisy[salt_coords] = 255
        
        # Pepper
        pepper_coords = (np.random.randint(0, h, n_pixels),np.random.randint(0, w, n_pixels))
        noisy[pepper_coords] = 0
        
        return noisy

    def brightness_contrast(self, img):
        """Adjust brightness and contrast"""
        alpha = self._get_param(self.hyp.get('contrast', {0.5: 50, 1.5: 50, 2.0: 20, }))
        beta = self._get_param(self.hyp.get('brightness', {10: 50, 20: 30, 30: 20,-10: 50 }))
        # print('alpha: ',alpha),print('beta: ',beta)
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    def speckle_noise(self, img):
        """Add multiplicative noise"""
        intensity = self._get_param(self.hyp.get('speckle_intensity', {0.1: 50, 0.2: 30, 0.3: 20}))
        # print("intensity: ",intensity)
        noise = np.random.normal(1, intensity, img.shape)
        noisy = img * noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def poisson_noise(self, img):
        """Add Poisson noise"""
        vals = len(np.unique(img))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(img * vals) / float(vals)
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def bilateral_filter(self, img):
        """Apply bilateral filtering"""
           
        mean, std = cv2.meanStdDev(img)
        mean_val = int(mean[0][0])
        if mean_val > 100:
            d = 7
            sigma_color = 45
            sigma_space = 45
            return cv2.bilateralFilter(img, d, sigma_color, sigma_space)
            
        d = self._get_param(self.hyp.get('bilateral_d', {9: 50, 15: 30, 21: 20}))
        d = int(d)
        sigma_color = self._get_param(self.hyp.get('bilateral_sigma_color', {45: 50, 60: 30, 75: 20}))
        sigma_space = self._get_param(self.hyp.get('bilateral_sigma_space', {45: 50, 60: 30, 75: 20}))
        # print('d:', d, 'sigma_color:', sigma_color, 'sigma_space:', sigma_space)
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

    def radial_blur(self, img,):
        """Apply radial blur effect"""
        strength = self._get_param(self.hyp.get('radial_strength', {10: 50, 20: 30, 30: 20}))
        strength = int(strength)

        h, w = img.shape[:2]
        center_x, center_y = w // 2, h // 2

        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        kernel_size = np.clip((dist_from_center / strength), 1, 20).astype(np.int32)

        # Precompute Gaussian kernels for unique kernel sizes
        unique_kernel_sizes = np.unique(kernel_size)
        gaussian_kernels = {
            k: cv2.getGaussianKernel(k, -1) @ cv2.getGaussianKernel(k, -1).T
            for k in unique_kernel_sizes if k % 2 == 1
        }

        # Apply Gaussian blur using precomputed kernels
        padded_img = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_REFLECT)
        blurred = np.zeros_like(img)

        for k in unique_kernel_sizes:
            if k % 2 == 0:
                continue
            mask = (kernel_size == k)
            if not np.any(mask):
                continue
            coords = np.argwhere(mask)
            for y, x in coords:
                y_p, x_p = y + 20, x + 20
                roi = padded_img[y_p - k//2:y_p + k//2 + 1, x_p - k//2:x_p + k//2 + 1]
                blurred[y, x] = np.sum(roi * gaussian_kernels[k])

        return blurred

    def defocus_blur(self, img,):
        """Apply defocus blur"""
        k = self._get_param(self.hyp.get('defocus_size', {3: 50, 5: 30, 7: 20}))
        # print('k:', k)
        k = int(k)
        k = k if k % 2 == 1 else k + 1
        kernel = np.zeros((k, k), np.uint8)
        cv2.circle(kernel, (k//2, k//2), k//2, 1, -1)
        kernel = kernel / kernel.sum()
        return cv2.filter2D(img, -1, kernel)

    def anisotropic_diffusion(self, img,):
        """Apply anisotropic diffusion filtering"""
        num_iter = self._get_param(self.hyp.get('aniso_iter',{5: 50, 3: 30, 15: 20}))
        num_iter = int(num_iter)
        gamma = self._get_param(self.hyp.get('aniso_gamma', {0.15: 50, 0.25: 30, 0.35: 20}))
        kappa = self._get_param(self.hyp.get('aniso_kappa', {25: 50, 35: 30, 45: 20}))
        kappa = int(kappa)
        # print(('num_iter:', num_iter, 'gamma:', gamma, 'kappa:', kappa))
        img_float = img.astype(np.float32)
        for _ in range(num_iter):
            gradN = np.roll(img_float, shift=1, axis=0) - img_float
            gradS = np.roll(img_float, shift=-1, axis=0) - img_float
            gradE = np.roll(img_float, shift=1, axis=1) - img_float
            gradW = np.roll(img_float, shift=-1, axis=1) - img_float
            
            cN = np.exp(-(gradN/kappa)**2)
            cS = np.exp(-(gradS/kappa)**2)
            cE = np.exp(-(gradE/kappa)**2)
            cW = np.exp(-(gradW/kappa)**2)
            
            img_float = img_float + gamma*(cN*gradN + cS*gradS + cE*gradE + cW*gradW)
            
        return np.clip(img_float, 0, 255).astype(np.uint8)    

    def sharpen(self, img):
        """Apply sharpening using unsharp masking"""
        kernel_size = self._get_param(self.hyp.get('sharpen_kernel', {3: 50, 5: 30, 7: 20}))
        amount = self._get_param(self.hyp.get('sharpen_amount', (0.5, 2.0)))
        
        blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def denoise(self, img):
        """Apply denoising using Non-local Means Denoising"""
        h = self._get_param(self.hyp.get('denoise_h', {10: 50, 15: 30, 20: 20}))
        
        return cv2.fastNlMeansDenoisingColored(img, None, h, h, 2, 3)

    def adaptive_sharpen(self, img):
        """Adaptive sharpening based on local variance"""
        kernel_size = self._get_param(self.hyp.get('adaptive_sharp_kernel', {3: 50, 5: 30}))
        strength = self._get_param(self.hyp.get('adaptive_sharp_strength', (0.5, 2.0)))
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        detail = cv2.subtract(gray, blurred)
        enhanced = cv2.add(gray, cv2.multiply(detail, strength))
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

