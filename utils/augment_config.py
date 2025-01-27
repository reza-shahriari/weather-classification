PREPROCESSING_HYP = {
    # Slow but high-quality augmentations for preprocessing
    'blur_vs_sharpen_p': 0.5,
    
    # Noise augmentations
    'fourier_noise_p': 0.3,
    'fourier_strength': (0.1, 0.8),
    'salt_pepper_noise_p': 0.3,
    'sp_density': (0.003, 0.02),
    'speckle_p': 0.3,
    'speckle_intensity': {0.1: 50, 0.2: 30, 0.3: 20},
    'poisson_p': 0.3,
    
    # Complex blurs
    'radial_p': 0.3,
    'radial_strength': {10: 50, 20: 30, 30: 20},
    'defocus_p': 0.3,
    'defocus_size': {3: 50, 5: 30, 7: 20},
    'aniso_p': 0.3,
    'aniso_iter': {5: 50, 3: 30, 15: 20},
    'aniso_gamma': {0.15: 50, 0.25: 30, 0.35: 20},
    'aniso_kappa': {25: 50, 35: 30, 45: 20},
    
    # Advanced sharpening
    'denoise_p': 0.3,
    'denoise_h': {10: 50, 15: 30, 20: 20},
    'adaptive_sharpen_p': 0.3,
    'adaptive_sharp_kernel': {3: 50, 5: 30},
    'adaptive_sharp_strength': (0.5, 2.0),
    
    'imgsz': 640,
}

REALTIME_HYP = {
    # Fast augmentations suitable for real-time
    'blur_vs_sharpen_p': 0.5,
    
    # Basic blurs
    'blur_p': 0.3,
    'blur_k': {3: 50, 5: 30, 7: 20},
    'gaussian_blur_p': 0.3,
    'gaussian_blur_k': 7,
    'gaussian_blur_sigma': 1,
    'median_blur_p': 0.3,
    'median_blur_k': {3: 50, 5: 30, 7: 20},
    
    # Simple enhancements
    'brightness_contrast_p': 0.5,
    'contrast': {0.5: 50, 1.5: 50, 2.0: 20},
    'brightness': {10: 50, 20: 30, 30: 20, -10: 50},
    'WB_Hot_p': 0.3,
    
    # Basic sharpening
    'sharpen_p': 0.3,
    'sharpen_kernel': {3: 50, 5: 30, 7: 20},
    'sharpen_amount': (0.5, 2.0),
    'filter2D_p': 0.3,
    
    'imgsz': 640,
}
