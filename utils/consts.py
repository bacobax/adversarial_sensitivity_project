import os

# Project root directory (detectors/)
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Supported attacks and image types
SUPPORTED_ATTACKS = {'pgd', 'fgsm', 'deepfool'}
SUPPORTED_IMAGE_TYPES = {'real', 'samecat', 'diffcat'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# Detector configurations
DETECTOR_MAP = {
    'AnomalyOV': ('AnomalyOVDetector', os.path.join('models', 'anomaly_ov', 'detector.py')),
    'R50_nodown': ('R50NoDownDetector', os.path.join('models', 'R50_nodown', 'detector.py')),
    'WaveRep': ('WaveRepDetector', os.path.join('models', 'WaveRep', 'detector.py')),
}
