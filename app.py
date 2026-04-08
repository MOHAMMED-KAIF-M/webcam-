import base64
import importlib
import io
import json
import os
import sys
import threading
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, url_for
from PIL import Image

app = Flask(__name__)
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
APP_ROOT = Path(__file__).resolve().parent
YOLOV5_REPO = APP_ROOT / 'legacy_yolov5'
MODELS_DIR = APP_ROOT / 'mod'
LEGACY_MODELS_DIR = APP_ROOT / 'models'
FACE_CASCADE_PATH = MODELS_DIR / 'haarcascade_frontalface_default.xml'
EMOTION_MODEL_H5_PATH = MODELS_DIR / 'model.h5'
EMOTION_MODEL_SAFE_PATH = MODELS_DIR / 'model.safetensors'
EMOTION_MODEL_BIN_PATH = MODELS_DIR / 'pytorch_model.bin'
ACTION_MODEL_PATH = MODELS_DIR / 'action_best_model.pt'
ACTION_LABEL_MAP_PATH = MODELS_DIR / 'action_label_map.json'
ACTION_HF_MODEL_DIR = MODELS_DIR / 'action_hf_model'
POSE_MODEL_PATH = MODELS_DIR / 'keypointrcnn_resnet50_fpn_coco-fc266e95.pth'
LEGACY_FACE_CASCADE_PATH = APP_ROOT / 'haarcascade_frontalface_default.xml'
LEGACY_EMOTION_MODEL_H5_PATH = APP_ROOT / 'model.h5'
LEGACY_EMOTION_MODEL_SAFE_PATH = APP_ROOT / 'model.safetensors'
LEGACY_EMOTION_MODEL_BIN_PATH = APP_ROOT / 'pytorch_model.bin'
LEGACY_POSE_MODEL_PATH = APP_ROOT / 'keypointrcnn_resnet50_fpn_coco-fc266e95.pth'
EMOTION_LABELS_PATH = MODELS_DIR / 'emotion_labels.json'
LEGACY_EMOTION_LABELS_PATH = APP_ROOT / 'emotion_labels.json'
LEGACY_ACTION_MODEL_PATH = APP_ROOT / 'best_model.pt'
LEGACY_ACTION_LABEL_MAP_PATH = APP_ROOT / 'label_map.json'
ACTION_PROJECT_DIR = APP_ROOT.parent / 'Human Action Recognition'
ACTION_OUTPUT_DIR = ACTION_PROJECT_DIR / 'outputs'
GENERATED_MEDIA_DIR = APP_ROOT / 'generated_media'

# This default order matches common FER2013-style Keras checkpoints. Override it
# by placing an emotion_labels.json file in the mod folder if needed.
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HUMAN_CLASS_LABELS = {'human', 'person'}
IMAGE_FILE_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.webp'}
VIDEO_FILE_EXTENSIONS = {'.avi', '.mkv', '.mov', '.mp4', '.mpeg', '.mpg', '.webm'}
VIDEO_MAX_INFERENCE_FRAMES = 60
VIDEO_MAX_PROCESSING_EDGE = 640
POSE_DETECTION_THRESHOLD = 0.5
POSE_HUMAN_MATCH_IOU_THRESHOLD = 0.25
ACTION_CROP_EXPANSION_RATIO = 0.18
COCO_PERSON_KEYPOINT_NAMES = (
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
)
COCO_PERSON_SKELETON_EDGES = (
    ('nose', 'left_eye'),
    ('nose', 'right_eye'),
    ('left_eye', 'left_ear'),
    ('right_eye', 'right_ear'),
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle'),
)
COMBINED_CONFIDENCE_THRESHOLDS = {
    'human': 0.4,
    'object': 0.4,
}
UNCERTAINTY_THRESHOLDS = {
    'object': 0.55,
    'human': 0.55,
    'face': 0.75,
    'emotion': 0.6,
    'action': 0.6,
}
PERSON_OVERLAP_IOU_THRESHOLD = 0.55
FRAME_COUNTER = count(1)
GENERATED_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_JOBS = {}

MODEL_CONFIGS = {
    'face': {
        'label': 'Face Detection',
        'path': FACE_CASCADE_PATH,
        'runtime': 'haar',
    },
    'human': {
        'label': 'Human Detection',
        'path': MODELS_DIR / 'best.pt',
        'runtime': 'detector',
        'runtime_preferences': ('ultralytics', 'yolov5'),
        'fallbacks': ['last.pt', 'yolov5su.pt'],
        'class_filter': HUMAN_CLASS_LABELS,
        'output_label': 'human',
    },
    'object': {
        'label': 'Object Detection',
        'path': MODELS_DIR / 'Object Detection Model.pt',
        'runtime': 'detector',
        'runtime_preferences': ('yolov5', 'ultralytics'),
        'fallbacks': ['yolov5su.pt'],
    },
    'emotion': {
        'label': 'Emotion Detection',
        'path': EMOTION_MODEL_H5_PATH,
        'runtime': 'emotion',
    },
    'action': {
        'label': 'Action Recognition',
        'path': ACTION_MODEL_PATH,
        'runtime': 'action',
    },
}
LAZY_MODEL_CONFIGS = {
    'pose': {
        'label': 'Pose Detection (Keypoint R-CNN)',
        'path': POSE_MODEL_PATH,
        'runtime': 'keypoint_rcnn',
    },
}
ALL_MODEL_CONFIGS = {**MODEL_CONFIGS, **LAZY_MODEL_CONFIGS}
LAZY_MODELS = {}
LAZY_MODEL_LOAD_ERRORS = {}

DETECTION_MODES = {
    'combined': {
        'label': 'Phase 1 Scene Understanding',
        'models': ['face', 'emotion', 'human', 'action', 'object'],
    },
    'face_emotion': {
        'label': 'Face + Emotion Detection',
        'models': ['face', 'emotion'],
    },
    'face': {
        'label': 'Face Detection',
        'models': ['face'],
    },
    'human': {
        'label': 'Human Detection',
        'models': ['human'],
    },
    'action': {
        'label': 'Action Detection',
        'models': ['human', 'action'],
    },
    'pose': {
        'label': 'Pose Detection (Keypoint R-CNN)',
        'models': ['pose'],
    },
    'object': {
        'label': 'Object Detection',
        'models': ['object'],
    },
}


def load_face_model(model_path):
    cascade_candidates = (
        model_path,
        LEGACY_MODELS_DIR / model_path.name,
        LEGACY_FACE_CASCADE_PATH,
    )
    cascade_path = next((candidate for candidate in cascade_candidates if candidate.exists()), None)

    if cascade_path is None:
        cascade_path = Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml'

    if not cascade_path.exists():
        raise FileNotFoundError(f'Face cascade file not found: {cascade_path}')

    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError(f'Failed to load Haar cascade: {cascade_path}')

    eye_cascade_path = Path(cv2.data.haarcascades) / 'haarcascade_eye.xml'
    eye_cascade = cv2.CascadeClassifier(str(eye_cascade_path)) if eye_cascade_path.exists() else None
    if eye_cascade is not None and eye_cascade.empty():
        eye_cascade = None

    smile_cascade_path = Path(cv2.data.haarcascades) / 'haarcascade_smile.xml'
    smile_cascade = cv2.CascadeClassifier(str(smile_cascade_path)) if smile_cascade_path.exists() else None
    if smile_cascade is not None and smile_cascade.empty():
        smile_cascade = None

    return {
        'cascade': cascade,
        'cascade_path': cascade_path,
        'eye_cascade': eye_cascade,
        'smile_cascade': smile_cascade,
    }


def resolve_emotion_weights_path(preferred_path):
    candidates = [
        preferred_path,
        EMOTION_MODEL_H5_PATH,
        LEGACY_MODELS_DIR / 'model.h5',
        LEGACY_EMOTION_MODEL_H5_PATH,
        EMOTION_MODEL_SAFE_PATH,
        LEGACY_MODELS_DIR / 'model.safetensors',
        LEGACY_EMOTION_MODEL_SAFE_PATH,
        EMOTION_MODEL_BIN_PATH,
        LEGACY_MODELS_DIR / 'pytorch_model.bin',
        LEGACY_EMOTION_MODEL_BIN_PATH,
    ]
    seen = set()
    checked = []

    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        checked.append(candidate.name)
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Emotion model file not found. Checked: {', '.join(checked)}")


def resolve_emotion_labels():
    for labels_path in (EMOTION_LABELS_PATH, LEGACY_MODELS_DIR / 'emotion_labels.json', LEGACY_EMOTION_LABELS_PATH):
        if not labels_path.exists():
            continue

        with labels_path.open('r', encoding='utf-8') as fh:
            loaded = json.load(fh)

        if not isinstance(loaded, list) or not all(isinstance(label, str) for label in loaded):
            raise ValueError(f'{labels_path.name} must contain a JSON array of label strings.')

        return loaded

    return EMOTION_LABELS


def resolve_action_checkpoint_path(preferred_path):
    candidates = [
        preferred_path,
        MODELS_DIR / 'best_model.pt',
        LEGACY_MODELS_DIR / preferred_path.name,
        LEGACY_MODELS_DIR / 'best_model.pt',
        LEGACY_ACTION_MODEL_PATH,
        ACTION_OUTPUT_DIR / 'best_model.pt',
    ]
    seen = set()
    checked = []

    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        checked.append(candidate)
        if candidate.exists():
            return candidate

    checked_paths = ', '.join(str(path) for path in checked)
    raise FileNotFoundError(
        'Action model file not found. Checked: '
        + checked_paths
        + '. Train the image-based action model first and place best_model.pt in mod/ or Human Action Recognition/outputs/.'
    )

def resolve_action_class_names():
    candidates = [
        ACTION_LABEL_MAP_PATH,
        MODELS_DIR / 'label_map.json',
        LEGACY_MODELS_DIR / 'action_label_map.json',
        LEGACY_MODELS_DIR / 'label_map.json',
        LEGACY_ACTION_LABEL_MAP_PATH,
        ACTION_OUTPUT_DIR / 'label_map.json',
    ]

    for labels_path in candidates:
        if not labels_path.exists():
            continue

        with labels_path.open('r', encoding='utf-8') as fh:
            loaded = json.load(fh)

        if not isinstance(loaded, dict):
            raise ValueError(f'{labels_path.name} must contain a JSON object of label-to-index mappings.')

        ordered = sorted(loaded.items(), key=lambda item: int(item[1]))
        return [label for label, _ in ordered]

    raise FileNotFoundError(
        'Action label map not found. Provide label_map.json alongside the action checkpoint or ensure the checkpoint stores class_names.'
    )


def resolve_pose_checkpoint_path(preferred_path):
    candidates = [preferred_path, LEGACY_MODELS_DIR / preferred_path.name, LEGACY_POSE_MODEL_PATH]
    seen = set()
    checked = []

    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        checked.append(candidate)
        if candidate.exists():
            return candidate

    checked_paths = ', '.join(str(path) for path in checked)
    raise FileNotFoundError(
        'Pose model file not found. Checked: '
        + checked_paths
        + '. Download keypointrcnn_resnet50_fpn_coco-fc266e95.pth into mod/ to enable pose detection.'
    )


def resolve_detector_weights_paths(preferred_path, fallback_names=()):
    candidates = [preferred_path]
    if preferred_path.parent == MODELS_DIR:
        candidates.extend([LEGACY_MODELS_DIR / preferred_path.name, APP_ROOT / preferred_path.name])

    for name in fallback_names:
        fallback_path = Path(name)
        if fallback_path.is_absolute():
            candidates.append(fallback_path)
        else:
            candidates.extend([MODELS_DIR / name, LEGACY_MODELS_DIR / name, APP_ROOT / name])

    resolved = []
    seen = set()

    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        resolved.append(candidate)

    return resolved


class EmotionH5CNN(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.25)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(2048, 1024)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, num_labels)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        x = torch.relu(self.conv3(x))
        x = self.pool2(x)
        x = torch.relu(self.conv4(x))
        x = self.pool3(x)
        x = self.dropout2(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout3(x)
        return self.fc2(x)


def _load_h5_conv_layer_weights(h5_file, layer_name, module):
    layer_group = h5_file[layer_name][layer_name]
    kernel = np.array(layer_group['kernel:0'], dtype=np.float32)
    bias = np.array(layer_group['bias:0'], dtype=np.float32)
    module.weight.copy_(torch.from_numpy(np.transpose(kernel, (3, 2, 0, 1))))
    module.bias.copy_(torch.from_numpy(bias))


def _load_h5_dense_layer_weights(h5_file, layer_name, module):
    layer_group = h5_file[layer_name][layer_name]
    kernel = np.array(layer_group['kernel:0'], dtype=np.float32)
    bias = np.array(layer_group['bias:0'], dtype=np.float32)
    module.weight.copy_(torch.from_numpy(np.transpose(kernel, (1, 0))))
    module.bias.copy_(torch.from_numpy(bias))


def load_h5_emotion_model(model_path, labels):
    try:
        h5py = importlib.import_module('h5py')
    except ImportError as exc:
        raise ImportError(
            'Emotion detection with model.h5 requires the optional package `h5py`. '
            'Install the current requirements into the Python environment that runs app.py.'
        ) from exc

    model = EmotionH5CNN(num_labels=len(labels))
    with torch.no_grad():
        with h5py.File(model_path, 'r') as h5_file:
            _load_h5_conv_layer_weights(h5_file, 'conv2d', model.conv1)
            _load_h5_conv_layer_weights(h5_file, 'conv2d_1', model.conv2)
            _load_h5_conv_layer_weights(h5_file, 'conv2d_2', model.conv3)
            _load_h5_conv_layer_weights(h5_file, 'conv2d_3', model.conv4)
            _load_h5_dense_layer_weights(h5_file, 'dense', model.fc1)
            _load_h5_dense_layer_weights(h5_file, 'dense_1', model.fc2)

    model.eval()
    return {
        'model': model,
        'labels': labels,
        'weights_path': model_path,
        'runtime': 'keras_h5',
        'input_size': (48, 48),
    }


def load_transformers_emotion_model(model_path, labels):
    try:
        transformers = importlib.import_module('transformers')
        safetensors_torch = importlib.import_module('safetensors.torch')
    except ImportError as exc:
        raise ImportError(
            'Emotion detection requires the optional packages `transformers` and `safetensors`. '
            'Install the current requirements into the Python environment that runs app.py.'
        ) from exc

    ViTConfig = transformers.ViTConfig
    ViTForImageClassification = transformers.ViTForImageClassification
    ViTImageProcessor = transformers.ViTImageProcessor
    config = ViTConfig(
        image_size=224,
        patch_size=16,
        num_channels=3,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        qkv_bias=True,
        num_labels=len(labels),
    )
    model = ViTForImageClassification(config)

    if model_path.suffix == '.safetensors':
        state_dict = safetensors_torch.load_file(str(model_path))
    else:
        state_dict = torch.load(str(model_path), map_location='cpu')

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f'Emotion model state mismatch. Missing keys: {missing}. Unexpected keys: {unexpected}.'
        )

    processor = ViTImageProcessor(
        do_resize=True,
        size={'height': 224, 'width': 224},
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )
    model.eval()
    return {
        'model': model,
        'processor': processor,
        'labels': labels,
        'weights_path': model_path,
        'runtime': 'transformers',
    }


def load_emotion_model(model_path):
    weights_path = resolve_emotion_weights_path(model_path)
    labels = resolve_emotion_labels()
    if weights_path.suffix.lower() == '.h5':
        return load_h5_emotion_model(weights_path, labels)
    return load_transformers_emotion_model(weights_path, labels)


def build_action_classifier(num_classes):
    try:
        torchvision_models = importlib.import_module('torchvision.models')
        torchvision_transforms = importlib.import_module('torchvision.transforms')
    except ImportError as exc:
        raise ImportError(
            'Action recognition requires torchvision. Install the current requirements into the Python environment that runs app.py.'
        ) from exc

    model = torchvision_models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    transform = torchvision_transforms.Compose(
        [
            torchvision_transforms.Resize((224, 224)),
            torchvision_transforms.ToTensor(),
            torchvision_transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return model, transform


def load_action_model(model_path):
    for model_dir in (ACTION_HF_MODEL_DIR, LEGACY_MODELS_DIR / 'action_hf_model'):
        if model_dir.exists() and model_dir.is_dir():
            return load_hf_action_model(model_dir)

    checkpoint_path = resolve_action_checkpoint_path(model_path)
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
    checkpoint_payload = checkpoint if isinstance(checkpoint, dict) else {}
    class_names = checkpoint_payload.get('class_names') or resolve_action_class_names()
    image_size = int(checkpoint_payload.get('image_size', 224))
    state_dict = checkpoint_payload.get('model_state_dict', checkpoint)

    model, transform = build_action_classifier(len(class_names))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f'Action model state mismatch. Missing keys: {missing}. Unexpected keys: {unexpected}.'
        )

    if image_size != 224:
        torchvision_transforms = importlib.import_module('torchvision.transforms')
        transform = torchvision_transforms.Compose(
            [
                torchvision_transforms.Resize((image_size, image_size)),
                torchvision_transforms.ToTensor(),
                torchvision_transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    model.eval()
    return {
        'model': model,
        'transform': transform,
        'labels': class_names,
        'weights_path': checkpoint_path,
        'runtime': 'efficientnet_b0',
        'input_size': image_size,
    }


def load_pose_model(model_path):
    checkpoint_path = resolve_pose_checkpoint_path(model_path)
    torchvision_detection = importlib.import_module('torchvision.models.detection')
    keypointrcnn_resnet50_fpn = torchvision_detection.keypointrcnn_resnet50_fpn
    weights_enum = torchvision_detection.KeypointRCNN_ResNet50_FPN_Weights

    model = keypointrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get('model_state_dict'), dict):
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and isinstance(checkpoint.get('state_dict'), dict):
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f'Pose model state mismatch. Missing keys: {missing}. Unexpected keys: {unexpected}.'
        )

    model.eval()
    keypoint_names = tuple(weights_enum.DEFAULT.meta.get('keypoint_names', COCO_PERSON_KEYPOINT_NAMES))
    return {
        'model': model,
        'weights_path': checkpoint_path,
        'runtime': 'keypoint_rcnn',
        'keypoint_names': keypoint_names,
    }


def load_hf_action_model(model_dir):
    try:
        transformers = importlib.import_module('transformers')
    except ImportError as exc:
        raise ImportError(
            'Action recognition requires transformers. Install the current requirements into the Python environment that runs app.py.'
        ) from exc

    AutoImageProcessor = transformers.AutoImageProcessor
    AutoModelForImageClassification = transformers.AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(str(model_dir), local_files_only=True)
    id2label = getattr(model.config, 'id2label', {}) or {}
    class_names = [
        str(label)
        for _, label in sorted(
            ((int(idx), label) for idx, label in id2label.items()),
            key=lambda item: item[0],
        )
    ]

    if not class_names:
        raise RuntimeError(f'No action labels were found in {model_dir}.')

    model.eval()
    return {
        'model': model,
        'processor': processor,
        'labels': class_names,
        'weights_path': model_dir,
        'runtime': 'hf_image_classification',
        'input_size': getattr(model.config, 'image_size', 224),
    }
def load_detector_with_runtime(candidate, runtime):
    if str(YOLOV5_REPO) not in sys.path:
        sys.path.insert(0, str(YOLOV5_REPO))

    if runtime == 'ultralytics':
        from ultralytics import YOLO

        return YOLO(str(candidate))

    if runtime == 'yolov5':
        from models.common import AutoShape
        from models.experimental import attempt_load

        return AutoShape(attempt_load(str(candidate), device='cpu', fuse=False))

    raise ValueError(f'Unsupported detector runtime: {runtime}')


def load_detector_model(config):
    model_path = config['path']
    if not YOLOV5_REPO.exists():
        raise FileNotFoundError(f'YOLOv5 runtime not found: {YOLOV5_REPO}')

    attempted = []

    for candidate in resolve_detector_weights_paths(model_path, config.get('fallbacks', ())):
        if not candidate.exists():
            attempted.append(f'{candidate.name} (missing)')
            continue

        for runtime in config.get('runtime_preferences', ('ultralytics', 'yolov5')):
            try:
                model = load_detector_with_runtime(candidate, runtime)
                return {
                    'model': model,
                    'weights_path': candidate,
                    'runtime': runtime,
                    'class_filter': set(config.get('class_filter', ())),
                    'output_label': config.get('output_label'),
                }
            except Exception as exc:
                attempted.append(f'{candidate.name} ({runtime}: {exc})')

    raise RuntimeError(
        f"Unable to load {config['label'].lower()}. Tried: "
        + '; '.join(attempted)
    )


def load_models():
    models = {}
    load_errors = {}

    for model_type, config in MODEL_CONFIGS.items():
        try:
            if config['runtime'] == 'haar':
                models[model_type] = load_face_model(config['path'])
            elif config['runtime'] == 'emotion':
                models[model_type] = load_emotion_model(config['path'])
            elif config['runtime'] == 'action':
                models[model_type] = load_action_model(config['path'])
            elif config['runtime'] == 'detector':
                models[model_type] = load_detector_model(config)
            else:
                raise ValueError(f"Unsupported runtime for {model_type}: {config['runtime']}")
        except Exception as exc:
            load_errors[model_type] = str(exc)
            print(f"Failed to load {config['label']}: {exc}")

    if not models:
        raise RuntimeError('No detection models could be loaded.')

    return models, load_errors


MODELS, MODEL_LOAD_ERRORS = load_models()


def resolve_model_path(model_type, config):
    if model_type == 'pose':
        return resolve_pose_checkpoint_path(config['path'])
    return config['path']


def model_is_available(model_type):
    if model_type in MODELS or model_type in LAZY_MODELS:
        return True

    config = LAZY_MODEL_CONFIGS.get(model_type)
    if not config:
        return False

    try:
        resolve_model_path(model_type, config)
    except Exception:
        return False
    return True


def get_model_bundle(model_type):
    if model_type in MODELS:
        return MODELS[model_type]
    if model_type in LAZY_MODELS:
        return LAZY_MODELS[model_type]

    config = LAZY_MODEL_CONFIGS.get(model_type)
    if config is None:
        raise KeyError(f'Model {model_type!r} is not configured.')

    runtime = config['runtime']
    if runtime == 'keypoint_rcnn':
        model_bundle = load_pose_model(config['path'])
    else:
        raise ValueError(f'Unsupported lazy runtime for {model_type}: {runtime}')

    LAZY_MODELS[model_type] = model_bundle
    LAZY_MODEL_LOAD_ERRORS.pop(model_type, None)
    return model_bundle


def combined_model_load_errors():
    return {
        **MODEL_LOAD_ERRORS,
        **LAZY_MODEL_LOAD_ERRORS,
    }


def available_detection_modes():
    return [
        {
            'key': mode_key,
            'label': config['label'],
        }
        for mode_key, config in DETECTION_MODES.items()
        if all(model_is_available(model_name) for model_name in config['models'])
    ]


AVAILABLE_MODE_KEYS = [mode['key'] for mode in available_detection_modes()]
DEFAULT_MODE = 'combined' if 'combined' in AVAILABLE_MODE_KEYS else AVAILABLE_MODE_KEYS[0]


def available_models():
    models = []

    for model_type, config in ALL_MODEL_CONFIGS.items():
        if not model_is_available(model_type):
            continue

        model_bundle = MODELS.get(model_type) or LAZY_MODELS.get(model_type)
        filename = config['path'].name

        if isinstance(model_bundle, dict):
            if 'weights_path' in model_bundle:
                filename = model_bundle['weights_path'].name
            elif 'cascade_path' in model_bundle:
                filename = model_bundle['cascade_path'].name
        else:
            try:
                filename = resolve_model_path(model_type, config).name
            except Exception:
                pass

        models.append({
            'key': model_type,
            'label': config['label'],
            'filename': filename,
        })

    return models


def get_template_context(selected_mode=None, **extra):
    if selected_mode not in AVAILABLE_MODE_KEYS:
        selected_mode = DEFAULT_MODE

    context = {
        'available_detection_modes': available_detection_modes(),
        'available_models': available_models(),
        'selected_mode': selected_mode,
        'active_model_label': DETECTION_MODES[selected_mode]['label'],
        'model_load_errors': combined_model_load_errors(),
    }
    context.update(extra)
    return context


def render_image_detection_page(selected_mode=None, **extra):
    page_defaults = {
        'result': None,
        'result_video_url': None,
        'result_video_mime': 'video/mp4',
        'media_type': 'image',
        'media_label': 'Image',
        'video_job_id': None,
        'job_status': None,
        'job_message': None,
        'detections': [],
        'detection_count': 0,
        'image_width': None,
        'image_height': None,
        'phase1_output': None,
        'frames_processed': None,
        'video_total_frames': None,
        'video_fps': None,
        'error': None,
    }
    page_defaults.update(extra)
    return render_template('detect_result.html', **get_template_context(selected_mode=selected_mode, **page_defaults))


def resolve_mode(payload=None):
    requested_mode = None
    if payload is not None:
        requested_mode = payload.get('detection_mode') or payload.get('model_type')
    else:
        requested_mode = request.form.get('detection_mode') or request.form.get('model_type')

    mode = (requested_mode or DEFAULT_MODE).strip().lower()
    if mode not in AVAILABLE_MODE_KEYS:
        available = ', '.join(mode_config['label'] for mode_config in available_detection_modes())
        raise ValueError(f"Detection mode '{mode}' is not available. Available modes: {available}")
    return mode


def get_upload_media_type(file_storage):
    filename = file_storage.filename or ''
    suffix = Path(filename).suffix.lower()
    content_type = (file_storage.content_type or '').lower()

    if content_type.startswith('video/') or suffix in VIDEO_FILE_EXTENSIONS:
        return 'video'
    if content_type.startswith('image/') or suffix in IMAGE_FILE_EXTENSIONS:
        return 'image'
    return None


def annotate_display_detections(image_bgr, detections):
    for det in detections:
        x1 = int(det['x1'])
        y1 = int(det['y1'])
        x2 = int(det['x2'])
        y2 = int(det['y2'])
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        overlay_lines = det.get('overlay_lines') or [det.get('display_label') or f"{det['label']} {det['confidence']:.2f}"]
        for line_index, overlay_text in enumerate(overlay_lines):
            text_y = max(20, y1 - 10 - (len(overlay_lines) - line_index - 1) * 18)
            cv2.putText(
                image_bgr,
                str(overlay_text),
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        draw_pose_overlay(image_bgr, det.get('keypoints'))

    return image_bgr


def build_video_job_context(job_id, selected_mode, **extra):
    payload = {
        'video_job_id': job_id,
        'job_status': 'processing',
        'job_message': 'Video uploaded. Detection is running in the background.',
        'media_type': 'video',
        'media_label': 'Video',
        'result_video_url': None,
        'result_video_mime': 'video/mp4',
        'detections': [],
        'detection_count': 0,
        'image_width': None,
        'image_height': None,
        'phase1_output': None,
        'frames_processed': 0,
        'video_total_frames': None,
        'video_fps': None,
        'selected_mode': selected_mode,
    }
    payload.update(extra)
    return payload


def create_compatible_video_writer(output_stem, fps, frame_width, frame_height):
    candidates = [
        ('.mp4', 'video/mp4', 'avc1'),
        ('.mp4', 'video/mp4', 'H264'),
        ('.webm', 'video/webm', 'VP80'),
        ('.mp4', 'video/mp4', 'mp4v'),
    ]

    for extension, mime_type, codec in candidates:
        output_path = GENERATED_MEDIA_DIR / f'{output_stem}_annotated{extension}'
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (frame_width, frame_height),
        )
        if writer.isOpened():
            return writer, output_path, mime_type
        writer.release()

    raise RuntimeError('Unable to initialize a compatible video writer for the processed output.')


def prepare_video_inference_frame(frame_bgr):
    frame_height, frame_width = frame_bgr.shape[:2]
    max_edge = max(frame_width, frame_height)

    if max_edge <= VIDEO_MAX_PROCESSING_EDGE:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    scale = VIDEO_MAX_PROCESSING_EDGE / max_edge
    resized_width = max(1, int(round(frame_width * scale)))
    resized_height = max(1, int(round(frame_height * scale)))
    resized = cv2.resize(frame_bgr, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def process_video_job(job_id, input_path, output_stem, selected_mode):
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        VIDEO_JOBS[job_id].update({
            'job_status': 'failed',
            'job_message': 'Unable to open the uploaded video.',
        })
        input_path.unlink(missing_ok=True)
        return

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 24.0

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    inference_stride = max(1, int(np.ceil(total_frames / VIDEO_MAX_INFERENCE_FRAMES))) if total_frames else 1

    VIDEO_JOBS[job_id].update({
        'video_total_frames': total_frames or None,
        'video_fps': round(fps, 2),
        'image_width': frame_width or None,
        'image_height': frame_height or None,
        'job_message': 'Video uploaded. Detection is running frame by frame.',
    })

    writer = None
    output_path = None
    output_mime = 'video/mp4'
    frames_processed = 0
    detection_count = 0
    preview_output = None
    preview_detections = []
    last_detections = []

    try:
        while True:
            success, frame_bgr = capture.read()
            if not success:
                break

            if frame_width <= 0 or frame_height <= 0:
                frame_height, frame_width = frame_bgr.shape[:2]

            if writer is None:
                writer, output_path, output_mime = create_compatible_video_writer(
                    output_stem,
                    fps,
                    frame_width,
                    frame_height,
                )

            if frames_processed == 0 or frames_processed % inference_stride == 0:
                inference_rgb = prepare_video_inference_frame(frame_bgr)
                phase1_output = run_phase1_pipeline(
                    selected_mode,
                    inference_rgb,
                    display_width=frame_width,
                    display_height=frame_height,
                )
                last_detections = phase1_output['detections']
                detection_count += len(last_detections)

                if last_detections or preview_output is None:
                    preview_output = phase1_output
                    preview_detections = list(last_detections)

            annotate_display_detections(frame_bgr, last_detections)
            writer.write(frame_bgr)

            frames_processed += 1
            if frames_processed % 10 == 0 or frames_processed == 1:
                VIDEO_JOBS[job_id].update({
                    'frames_processed': frames_processed,
                    'job_message': f'Processing video frames: {frames_processed}/{total_frames or "?"}',
                    'image_width': frame_width,
                    'image_height': frame_height,
                })

        if frames_processed == 0:
            raise ValueError('The uploaded video does not contain readable frames.')

        if preview_output is None:
            preview_output = {
                'objects': [],
                'humans': [],
                'faces': [],
                'emotions': [],
            }

        VIDEO_JOBS[job_id].update({
            'job_status': 'completed',
            'job_message': 'Detected video is ready.',
            'result_video_url': f'/generated-media/{output_path.name}',
            'result_video_mime': output_mime,
            'detections': preview_detections,
            'detection_count': detection_count,
            'image_width': frame_width,
            'image_height': frame_height,
            'phase1_output': preview_output,
            'frames_processed': frames_processed,
            'video_total_frames': total_frames or frames_processed,
            'video_fps': round(fps, 2),
        })
    except Exception as exc:
        VIDEO_JOBS[job_id].update({
            'job_status': 'failed',
            'job_message': str(exc),
        })
        if output_path is not None:
            output_path.unlink(missing_ok=True)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        input_path.unlink(missing_ok=True)


def start_video_job(file_storage, selected_mode):
    input_suffix = Path(file_storage.filename or '').suffix.lower() or '.mp4'
    media_stem = f'media_{uuid4().hex}'
    input_path = GENERATED_MEDIA_DIR / f'{media_stem}{input_suffix}'
    job_id = uuid4().hex

    file_storage.save(str(input_path))
    VIDEO_JOBS[job_id] = build_video_job_context(job_id, selected_mode)

    worker = threading.Thread(
        target=process_video_job,
        args=(job_id, input_path, media_stem, selected_mode),
        daemon=True,
    )
    worker.start()
    return job_id


def get_label(names, cls_idx):
    if isinstance(names, dict):
        return str(names.get(cls_idx, cls_idx))
    if isinstance(names, (list, tuple)) and 0 <= cls_idx < len(names):
        return str(names[cls_idx])
    return str(cls_idx)


def as_int_point(x, y):
    return [int(round(x)), int(round(y))]


def estimate_face_landmarks(x, y, w, h):
    left_eye = as_int_point(x + w * 0.32, y + h * 0.38)
    right_eye = as_int_point(x + w * 0.68, y + h * 0.38)
    nose = as_int_point(x + w * 0.5, y + h * 0.58)
    mouth_left = as_int_point(x + w * 0.36, y + h * 0.76)
    mouth_right = as_int_point(x + w * 0.64, y + h * 0.76)
    return {
        'left_eye': left_eye,
        'right_eye': right_eye,
        'nose': nose,
        'mouth_left': mouth_left,
        'mouth_right': mouth_right,
        'source': 'estimated',
    }


def detect_face_landmarks(model, gray_image, x, y, w, h):
    landmarks = estimate_face_landmarks(x, y, w, h)
    roi_gray = gray_image[y:y + h, x:x + w]
    if roi_gray.size == 0:
        return landmarks

    eye_cascade = model.get('eye_cascade')
    if eye_cascade is not None:
        eye_region = roi_gray[: max(1, int(h * 0.6)), :]
        eyes = eye_cascade.detectMultiScale(
            eye_region,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(max(10, w // 10), max(10, h // 10)),
        )
        if len(eyes) >= 2:
            sorted_eyes = sorted(eyes, key=lambda eye_box: eye_box[0])[:2]
            left_eye_box, right_eye_box = sorted_eyes
            landmarks['left_eye'] = as_int_point(
                x + left_eye_box[0] + left_eye_box[2] / 2,
                y + left_eye_box[1] + left_eye_box[3] / 2,
            )
            landmarks['right_eye'] = as_int_point(
                x + right_eye_box[0] + right_eye_box[2] / 2,
                y + right_eye_box[1] + right_eye_box[3] / 2,
            )
            landmarks['source'] = 'cascade_heuristic'

    smile_cascade = model.get('smile_cascade')
    if smile_cascade is not None:
        smile_region_y = int(h * 0.45)
        smile_region = roi_gray[smile_region_y:, :]
        smiles = smile_cascade.detectMultiScale(
            smile_region,
            scaleFactor=1.7,
            minNeighbors=20,
            minSize=(max(20, w // 5), max(12, h // 10)),
        )
        if len(smiles) > 0:
            smile_box = max(smiles, key=lambda smile: smile[2] * smile[3])
            smile_x, smile_y, smile_w, smile_h = smile_box
            smile_y += smile_region_y
            mouth_center_y = y + smile_y + smile_h / 2
            landmarks['mouth_left'] = as_int_point(x + smile_x, mouth_center_y)
            landmarks['mouth_right'] = as_int_point(x + smile_x + smile_w, mouth_center_y)
            landmarks['source'] = 'cascade_heuristic'

    eye_mid_x = (landmarks['left_eye'][0] + landmarks['right_eye'][0]) / 2
    mouth_mid_y = (landmarks['mouth_left'][1] + landmarks['mouth_right'][1]) / 2
    landmarks['nose'] = as_int_point(eye_mid_x, (landmarks['left_eye'][1] + mouth_mid_y) / 2)
    return landmarks


def run_face_detection(model, image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = model['cascade'].detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    if len(faces) == 0:
        return []

    detections = []
    for x, y, w, h in sorted(faces, key=lambda box: (box[1], box[0])):
        detections.append({
            'x1': int(x),
            'y1': int(y),
            'x2': int(x + w),
            'y2': int(y + h),
            'confidence': 1.0,
            'label': 'face',
            'landmarks': detect_face_landmarks(model, gray, int(x), int(y), int(w), int(h)),
        })
    return detections


def normalize_keypoints(keypoints, keypoint_scores, image_width, image_height, keypoint_names):
    max_x = max(0, image_width - 1)
    max_y = max(0, image_height - 1)
    normalized = []
    visible_total = 0

    for idx, name in enumerate(keypoint_names):
        raw_point = keypoints[idx]
        raw_score = keypoint_scores[idx] if keypoint_scores is not None else 1.0
        is_visible = bool(float(raw_point[2]) > 0)
        if is_visible:
            visible_total += 1

        normalized.append({
            'name': str(name),
            'x': max(0, min(int(round(float(raw_point[0]))), max_x)),
            'y': max(0, min(int(round(float(raw_point[1]))), max_y)),
            'score': float(raw_score),
            'visible': is_visible,
        })

    return normalized, visible_total


def draw_pose_overlay(image_bgr, keypoints):
    if not keypoints:
        return

    visible_points = {
        point['name']: (int(point['x']), int(point['y']))
        for point in keypoints
        if point.get('visible')
    }

    for start_name, end_name in COCO_PERSON_SKELETON_EDGES:
        start_point = visible_points.get(start_name)
        end_point = visible_points.get(end_name)
        if start_point is None or end_point is None:
            continue
        cv2.line(image_bgr, start_point, end_point, (255, 191, 0), 2, cv2.LINE_AA)

    for point in keypoints:
        if not point.get('visible'):
            continue
        cv2.circle(
            image_bgr,
            (int(point['x']), int(point['y'])),
            3,
            (255, 140, 0),
            -1,
            lineType=cv2.LINE_AA,
        )


def run_pose_detection(model, image_np):
    detector = model['model']
    image_height, image_width = image_np.shape[:2]
    image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float().div(255.0)

    with torch.inference_mode():
        outputs = detector([image_tensor])[0]

    detections = []
    boxes = outputs.get('boxes')
    labels = outputs.get('labels')
    scores = outputs.get('scores')
    keypoints = outputs.get('keypoints')
    keypoints_scores = outputs.get('keypoints_scores')

    if boxes is None or labels is None or scores is None or keypoints is None:
        return detections

    resolved_keypoint_names = tuple(model.get('keypoint_names') or COCO_PERSON_KEYPOINT_NAMES)
    for box, label, score, person_keypoints, person_keypoint_scores in zip(
        boxes.cpu().numpy(),
        labels.cpu().numpy(),
        scores.cpu().numpy(),
        keypoints.cpu().numpy(),
        keypoints_scores.cpu().numpy() if keypoints_scores is not None else [None] * len(boxes),
    ):
        if int(label) != 1 or float(score) < POSE_DETECTION_THRESHOLD:
            continue

        x1, y1, x2, y2 = clamp_box(box[0], box[1], box[2], box[3], image_width, image_height)
        if x2 <= x1 or y2 <= y1:
            continue

        normalized_keypoints, visible_total = normalize_keypoints(
            person_keypoints,
            person_keypoint_scores,
            image_width,
            image_height,
            resolved_keypoint_names,
        )
        detections.append({
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'confidence': float(score),
            'label': 'person',
            'keypoints': normalized_keypoints,
            'visible_keypoint_count': visible_total,
            'keypoint_count': len(normalized_keypoints),
        })

    return detections


def run_detector_detection(model, image_np):
    detector = model['model']
    detections = []

    if model.get('runtime') == 'ultralytics':
        results = detector.predict(source=image_np, imgsz=640, device='cpu', verbose=False)
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()
        names = results[0].names
    else:
        results = detector(image_np, size=640)
        raw_boxes = results.xyxy[0].cpu().numpy()
        if raw_boxes.size == 0:
            return detections

        xyxy = raw_boxes[:, :4]
        confidences = raw_boxes[:, 4]
        class_ids = raw_boxes[:, 5]
        names = detector.names

    class_filter = model.get('class_filter') or None
    output_label = model.get('output_label')

    for box, conf, cls in zip(xyxy, confidences, class_ids):
        x1, y1, x2, y2 = box
        cls_idx = int(cls)
        label = get_label(names, cls_idx).strip()
        if class_filter and label.lower() not in class_filter:
            continue

        detections.append({
            'x1': int(x1),
            'y1': int(y1),
            'x2': int(x2),
            'y2': int(y2),
            'confidence': float(conf),
            'label': output_label or label,
        })

    return detections


def format_entity_label(label):
    return str(label or '').replace('_', ' ').replace('-', ' ').strip().title() or 'Detection'


def format_overlay_text(label, confidence):
    return f'{format_entity_label(label)} ({float(confidence):.2f})'


def build_overlay_label(det):
    if det.get('display_label'):
        return det['display_label']

    overlay_lines = det.get('overlay_lines') or []
    if overlay_lines:
        return ' | '.join(overlay_lines)

    return format_overlay_text(det.get('label', 'detection'), det.get('confidence', 0.0))


def next_frame_metadata():
    return {
        'frame_id': next(FRAME_COUNTER),
        'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
    }


def safe_positive_int(value, fallback):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def build_spatial_context(image_np, display_width=None, display_height=None):
    model_height, model_width = image_np.shape[:2]
    resolved_display_width = safe_positive_int(display_width, model_width)
    resolved_display_height = safe_positive_int(display_height, model_height)
    return {
        'model_width': model_width,
        'model_height': model_height,
        'display_width': resolved_display_width,
        'display_height': resolved_display_height,
        'scale_x': resolved_display_width / model_width if model_width else 1.0,
        'scale_y': resolved_display_height / model_height if model_height else 1.0,
    }


def scale_box_coordinates(det, spatial_context=None):
    if not spatial_context:
        return {
            'x1': int(round(det['x1'])),
            'y1': int(round(det['y1'])),
            'x2': int(round(det['x2'])),
            'y2': int(round(det['y2'])),
        }

    scale_x = spatial_context['scale_x']
    scale_y = spatial_context['scale_y']
    return {
        'x1': int(round(det['x1'] * scale_x)),
        'y1': int(round(det['y1'] * scale_y)),
        'x2': int(round(det['x2'] * scale_x)),
        'y2': int(round(det['y2'] * scale_y)),
    }


def scale_keypoints(keypoints, spatial_context=None):
    if not keypoints:
        return []

    if not spatial_context:
        return [
            {
                **point,
                'x': int(round(point['x'])),
                'y': int(round(point['y'])),
            }
            for point in keypoints
        ]

    scale_x = spatial_context['scale_x']
    scale_y = spatial_context['scale_y']
    return [
        {
            **point,
            'x': int(round(point['x'] * scale_x)),
            'y': int(round(point['y'] * scale_y)),
        }
        for point in keypoints
    ]


def as_bbox(det, spatial_context=None):
    scaled = scale_box_coordinates(det, spatial_context)
    return {
        'x': scaled['x1'],
        'y': scaled['y1'],
        'width': max(0, scaled['x2'] - scaled['x1']),
        'height': max(0, scaled['y2'] - scaled['y1']),
    }


def sort_detections(detections):
    return sorted(detections, key=lambda det: (int(det['y1']), int(det['x1']), -box_area(det)))


def assign_entity_ids(detections, prefix):
    for idx, det in enumerate(detections, start=1):
        det['id'] = f'{prefix}_{idx}'
    return detections


def box_contains(inner_det, outer_det):
    return (
        inner_det['x1'] >= outer_det['x1']
        and inner_det['y1'] >= outer_det['y1']
        and inner_det['x2'] <= outer_det['x2']
        and inner_det['y2'] <= outer_det['y2']
    )


def match_face_to_human(face_det, human_detections):
    if not human_detections:
        return None

    containing_humans = [
        human_det for human_det in human_detections
        if box_contains(face_det, human_det)
    ]
    if not containing_humans:
        return None

    return max(containing_humans, key=lambda human_det: compute_iou(face_det, human_det))


def map_faces_to_humans(face_detections, human_detections):
    for face_det in face_detections:
        parent_human = match_face_to_human(face_det, human_detections)
        face_det['parent_human_id'] = parent_human['id'] if parent_human else None
    return face_detections


def is_uncertain(confidence, category):
    threshold = UNCERTAINTY_THRESHOLDS.get(category)
    return threshold is not None and confidence is not None and confidence < threshold


def uncertainty_reason(confidence, category):
    threshold = UNCERTAINTY_THRESHOLDS.get(category)
    if threshold is None or confidence is None or confidence >= threshold:
        return None
    return f'confidence_below_{threshold:.2f}'


def box_area(det):
    width = max(0, det['x2'] - det['x1'])
    height = max(0, det['y2'] - det['y1'])
    return width * height


def compute_iou(det_a, det_b):
    x1 = max(det_a['x1'], det_b['x1'])
    y1 = max(det_a['y1'], det_b['y1'])
    x2 = min(det_a['x2'], det_b['x2'])
    y2 = min(det_a['y2'], det_b['y2'])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)
    intersection = intersection_width * intersection_height
    if intersection == 0:
        return 0.0

    union = box_area(det_a) + box_area(det_b) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def merge_human_detections(human_detections, supplemental_detections):
    merged = [dict(det) for det in human_detections]

    for supplemental_det in supplemental_detections:
        supplemental_copy = dict(supplemental_det)
        best_match = None
        best_iou = 0.0

        for human_det in merged:
            iou = compute_iou(human_det, supplemental_copy)
            if iou > best_iou:
                best_iou = iou
                best_match = human_det

        if best_match is not None and best_iou >= POSE_HUMAN_MATCH_IOU_THRESHOLD:
            supplemental_visible = int(supplemental_copy.get('visible_keypoint_count') or 0)
            existing_visible = int(best_match.get('visible_keypoint_count') or 0)
            if supplemental_copy.get('keypoints') and supplemental_visible >= existing_visible:
                best_match['keypoints'] = supplemental_copy['keypoints']
                best_match['visible_keypoint_count'] = supplemental_copy.get('visible_keypoint_count')
                best_match['keypoint_count'] = supplemental_copy.get('keypoint_count')
            supplemental_source = str(supplemental_copy.get('source_model') or '').strip()
            if supplemental_source:
                existing_source = str(best_match.get('source_model') or '').strip()
                existing_parts = [part for part in existing_source.split('+') if part]
                if supplemental_source not in existing_parts:
                    best_match['source_model'] = '+'.join(existing_parts + [supplemental_source]) if existing_parts else supplemental_source
            continue

        merged.append(supplemental_copy)

    return sort_detections(merged)

def filter_combined_detections(human_detections, object_detections):
    filtered_humans = [
        det for det in human_detections
        if det['confidence'] >= COMBINED_CONFIDENCE_THRESHOLDS['human']
    ]
    filtered_objects = []

    for det in object_detections:
        if det['confidence'] < COMBINED_CONFIDENCE_THRESHOLDS['object']:
            continue

        if det['label'].strip().lower() == 'person':
            if any(compute_iou(det, human_det) >= PERSON_OVERLAP_IOU_THRESHOLD for human_det in filtered_humans):
                continue

        filtered_objects.append(det)

    return filtered_humans, filtered_objects


def finalize_detections(detections):
    for det in detections:
        det['overlay_label'] = build_overlay_label(det)
    return detections


def clamp_box(x1, y1, x2, y2, width, height):
    return (
        max(0, min(int(x1), width)),
        max(0, min(int(y1), height)),
        max(0, min(int(x2), width)),
        max(0, min(int(y2), height)),
    )


def expand_box(x1, y1, x2, y2, width, height, ratio):
    box_width = max(0, x2 - x1)
    box_height = max(0, y2 - y1)
    pad_x = box_width * ratio
    pad_y = box_height * ratio
    return clamp_box(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, width, height)

def preprocess_h5_face_crops(face_crops, input_size):
    width, height = input_size
    arrays = []

    for crop in face_crops:
        grayscale = crop.convert('L').resize((width, height), Image.Resampling.BILINEAR)
        arrays.append(np.asarray(grayscale, dtype=np.float32) / 255.0)

    batch = np.stack(arrays, axis=0)[:, None, :, :]
    return torch.from_numpy(batch)


def annotate_face_emotions(image_np, face_detections):
    if 'emotion' not in MODELS or not face_detections:
        return face_detections

    image = Image.fromarray(image_np)
    image_width, image_height = image.size
    face_crops = []
    valid_indices = []

    for idx, det in enumerate(face_detections):
        x1, y1, x2, y2 = clamp_box(det['x1'], det['y1'], det['x2'], det['y2'], image_width, image_height)
        if x2 <= x1 or y2 <= y1:
            continue
        face_crops.append(image.crop((x1, y1, x2, y2)))
        valid_indices.append(idx)

    if not face_crops:
        return face_detections

    emotion_bundle = MODELS['emotion']
    model = emotion_bundle['model']
    labels = emotion_bundle['labels']

    if emotion_bundle.get('runtime') == 'keras_h5':
        inputs = preprocess_h5_face_crops(face_crops, emotion_bundle['input_size'])
        with torch.inference_mode():
            logits = model(inputs)
            probabilities = torch.softmax(logits, dim=-1)
    else:
        processor = emotion_bundle['processor']
        inputs = processor(images=face_crops, return_tensors='pt')
        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)

    for probs_idx, det_idx in enumerate(valid_indices):
        probs = probabilities[probs_idx]
        top_idx = int(probs.argmax().item())
        emotion_confidence = float(probs[top_idx].item())
        emotion_label = labels[top_idx] if top_idx < len(labels) else f'emotion_{top_idx}'
        emotion_scores = {
            labels[label_idx] if label_idx < len(labels) else f'emotion_{label_idx}': float(probs[label_idx].item())
            for label_idx in range(len(probs))
        }
        face_detections[det_idx]['emotion_label'] = emotion_label
        face_detections[det_idx]['emotion_confidence'] = emotion_confidence
        face_detections[det_idx]['emotion_scores'] = emotion_scores
        face_detections[det_idx]['display_label'] = (
            f"{face_detections[det_idx]['label']} | {emotion_label} {emotion_confidence:.2f}"
        )

    return face_detections


def detect_objects(frame, mode='combined'):
    detections = []

    if mode in {'combined', 'object'} and 'object' in MODELS:
        detections = run_detector_detection(MODELS['object'], frame)
        for det in detections:
            det['source_model'] = 'object'
    elif mode == 'pose' and model_is_available('pose'):
        detections = run_pose_detection(get_model_bundle('pose'), frame)
        for det in detections:
            det['label'] = 'person'
            det['source_model'] = 'pose'
    elif mode in {'human', 'action'} and 'human' in MODELS:
        detections = run_detector_detection(MODELS['human'], frame)
        for det in detections:
            det['label'] = 'person'
            det['source_model'] = 'human'
        if mode == 'action' and model_is_available('pose'):
            pose_detections = run_pose_detection(get_model_bundle('pose'), frame)
            for det in pose_detections:
                det['label'] = 'person'
                det['source_model'] = 'pose'
            detections = merge_human_detections(detections, pose_detections)
    elif mode == 'action' and model_is_available('pose'):
        detections = run_pose_detection(get_model_bundle('pose'), frame)
        for det in detections:
            det['label'] = 'person'
            det['source_model'] = 'pose'

    return sort_detections(detections)


def run_combined_detections(frame):
    object_detections = []
    if 'object' in MODELS:
        object_detections = run_detector_detection(MODELS['object'], frame)
        for det in object_detections:
            det['source_model'] = 'object'

    human_detections = []
    if 'human' in MODELS:
        human_detections = run_detector_detection(MODELS['human'], frame)
        for det in human_detections:
            det['label'] = 'person'
            det['source_model'] = 'human'

    object_person_detections = extract_humans(object_detections)
    if object_person_detections:
        for det in object_person_detections:
            det['source_model'] = det.get('source_model', 'object')
        human_detections = merge_human_detections(human_detections, object_person_detections)

    if model_is_available('pose'):
        pose_detections = run_pose_detection(get_model_bundle('pose'), frame)
        for det in pose_detections:
            det['label'] = 'person'
            det['source_model'] = 'pose'
        human_detections = merge_human_detections(human_detections, pose_detections)

    human_detections, object_detections = filter_combined_detections(human_detections, object_detections)
    return sort_detections(object_detections), sort_detections(human_detections)


def extract_humans(objects):
    humans = []

    for obj_det in objects:
        if str(obj_det.get('label', '')).strip().lower() not in HUMAN_CLASS_LABELS:
            continue

        human_det = dict(obj_det)
        human_det['label'] = 'person'
        humans.append(human_det)

    return sort_detections(humans)


def detect_faces(frame):
    if 'face' not in MODELS:
        return []

    detections = run_face_detection(MODELS['face'], frame)
    for det in detections:
        det['source_model'] = 'face'
    return sort_detections(detections)


def detect_emotions(frame, face_detections):
    annotated_faces = annotate_face_emotions(frame, face_detections)
    emotions = []

    for face_det in annotated_faces:
        if not face_det.get('emotion_label'):
            continue

        emotions.append({
            'face_id': face_det['id'],
            'parent_human_id': face_det.get('parent_human_id'),
            'label': face_det['emotion_label'],
            'confidence': face_det.get('emotion_confidence'),
        })

    return annotated_faces, emotions


def build_human_summaries(human_detections, face_detections, emotions):
    emotions_by_face_id = {emotion['face_id']: emotion for emotion in emotions}

    for human_det in human_detections:
        linked_faces = [
            face_det for face_det in face_detections
            if face_det.get('parent_human_id') == human_det['id']
        ]
        primary_face = max(
            linked_faces,
            key=lambda face_det: face_det.get('emotion_confidence', face_det.get('confidence', 0.0)),
            default=None,
        )
        linked_emotion = emotions_by_face_id.get(primary_face['id']) if primary_face else None
        human_det['summary'] = {
            'face': 'Yes' if primary_face else 'No',
            'emotion': format_entity_label(linked_emotion['label']) if linked_emotion else 'N/A',
            'action': format_entity_label(human_det.get('action_label')) if human_det.get('action_label') else 'N/A',
        }

    return human_detections


def annotate_human_actions(image_np, human_detections):
    if 'action' not in MODELS or not human_detections:
        return human_detections

    image = Image.fromarray(image_np)
    image_width, image_height = image.size
    human_crops = []
    valid_indices = []

    for idx, det in enumerate(human_detections):
        x1, y1, x2, y2 = expand_box(
            det['x1'],
            det['y1'],
            det['x2'],
            det['y2'],
            image_width,
            image_height,
            ACTION_CROP_EXPANSION_RATIO,
        )
        if x2 <= x1 or y2 <= y1:
            continue
        human_crops.append(image.crop((x1, y1, x2, y2)))
        valid_indices.append(idx)

    if not human_crops:
        return human_detections

    action_bundle = MODELS['action']
    model = action_bundle['model']
    labels = action_bundle['labels']

    if action_bundle.get('runtime') == 'hf_image_classification':
        processor = action_bundle['processor']
        inputs = processor(images=human_crops, return_tensors='pt')
        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)
    else:
        batch = torch.stack([action_bundle['transform'](crop) for crop in human_crops], dim=0)
        with torch.inference_mode():
            logits = model(batch)
            probabilities = torch.softmax(logits, dim=-1)

    for probs_idx, det_idx in enumerate(valid_indices):
        probs = probabilities[probs_idx]
        top_idx = int(probs.argmax().item())
        action_confidence = float(probs[top_idx].item())
        action_label = labels[top_idx] if top_idx < len(labels) else f'action_{top_idx}'
        action_scores = {
            labels[label_idx] if label_idx < len(labels) else f'action_{label_idx}': float(probs[label_idx].item())
            for label_idx in range(len(probs))
        }
        human_detections[det_idx]['action_label'] = action_label
        human_detections[det_idx]['action_confidence'] = action_confidence
        human_detections[det_idx]['action_scores'] = action_scores
        human_detections[det_idx]['display_label'] = (
            f"{human_detections[det_idx]['label']} | {action_label} {action_confidence:.2f}"
        )

    return human_detections


def build_frame_notes(object_detections, human_detections, face_detections, emotions, actions):
    notes = []

    if not object_detections and not human_detections and not face_detections:
        notes.append('No objects, humans, or faces detected in the frame.')

    if not face_detections:
        notes.append('No face detected; emotion analysis skipped.')
    elif not emotions:
        if 'emotion' not in MODELS:
            notes.append('Emotion model unavailable; emotion outputs omitted.')
        else:
            notes.append('Faces detected, but no emotion result was produced.')

    if human_detections and 'action' not in MODELS:
        notes.append('Action model unavailable; action outputs omitted.')
    elif human_detections and not actions:
        notes.append('No action detected for the current human detections.')

    low_confidence_total = (
        sum(1 for det in object_detections if is_uncertain(det.get('confidence'), 'object'))
        + sum(1 for det in human_detections if is_uncertain(det.get('confidence'), 'human'))
        + sum(1 for det in face_detections if is_uncertain(det.get('confidence'), 'face'))
    )
    if low_confidence_total:
        notes.append(f'{low_confidence_total} detection(s) flagged as uncertain due to low confidence.')

    return notes


def build_output(frame_meta, object_detections, human_detections, face_detections, emotions, spatial_context):
    actions = [
        {
            'human_id': det['id'],
            'label': det['action_label'],
            'confidence': det.get('action_confidence'),
        }
        for det in human_detections
        if det.get('action_label')
    ]

    return {
        **frame_meta,
        'objects': [
            {
                'id': det['id'],
                'label': det['label'],
                'confidence': det['confidence'],
                'bbox': as_bbox(det, spatial_context),
            }
            for det in object_detections
        ],
        'humans': [
            {
                'id': det['id'],
                'confidence': det['confidence'],
                'bbox': as_bbox(det, spatial_context),
                'keypoints': scale_keypoints(det.get('keypoints'), spatial_context),
                'visible_keypoint_count': det.get('visible_keypoint_count'),
                'keypoint_count': det.get('keypoint_count'),
                'summary': det.get('summary', {'face': 'No', 'emotion': 'N/A'}),
            }
            for det in human_detections
        ],
        'faces': [
            {
                'id': det['id'],
                'parent_human_id': det.get('parent_human_id'),
                'confidence': det['confidence'],
                'bbox': as_bbox(det, spatial_context),
            }
            for det in face_detections
        ],
        'emotions': emotions,
        'actions': actions,
    }


def build_display_detections(object_detections, human_detections, face_detections, spatial_context):
    display_detections = []

    for det in object_detections:
        if str(det.get('label', '')).strip().lower() in HUMAN_CLASS_LABELS:
            continue

        scaled = scale_box_coordinates(det, spatial_context)
        display_detections.append({
            **scaled,
            'id': det['id'],
            'label': det['label'],
            'confidence': det['confidence'],
            'source_model': det.get('source_model', 'object'),
            'overlay_lines': [format_overlay_text(det['label'], det['confidence'])],
            'display_label': format_overlay_text(det['label'], det['confidence']),
        })

    for det in human_detections:
        scaled = scale_box_coordinates(det, spatial_context)
        overlay_lines = [format_overlay_text('person', det['confidence'])]
        if det.get('action_label') and det.get('action_confidence') is not None:
            overlay_lines.append(format_overlay_text(det['action_label'], det['action_confidence']))
        if det.get('keypoints'):
            visible_total = det.get('visible_keypoint_count', 0)
            keypoint_total = det.get('keypoint_count', len(det.get('keypoints') or ()))
            overlay_lines.append(f'Pose {visible_total}/{keypoint_total}')

        display_detections.append({
            **scaled,
            'id': det['id'],
            'label': 'person',
            'confidence': det['confidence'],
            'source_model': det.get('source_model', 'human'),
            'keypoints': scale_keypoints(det.get('keypoints'), spatial_context),
            'visible_keypoint_count': det.get('visible_keypoint_count'),
            'keypoint_count': det.get('keypoint_count'),
            'overlay_lines': overlay_lines,
            'display_label': ' | '.join(overlay_lines),
        })

    for det in face_detections:
        scaled = scale_box_coordinates(det, spatial_context)
        overlay_lines = [format_overlay_text('face', det['confidence'])]
        if det.get('emotion_label') and det.get('emotion_confidence') is not None:
            overlay_lines.append(format_overlay_text(det['emotion_label'], det['emotion_confidence']))

        display_detections.append({
            **scaled,
            'id': det['id'],
            'label': 'face',
            'confidence': det['confidence'],
            'parent_human_id': det.get('parent_human_id'),
            'source_model': det.get('source_model', 'face'),
            'overlay_lines': overlay_lines,
            'display_label': ' | '.join(overlay_lines),
        })

    return finalize_detections(display_detections)


def run_detection(model_type, image_np):
    if model_type == 'face':
        return finalize_detections(run_face_detection(MODELS[model_type], image_np))
    if model_type == 'pose':
        return finalize_detections(run_pose_detection(get_model_bundle(model_type), image_np))
    if model_type in {'human', 'object'}:
        return finalize_detections(run_detector_detection(MODELS[model_type], image_np))
    raise ValueError(f'Unsupported model type: {model_type}')


def run_phase1_pipeline(mode, image_np, display_width=None, display_height=None):
    frame_meta = next_frame_metadata()
    spatial_context = build_spatial_context(image_np, display_width=display_width, display_height=display_height)

    if mode == 'combined':
        object_detections, human_detections = run_combined_detections(image_np)
    else:
        object_detections = detect_objects(image_np, mode=mode)
        human_detections = extract_humans(object_detections)

    assign_entity_ids(object_detections, 'obj')
    assign_entity_ids(human_detections, 'human')

    face_detections = detect_faces(image_np) if 'face' in DETECTION_MODES[mode]['models'] else []
    assign_entity_ids(face_detections, 'face')
    face_detections = map_faces_to_humans(face_detections, human_detections)

    emotions = []
    if 'emotion' in DETECTION_MODES[mode]['models']:
        face_detections, emotions = detect_emotions(image_np, face_detections)

    if (mode == 'combined' or 'action' in DETECTION_MODES[mode]['models']) and 'action' in MODELS:
        human_detections = annotate_human_actions(image_np, human_detections)

    human_detections = build_human_summaries(human_detections, face_detections, emotions)
    phase1_output = build_output(
        frame_meta,
        object_detections,
        human_detections,
        face_detections,
        emotions,
        spatial_context,
    )
    phase1_output['notes'] = build_frame_notes(
        object_detections,
        human_detections,
        face_detections,
        emotions,
        phase1_output.get('actions', []),
    )
    phase1_output['detections'] = build_display_detections(
        object_detections,
        human_detections,
        face_detections,
        spatial_context,
    )
    return phase1_output


def run_detection_mode(mode, image_np, display_width=None, display_height=None):
    return run_phase1_pipeline(
        mode,
        image_np,
        display_width=display_width,
        display_height=display_height,
    )['detections']


@app.route('/generated-media/<path:filename>')
def generated_media(filename):
    return send_from_directory(GENERATED_MEDIA_DIR, filename)


@app.route('/video-job-status/<job_id>')
def video_job_status(job_id):
    job = VIDEO_JOBS.get(job_id)
    if not job:
        return jsonify({'error': 'Video job not found.'}), 404
    return jsonify(job)


@app.route('/')
def index():
    return render_template('index.html', **get_template_context(selected_mode=DEFAULT_MODE))

@app.route('/image-detection')
def image_detection():
    selected_mode = request.args.get('mode')
    job_id = request.args.get('job_id')
    if job_id and job_id in VIDEO_JOBS:
        job_payload = dict(VIDEO_JOBS[job_id])
        job_payload.pop('selected_mode', None)
        return render_image_detection_page(selected_mode=selected_mode, **job_payload)
    return render_image_detection_page(selected_mode=selected_mode)

@app.route('/detect', methods=['POST'])
def detect():
    image_np = None
    selected_mode = DEFAULT_MODE
    display_width = None
    display_height = None
    upload_media_type = 'image'
    try:
        data = request.get_json(silent=True) if request.is_json else None
        selected_mode = resolve_mode(data)

        if request.is_json:
            data = data or {}
            display_width = data.get('display_width')
            display_height = data.get('display_height')
            if 'image' in data:
                # Decode base64 image
                img_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64,
                image = Image.open(io.BytesIO(base64.b64decode(img_data))).convert('RGB')
                image_np = np.array(image)
        else:
            if 'file' not in request.files:
                return render_image_detection_page(selected_mode=selected_mode, error='No file uploaded')
            file = request.files['file']
            if file.filename == '':
                return render_image_detection_page(selected_mode=selected_mode, error='No file selected')
            upload_media_type = get_upload_media_type(file)
            if upload_media_type == 'video':
                job_id = start_video_job(file, selected_mode)
                return redirect(url_for('image_detection', mode=selected_mode, job_id=job_id))
            if upload_media_type != 'image':
                return render_image_detection_page(
                    selected_mode=selected_mode,
                    error='Unsupported file type. Upload an image or a video.',
                )

            image = Image.open(io.BytesIO(file.read())).convert('RGB')
            image_np = np.array(image)
        
        if image_np is None:
            return jsonify({'error': 'No image provided'}), 400

        phase1_output = run_phase1_pipeline(
            selected_mode,
            image_np,
            display_width=display_width,
            display_height=display_height,
        )
        detections = phase1_output['detections']
        active_model_label = DETECTION_MODES[selected_mode]['label']

        if request.is_json:
            return jsonify({
                **phase1_output,
                'detections': detections,
                'model_type': selected_mode,
                'detection_mode': selected_mode,
                'model_label': active_model_label,
            })
        else:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            annotate_display_detections(image_bgr, detections)
            image_np = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            image_height, image_width = image_np.shape[:2]
            result_image = Image.fromarray(image_np)
            buf = io.BytesIO()
            result_image.save(buf, format='JPEG')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            return render_image_detection_page(
                selected_mode=selected_mode,
                result=img_base64,
                media_type='image',
                media_label='Image',
                detections=detections,
                detection_count=len(detections),
                image_width=image_width,
                image_height=image_height,
                phase1_output=phase1_output,
            )
    except Exception as e:
        print(f"Error in /detect: {e}")
        if request.is_json:
            return jsonify({'error': str(e)}), 500
        else:
            return render_image_detection_page(selected_mode=selected_mode, error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
