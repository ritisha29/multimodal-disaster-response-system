import os
import csv
import warnings
from typing import Callable, Optional, Tuple, Any
import torch
from torch.utils.data import Dataset
from PIL import Image

try:
    from kaggle_paths import get_data_dir
except ImportError:
    try:
        from src.kaggle_paths import get_data_dir
    except ImportError:
        get_data_dir = lambda: "data"


class CrisisMMDDataset(Dataset):
    """
    PyTorch Dataset for CrisisMMD multimodal disaster classification.
    
    Task: Informative classification ('informative' vs 'not_informative')
    Splits: 'train', 'dev', 'test'
    """
    LABEL_MAP = {
        'not_informative': 0,
        'informative': 1
    }

    def __init__(
        self,
        split: str = 'train',
        task: str = 'informative',
        data_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        verify_images: bool = False
    ):
        """
        Args:
            split (str): One of 'train', 'dev', or 'test'.
            task (str): Task name, default 'informative'.
            data_dir (str, optional): Directory containing 'CrisisMMD_v2.0' and 'crisismmd_datasplit_agreed_label'.
                                      If None, auto-detected via kaggle_paths.
            transform (Callable, optional): Optional torchvision transform to apply to images.
            verify_images (bool): If True, attempt to open every image on initialization to filter out corrupt ones.
        """
        super().__init__()
        self.split = split.lower()
        self.task = task.lower()

        # Configurable dataset path (auto-resolves for Kaggle or local if None or not found)
        if data_dir is None:
            self.data_dir = get_data_dir()
        elif not os.path.exists(data_dir):
            resolved = get_data_dir(data_dir)
            self.data_dir = resolved if os.path.exists(resolved) else data_dir
        else:
            self.data_dir = data_dir

        self.transform = transform

        # Resolve paths
        self.tsv_path = self._resolve_tsv_path()
        self.images_root = self._resolve_images_root()

        # Load samples
        self.samples = []
        self._load_dataset(verify_images=verify_images)

    def _resolve_tsv_path(self) -> str:
        """Find the exact TSV file path for the requested split and task."""
        filename = f"task_{self.task}_text_img_agreed_lab_{self.split}.tsv"
        
        # Candidate locations
        candidates = [
            os.path.join(self.data_dir, "crisismmd_datasplit_agreed_label", "crisismmd_datasplit_agreed_label", filename),
            os.path.join(self.data_dir, "crisismmd_datasplit_agreed_label", filename),
            os.path.join(self.data_dir, filename),
        ]
        
        for cand in candidates:
            if os.path.exists(cand):
                return cand
                
        # Recursive fallback search in data_dir
        for root, _, files in os.walk(self.data_dir):
            if filename in files:
                return os.path.join(root, filename)

        raise FileNotFoundError(
            f"Could not locate TSV file '{filename}' inside data directory '{self.data_dir}'."
        )

    def _resolve_images_root(self) -> str:
        """Locate the base directory containing 'data_image'."""
        candidates = [
            os.path.join(self.data_dir, "CrisisMMD_v2.0", "CrisisMMD_v2.0"),
            os.path.join(self.data_dir, "CrisisMMD_v2.0"),
            self.data_dir
        ]
        for cand in candidates:
            if os.path.exists(os.path.join(cand, "data_image")):
                return cand

        # Fallback search for data_image directory
        for root, dirs, _ in os.walk(self.data_dir):
            if "data_image" in dirs:
                return root

        raise FileNotFoundError(
            f"Could not locate 'data_image' folder inside data directory '{self.data_dir}'."
        )

    def _load_dataset(self, verify_images: bool = False):
        """Parse TSV and validate image paths."""
        with open(self.tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row_idx, row in enumerate(reader):
                text = row.get('tweet_text', '').strip()
                rel_img_path = row.get('image', '').strip()
                raw_label = row.get('label', '').strip()

                if raw_label not in self.LABEL_MAP:
                    warnings.warn(f"Row {row_idx}: Unknown label '{raw_label}'. Skipping sample.")
                    continue
                label = self.LABEL_MAP[raw_label]

                # Normalize image path for cross-platform and Linux/Kaggle compatibility
                clean_rel_img = rel_img_path.lstrip(r"\/").replace("\\", "/")
                full_img_path = os.path.normpath(os.path.join(self.images_root, clean_rel_img))

                if not os.path.exists(full_img_path):
                    warnings.warn(f"Row {row_idx}: Image path not found '{full_img_path}'. Skipping sample.")
                    continue

                if verify_images:
                    try:
                        with Image.open(full_img_path) as img:
                            img.verify()
                    except Exception as e:
                        warnings.warn(f"Row {row_idx}: Corrupt image file '{full_img_path}': {e}. Skipping sample.")
                        continue

                self.samples.append({
                    'text': text,
                    'image_path': full_img_path,
                    'label': label
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[str, Any, int]:
        """
        Returns:
            text (str): Tweet text.
            image (PIL.Image or Tensor): Loaded RGB image (or transformed image).
            label (int): Integer label (0 for not_informative, 1 for informative).
        """
        sample = self.samples[index]
        img_path = sample['image_path']
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            warnings.warn(f"Failed to load image at '{img_path}': {e}. Returning placeholder image.")
            image = Image.new('RGB', (224, 224), color=0)

        if self.transform is not None:
            image = self.transform(image)

        return sample['text'], image, sample['label']


if __name__ == '__main__':
    data_directory = get_data_dir()

    print(f"Initializing CrisisMMDDataset (split='train') from: {data_directory}")
    train_dataset = CrisisMMDDataset(split='train', data_dir=data_directory)
    print(f"Total dataset size: {len(train_dataset)}\n")

    print("--- First 3 Samples ---")
    for i in range(min(3, len(train_dataset))):
        text, img, label = train_dataset[i]
        img_size = img.size if isinstance(img, Image.Image) else img.shape
        print(f"Sample {i + 1}:")
        print(f"  Text: {text}")
        print(f"  Image Size: {img_size}")
        print(f"  Label: {label}")
        print("-" * 50)
