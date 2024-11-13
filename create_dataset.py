import tensorflow as tf
from tensorflow.keras.utils import Sequence
import numpy as np
import cv2
import os
import re  # reモジュールを追加
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path
from itertools import chain
from dataclasses import dataclass
from tqdm import tqdm
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_generation.log')
    ]
)

@dataclass
class DataGeneratorConfig:
    """データジェネレーターの設定"""
    INPUT_SIZE: Tuple[int, int] = (299, 299)
    MAX_BOXES: int = 2
    CLASS_MAP: Dict[str, int] = None
    SHOW_PROGRESS: bool = True
    
    def __post_init__(self):
        if self.CLASS_MAP is None:
            self.CLASS_MAP = {'player': 0}

class MultiSourceDataGenerator:
    """複数ソースからのデータセット生成を管理するクラス"""
    def __init__(
        self,
        image_dirs: List[str],
        annotation_dirs: List[str],
        config: DataGeneratorConfig = DataGeneratorConfig()
    ):
        """
        Args:
            image_dirs: 画像ディレクトリのリスト
            annotation_dirs: アノテーションディレクトリのリスト
            config: 設定オブジェクト
        """
        self.start_time = time.time()
        if len(image_dirs) != len(annotation_dirs):
            raise ValueError("Number of image and annotation directories must match")
            
        self.config = config
        logging.info("Initializing data sources...")
        self.sources = self._initialize_sources(image_dirs, annotation_dirs)
        self.total_samples = sum(source['total_samples'] for source in self.sources)
        logging.info(f"Initialized {len(self.sources)} data sources with total {self.total_samples} samples")
        
    def _initialize_sources(
        self,
        image_dirs: List[str],
        annotation_dirs: List[str]
    ) -> List[Dict]:
        """各ソースの初期化"""
        sources = []
        for img_dir, ann_dir in tqdm(
            zip(image_dirs, annotation_dirs),
            desc="Initializing sources",
            disable=not self.config.SHOW_PROGRESS
        ):
            try:
                source_data = self._process_source(img_dir, ann_dir)
                if source_data['total_samples'] > 0:
                    sources.append(source_data)
                    logging.info(
                        f"Added source: {img_dir} with {source_data['total_samples']} samples "
                        f"({len(sources)} sources total)"
                    )
                else:
                    logging.warning(f"No valid samples found in {img_dir}")
            except Exception as e:
                logging.error(f"Error processing source {img_dir}: {str(e)}")
                continue
                
        return sources
        
    def _process_source(
        self,
        image_dir: str,
        annotation_dir: str
    ) -> Dict:
        """個別ソースの処理"""
        valid_indexes = self._extract_indexes(annotation_dir)
        image_files = self._get_matching_files(image_dir, valid_indexes, '.png')
        annotation_files = self._get_matching_files(annotation_dir, valid_indexes, '.xml')
        
        return {
            'image_dir': image_dir,
            'annotation_dir': annotation_dir,
            'image_files': image_files,
            'annotation_files': annotation_files,
            'total_samples': len(image_files)
        }
        
    @staticmethod
    def _extract_indexes(directory: str) -> List[int]:
        """ファイル名からインデックスを抽出"""
        indexes = []
        for file in os.listdir(directory):
            match = re.search(r'frame_(\d+)', file)
            if match:
                indexes.append(int(match.group(1)))
        return sorted(indexes)
        
    @staticmethod
    def _get_matching_files(
        directory: str,
        indexes: List[int],
        extension: str
    ) -> List[str]:
        """指定されたインデックスに一致するファイルを取得"""
        return sorted([
            os.path.join(directory, f) for f in os.listdir(directory)
            if f.endswith(extension) and any(f'frame_{idx}' in f for idx in indexes)
        ])
        
    def generate_dataset(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """全ソースからデータセットを生成"""
        if not self.sources:
            logging.error("No valid sources found to generate dataset")
            return np.array([]), {'bbox': np.array([]), 'class': np.array([])}

        all_images = []
        all_bboxes = []
        all_classes = []
        
        processed_samples = 0
        failed_samples = 0
        start_time = time.time()
        
        for source_idx, source in enumerate(self.sources, 1):
            logging.info(f"Processing source {source_idx}/{len(self.sources)}: {source['image_dir']}")
            
            images, bboxes, classes, failed = self._process_source_files(
                source['image_files'],
                source['annotation_files']
            )
            
            all_images.extend(images)
            all_bboxes.extend(bboxes)
            all_classes.extend(classes)
            
            processed_samples += len(images)
            failed_samples += failed
            
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:  # 0による除算を防ぐ
                samples_per_second = processed_samples / elapsed_time
                estimated_remaining = (self.total_samples - processed_samples) / samples_per_second if self.total_samples > processed_samples else 0
            else:
                samples_per_second = 0
                estimated_remaining = 0
            
            logging.info(
                f"Progress: {processed_samples}/{self.total_samples} samples "
                f"({processed_samples/self.total_samples*100:.1f}% if self.total_samples > 0 else 0%) | "
                f"Failed: {failed_samples} | "
                f"Speed: {samples_per_second:.1f} samples/s | "
                f"Est. remaining: {estimated_remaining/60:.1f} min"
            )
            
        elapsed_time = time.time() - start_time
        if processed_samples > 0:
            logging.info(
                f"Dataset generation completed in {elapsed_time/60:.1f} min | "
                f"Total samples: {processed_samples} | "
                f"Failed: {failed_samples} ({failed_samples/processed_samples*100:.1f}%)"
            )
        else:
            logging.warning("No samples were processed successfully")
            
        return (
            np.array(all_images),
            {
                'bbox': np.array(all_bboxes),
                'class': np.array(all_classes)
            }
        )
        
    def _process_source_files(
        self,
        image_files: List[str],
        annotation_files: List[str]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], int]:
        """ソースファイルの処理"""
        images = []
        bboxes = []
        classes = []
        failed_count = 0
        
        for img_file, ann_file in tqdm(
            zip(image_files, annotation_files),
            total=len(image_files),
            desc="Processing files",
            disable=not self.config.SHOW_PROGRESS
        ):
            try:
                image = self._load_image(img_file)
                bbox, class_label = self._load_annotation(ann_file)
                
                images.append(image)
                bboxes.append(bbox)
                classes.append(class_label)
                
            except Exception as e:
                failed_count += 1
                logging.error(f"Error processing files {img_file} and {ann_file}: {str(e)}")
                continue
                
        return images, bboxes, classes, failed_count
        
    def _load_image(self, file_path: str) -> np.ndarray:
        """画像の読み込みと前処理"""
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Failed to load image: {file_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.config.INPUT_SIZE)
        image = image.astype(np.float32) / 255.0
        
        return image
        
    def _load_annotation(
        self,
        file_path: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """アノテーションの読み込みと前処理"""
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        
        bboxes = []
        labels = []
        
        for obj in root.findall('object')[:self.config.MAX_BOXES]:
            label = obj.find('name').text
            bbox = obj.find('bndbox')
            
            # バウンディングボックスの正規化
            xmin = float(bbox.find('xmin').text) / width
            ymin = float(bbox.find('ymin').text) / height
            xmax = float(bbox.find('xmax').text) / width
            ymax = float(bbox.find('ymax').text) / height
            
            bboxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.config.CLASS_MAP.get(label, 0))
            
        # 必要な数のダミーボックスを追加
        while len(bboxes) < self.config.MAX_BOXES:
            bboxes.append([0, 0, 0, 0])
            labels.append(0)
            
        return np.array(bboxes), np.array(labels)

def create_dataset_from_multiple_sources(
    image_dirs: str,
    annotation_dirs: str,
    config: DataGeneratorConfig = DataGeneratorConfig()
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    複数のソースディレクトリからデータセットを作成する便利関数
    
    Args:
        image_dirs: 画像ディレクトリのパス
        annotation_dirs: アノテーションディレクトリのパス
        config: データジェネレーターの設定
        
    Returns:
        画像データと対応するアノテーションのタプル
    """
    logging.info("Starting dataset creation from multiple sources")
    start_time = time.time()
    
    # ディレクトリの存在チェック
    if not os.path.exists(image_dirs) or not os.path.exists(annotation_dirs):
        raise ValueError(f"Directory not found: {image_dirs} or {annotation_dirs}")
    
    image_dirs = [os.path.join(image_dirs, f) for f in os.listdir(image_dirs)]
    annotation_dirs = [os.path.join(annotation_dirs, f) for f in os.listdir(annotation_dirs)]
    
    valid_dirs = []
    for image_dir, annotation_dir in zip(image_dirs, annotation_dirs):
        if os.path.exists(image_dir) and os.path.exists(annotation_dir):
            valid_dirs.append((image_dir, annotation_dir))
        else:
            logging.warning(f"Skipping invalid directory structure at {image_dir}, {annotation_dir}")
            
    if not valid_dirs:
        raise ValueError("No valid directory pairs found")
            
    logging.info(f"Found {len(valid_dirs)} valid directory pairs")
    
    generator = MultiSourceDataGenerator(
        [d[0] for d in valid_dirs],
        [d[1] for d in valid_dirs],
        config
    )
    
    result = generator.generate_dataset()
    
    total_time = time.time() - start_time
    logging.info(
        f"Dataset creation completed in {total_time/60:.1f} minutes | "
        f"Final dataset size: {len(result[0])} samples"
    )
    
    return result

import pickle

# 使用例
if __name__ == "__main__":
    # 設定
    config = DataGeneratorConfig(
        INPUT_SIZE=(299, 299),
        MAX_BOXES=2,
        CLASS_MAP={'player': 0},
        SHOW_PROGRESS=True
    )
    
    # ディレクトリの指定
    image_dirs = 'photos'
    ann_dirs = 'annotation'
    
    try:
        # データセットの生成
        with open('datasets.pkl', "rb") as f:
            images, targets = pickle.load(f)
            
        logging.info(f"Successfully created dataset with {len(images)} samples")
        logging.info(f"Image shape: {images.shape}")
        logging.info(f"Bbox shape: {targets['bbox'].shape}")
        logging.info(f"Class shape: {targets['class'].shape}")
        
        
        
    except Exception as e:
        logging.error(f"Failed to create dataset: {str(e)}")
        raise  # スタックトレースを表示するために例外を再送出