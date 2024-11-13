import cv2
import numpy as np
from tensorflow.keras import models
import logging
from typing import Tuple, Optional
from pathlib import Path
import time
from dataclasses import dataclass
import tensorflow as tf
from tqdm import tqdm
import datetime

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class VideoConfig:
    """ビデオ処理の設定を管理するクラス"""
    INPUT_SIZE = (299, 299)
    CONFIDENCE_THRESHOLD = 0.5
    BATCH_SIZE = 4  # バッチ処理用
    MAX_FRAMES = None  # Noneの場合は全フレーム処理
    PROCESS_EVERY_N_FRAMES = 1  # フレームスキップの設定

class VideoProcessor:
    """ビデオ処理を管理するクラス"""
    def __init__(
        self,
        model_path: str,
        input_path: str,
        output_path: str,
        config: VideoConfig = VideoConfig()
    ):
        self.config = config
        self.setup_model(model_path)
        self.setup_video(input_path, output_path)
        self.processing_stats = {
            'processed_frames': 0,
            'start_time': None,
            'batch_times': []
        }
        
    def setup_model(self, model_path: str) -> None:
        """モデルのセットアップと最適化"""
        try:
            logging.info("Loading model...")
            self.model = models.load_model(model_path)
            
            # モデルの最適化（TensorRT変換を試みる）
            try:
                if tf.config.list_physical_devices('GPU'):
                    self.model = tf.keras.models.clone_model(self.model)
                    converter = tf.experimental.tensorrt.Converter(
                        input_saved_model_dir=model_path,
                        precision_mode='FP16'
                    )
                    self.model = converter.convert()
                    logging.info("Model optimized with TensorRT")
            except Exception as e:
                logging.warning(f"TensorRT optimization failed: {e}")
                
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
            
    def setup_video(self, input_path: str, output_path: str) -> None:
        """ビデオキャプチャとライターのセットアップ"""
        try:
            self.cap = cv2.VideoCapture(input_path)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video file: {input_path}")
                
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 実際に処理するフレーム数を計算
            self.frames_to_process = self.total_frames
            if self.config.MAX_FRAMES:
                self.frames_to_process = min(self.total_frames, self.config.MAX_FRAMES)
            self.frames_to_process = self.frames_to_process // self.config.PROCESS_EVERY_N_FRAMES
            
            # 出力ビデオの設定
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(
                str(output_path),
                fourcc,
                self.fps/1.5,
                (self.width, self.height)
            )
            
            logging.info(f"Video setup complete:")
            logging.info(f"- Resolution: {self.width}x{self.height}")
            logging.info(f"- FPS: {self.fps}")
            logging.info(f"- Total frames: {self.total_frames}")
            logging.info(f"- Frames to process: {self.frames_to_process}")
            
        except Exception as e:
            logging.error(f"Error setting up video: {e}")
            raise
            
    def update_processing_stats(self, batch_size: int) -> dict:
        """処理統計の更新と計算"""
        self.processing_stats['processed_frames'] += batch_size
        current_time = time.time()
        
        if not self.processing_stats['start_time']:
            self.processing_stats['start_time'] = current_time
            return {}
            
        elapsed_time = current_time - self.processing_stats['start_time']
        frames_processed = self.processing_stats['processed_frames']
        
        # 処理速度の計算
        fps = frames_processed / elapsed_time
        
        # 残り時間の推定
        remaining_frames = self.frames_to_process - frames_processed
        estimated_time_remaining = remaining_frames / fps if fps > 0 else 0
        
        return {
            'fps': fps,
            'elapsed_time': elapsed_time,
            'estimated_time_remaining': estimated_time_remaining,
            'progress_percentage': (frames_processed / self.frames_to_process) * 100
        }
        
    def format_time(self, seconds: float) -> str:
        """秒数を読みやすい形式に変換"""
        return str(datetime.timedelta(seconds=int(seconds)))
        
    def process_video(self) -> None:
        """ビデオの処理メインループ"""
        try:
            frame_count = 0
            batch_frames = []
            
            # tqdmプログレスバーの設定
            pbar = tqdm(total=self.frames_to_process, unit='frames')
            
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret or (self.config.MAX_FRAMES and frame_count >= self.config.MAX_FRAMES):
                    break
                    
                if frame_count % self.config.PROCESS_EVERY_N_FRAMES == 0:
                    batch_frames.append(frame)
                    
                    if len(batch_frames) == self.config.BATCH_SIZE:
                        processed_frames = self.process_batch(batch_frames)
                        for processed_frame in processed_frames:
                            self.out.write(processed_frame)
                            
                        # 統計情報の更新
                        stats = self.update_processing_stats(len(batch_frames))
                        if stats:
                            # プログレスバーの説明テキストを更新
                            pbar.set_description(
                                f"FPS: {stats['fps']:.1f} | "
                                f"Elapsed: {self.format_time(stats['elapsed_time'])} | "
                                f"Remaining: {self.format_time(stats['estimated_time_remaining'])}"
                            )
                            
                        pbar.update(len(batch_frames))
                        batch_frames = []
                        
                frame_count += 1
                        
            # 残りのフレームを処理
            if batch_frames:
                processed_frames = self.process_batch(batch_frames)
                for processed_frame in processed_frames:
                    self.out.write(processed_frame)
                pbar.update(len(batch_frames))
                
            pbar.close()
            
            # 最終的な処理結果の表示
            final_stats = self.update_processing_stats(0)
            logging.info("\nProcessing completed:")
            logging.info(f"- Total frames processed: {self.processing_stats['processed_frames']}")
            logging.info(f"- Average FPS: {final_stats['fps']:.1f}")
            logging.info(f"- Total time: {self.format_time(final_stats['elapsed_time'])}")
            
        except Exception as e:
            logging.error(f"Error processing video: {e}")
            raise
            
        finally:
            self.cleanup()
            
    # 他のメソッドは変更なし
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """フレームの前処理"""
        processed = cv2.resize(frame, self.config.INPUT_SIZE)
        processed = processed.astype(np.float32) / 255.0
        return processed
        
    def process_batch(self, frames: list) -> list:
        """フレームのバッチ処理"""
        if not frames:
            return []
            
        batch = np.stack([self.preprocess_frame(frame) for frame in frames])
        predictions = self.model.predict(batch, verbose=0)
        
        boxes_batch, classes_batch = predictions
        
        processed_frames = []
        for frame, boxes, classes in zip(frames, boxes_batch, classes_batch):
            processed_frame = self.draw_boxes(frame, boxes, classes)
            processed_frames.append(processed_frame)
            
        return processed_frames
        
    def draw_boxes(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        classes: np.ndarray
    ) -> np.ndarray:
        """バウンディングボックスの描画"""
        frame_copy = frame.copy()
        
        for box, confidence in zip(boxes, classes):
            xmin, ymin, xmax, ymax = box
            
            xmin = int(xmin * self.width)
            ymin = int(ymin * self.height)
            xmax = int(xmax * self.width)
            ymax = int(ymax * self.height)
            
            xmin = max(0, min(xmin, self.width))
            ymin = max(0, min(ymin, self.height))
            xmax = max(0, min(xmax, self.width))
            ymax = max(0, min(ymax, self.height))
            
            cv2.rectangle(
                frame_copy,
                (xmin, ymin),
                (xmax, ymax),
                (0, 255, 0),
                2
            )
            
        return frame_copy
        
    def cleanup(self) -> None:
        """リソースの解放"""
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

def main():
    # 設定
    config = VideoConfig()
    config.MAX_FRAMES = None  # 全フレーム処理
    config.PROCESS_EVERY_N_FRAMES = 2  # 2フレームごとに処理
    config.CONFIDENCE_THRESHOLD = 0.5
    
    try:
        processor = VideoProcessor(
            model_path='models\player_detect_model_InceptionResNetV2.keras',
            input_path=r'保存先フォルダのパス\2024関東学生秋季リーグ戦 森川亮介(法政大)vs吉田樹(早稲田大).mp4',
            output_path='output_video/output_video_4.mp4',
            config=config
        )
        
        processor.process_video()
        logging.info("Video processing completed successfully")
        
    except Exception as e:
        logging.error(f"Video processing failed: {e}")
        raise

if __name__ == "__main__":
    main()