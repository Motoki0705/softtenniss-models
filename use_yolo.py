import cv2
import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import pandas as pd
from datetime import datetime

@dataclass
class TrackerConfig:
    """トラッキングの設定を保持するデータクラス"""
    model_path: str = 'yolov8s.pt'
    max_age: int = 30
    n_init: int = 3
    person_class_id: int = 0
    confidence_threshold: float = 0.5
    output_path: str = 'tracking_data'

@dataclass
class CourtConfig:
    """テニスコートの設定を保持するデータクラス"""
    # コートの実際の寸法（メートル）
    court_length: float = 23.77  # ベースライン間の距離
    court_width: float = 8.23    # シングルスコートの幅
    
    # コート座標のキーポイント（左下が原点(0,0)）
    court_points_real = np.float32([
        [0, 0],              # 左下
        [0, court_width],    # 左上
        [court_length, court_width],  # 右上
        [court_length, 0]    # 右下
    ])

class PersonTracker:
    """人物追跡を行うためのクラス"""
    
    def __init__(self, config: TrackerConfig):
        """
        Args:
            config (TrackerConfig): トラッキングの設定
        """
        self.config = config
        self.target_ids: List[int] = []
        self._init_models()

    def _init_models(self) -> None:
        """YOLOとDeep SORTモデルを初期化"""
        try:
            self.model = YOLO(self.config.model_path)
            self.tracker = DeepSort(
                max_age=self.config.max_age,
                n_init=self.config.n_init
            )
        except Exception as e:
            raise RuntimeError(f"モデルの初期化に失敗: {str(e)}")

    def process_frame(self, frame) -> Tuple[List, List]:
        """
        フレームを処理し、検出結果とトラックを返す
        
        Args:
            frame: 処理する画像フレーム
            
        Returns:
            Tuple[List, List]: 検出結果とトラックのリスト
        """
        results = self.model(frame)
        detections = self._format_detections(results)
        tracks = self.tracker.update_tracks(detections, frame=frame)
        return results, tracks

    def _format_detections(self, results) -> List:
        """
        YOLOの検出結果をDeep SORT用に整形
        
        Args:
            results: YOLOの検出結果
            
        Returns:
            List: Deep SORT用の検出結果リスト
        """
        detections = []
        for result in results:
            for box in result.boxes:
                if (box.cls == self.config.person_class_id and 
                    box.conf >= self.config.confidence_threshold):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append(([x1, y1, x2 - x1, y2 - y1], 
                                    float(box.conf), 'person'))
        return detections

    def draw_tracks(self, frame, tracks) -> None:
        """
        トラッキング結果を画像に描画
        
        Args:
            frame: 描画対象のフレーム
            tracks: トラッキング結果
        """
        for track in tracks:
            if not track.is_confirmed():
                continue
            if track.track_id in self.target_ids:
                self._draw_single_track(frame, track)

    def _draw_single_track(self, frame, track) -> None:
        """
        単一のトラックを画像に描画
        
        Args:
            frame: 描画対象のフレーム
            track: 描画するトラック情報
        """
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track.track_id}", 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 255, 0), 2)

class CourtCalibrator:
    """テニスコートのキャリブレーションを行うクラス"""
    
    def __init__(self, config: CourtConfig):
        self.config = config
        self.court_points_image = None
        self.homography_matrix = None
        
    def calibrate(self, frame: np.ndarray) -> None:
        """
        コートのキャリブレーションを実行
        
        Args:
            frame: キャリブレーション用の画像フレーム
        """
        # コートの四隅を選択するためのGUI
        points = []
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append([x, y])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('Calibration', frame)
        
        cv2.namedWindow('Calibration')
        cv2.setMouseCallback('Calibration', mouse_callback)
        cv2.imshow('Calibration', frame)
        
        while len(points) < 4:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyWindow('Calibration')
        
        self.court_points_image = np.float32(points)
        self.homography_matrix = cv2.getPerspectiveTransform(
            self.court_points_image, 
            self.config.court_points_real
        )
    
    def image_to_court_coords(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        画像座標をコート座標に変換
        
        Args:
            point: 画像上の座標 (x, y)
            
        Returns:
            Tuple[float, float]: コート上の実座標 (x, y) メートル単位
        """
        if self.homography_matrix is None:
            raise RuntimeError("キャリブレーションが実行されていません")
        
        point_transformed = cv2.perspectiveTransform(
            np.array([[point]], dtype=np.float32),
            self.homography_matrix
        )
        return tuple(point_transformed[0][0])

class PlayerTracker(PersonTracker):
    """人物追跡とコート上の位置推定を行うクラス"""
    
    def __init__(self, tracker_config: TrackerConfig, court_calibrator: CourtCalibrator):
        super().__init__(tracker_config)
        self.court_calibrator = court_calibrator
        self.player_positions: Dict[int, List[Dict]] = {}
        self.output_path = Path(tracker_config.output_path)
        self.output_path.mkdir(exist_ok=True)
        
    def update_player_position(self, track) -> None:
        """
        プレイヤーの位置を更新して記録
        
        Args:
            track: トラッキング情報
        """
        if not track.is_confirmed():
            return
            
        if track.track_id not in self.target_ids:
            return
            
        # バウンディングボックスの下端中央を人物の位置として使用
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        foot_point = (int((x1 + x2) / 2), y2)
        
        # コート座標に変換
        court_position = self.court_calibrator.image_to_court_coords(foot_point)
        
        # 位置データを記録
        position_data = {
            'timestamp': datetime.now().isoformat(),
            'court_x': court_position[0],
            'court_y': court_position[1],
            'image_x': foot_point[0],
            'image_y': foot_point[1]
        }
        
        if track.track_id not in self.player_positions:
            self.player_positions[track.track_id] = []
        self.player_positions[track.track_id].append(position_data)
    
    def save_tracking_data(self) -> None:
        """トラッキングデータをCSVファイルとして保存"""
        for player_id, positions in self.player_positions.items():
            df = pd.DataFrame(positions)
            output_file = self.output_path / f'player_{player_id}_positions.csv'
            df.to_csv(output_file, index=False)
            print(f"Saved tracking data for player {player_id} to {output_file}")

class VideoProcessor:
    """動画処理を行うクラス"""
    
    def __init__(self, video_path: str, tracker: PlayerTracker):
        self.video_path = Path(video_path)
        self.tracker = tracker
        self.cap = None
        self._current_tracks = []
    
    def __enter__(self):
        """コンテキストマネージャーのエントリーポイント"""
        self._setup_video_capture()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーの終了処理"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def _setup_video_capture(self) -> None:
        """ビデオキャプチャの設定"""
        if not self.video_path.exists():
            raise FileNotFoundError(f"動画ファイルが見つかりません: {self.video_path}")
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError("動画ファイルを開けませんでした")
            
    def calibrate_court(self) -> None:
        """最初のフレームを使用してコートのキャリブレーションを実行"""
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("キャリブレーション用のフレームを読み込めませんでした")
            
        print("コートの四隅をクリックしてください（左下→左上→右上→右下）")
        self.tracker.court_calibrator.calibrate(frame)
        
        # キャプチャを最初に巻き戻す
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def _mouse_callback(self, event, x, y, flags, param) -> None:
        """
        マウスクリックイベントのコールバック関数
        
        Args:
            event: マウスイベントの種類
            x, y: クリック位置の座標
            flags: イベントフラグ
            param: 追加パラメータ
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self._handle_click(x, y)

    def _handle_click(self, x: int, y: int) -> None:
        """
        クリックイベントの処理
        
        Args:
            x (int): クリックのx座標
            y (int): クリックのy座標
        """
        for track in self._current_tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            if x1 < x < x2 and y1 < y < y2:
                if track.track_id not in self.tracker.target_ids:
                    self.tracker.target_ids.append(track.track_id)
                    print(f"Target ID added: {track.track_id}")
                break
    
    def process_video(self) -> None:
        """動画を処理し、トラッキング結果を表示"""
        cv2.namedWindow("Tracking")
        cv2.setMouseCallback("Tracking", self._mouse_callback)

        continue_processing = True  # 'q' 押下後も解析を続行するためのフラグ

        while continue_processing:
            ret, frame = self.cap.read()
            if not ret:
                break

            results, tracks = self.tracker.process_frame(frame)
            self._current_tracks = tracks

            # トラッキング対象の位置を更新
            for track in tracks:
                self.tracker.update_player_position(track)

            # トラッキング結果を描画
            self.tracker.draw_tracks(frame, tracks)

            # コート座標系を可視化
            self._draw_court_overlay(frame)

            # フレームを表示し、'q' 押下で表示を終了
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                continue_processing = False

        cv2.destroyWindow("Tracking")  # 表示のみ終了、解析は動画終了まで続行
        
    def _draw_court_overlay(self, frame: np.ndarray) -> None:
        """
        コートのオーバーレイを描画
        
        Args:
            frame: 描画対象のフレーム
        """
        if self.tracker.court_calibrator.homography_matrix is not None:
            # コートの輪郭を描画
            court_points = self.tracker.court_calibrator.court_points_image
            for i in range(4):
                pt1 = tuple(map(int, court_points[i]))
                pt2 = tuple(map(int, court_points[(i + 1) % 4]))
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

def main():
    """メイン実行関数"""
    # 設定の初期化
    tracker_config = TrackerConfig()
    court_config = CourtConfig()
    
    # コートキャリブレーターの初期化
    court_calibrator = CourtCalibrator(court_config)
    
    # トラッカーの初期化
    tracker = PlayerTracker(tracker_config=tracker_config, court_calibrator=court_calibrator)

    # 動画プロセッサの初期化
    video_path = r"保存先フォルダのパス\2024関東学生秋季リーグ戦 森川亮介(法政大)vs吉田樹(早稲田大).mp4"  # 処理する動画のパスを指定
    with VideoProcessor(video_path, tracker) as video_processor:
        # コートのキャリブレーション
        video_processor.calibrate_court()
        
        # 動画を処理してトラッキング実行
        video_processor.process_video()
        
        # トラッキングデータを保存
        tracker.save_tracking_data()

if __name__ == "__main__":
    main()
