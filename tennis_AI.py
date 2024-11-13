import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
from typing import List, Dict, Tuple

class TennisStatsTracker:
    def __init__(self):
        # YOLOモデルの初期化
        self.player_detector = YOLO('yolov8n.pt')
        self.ball_detector = YOLO('custom_ball_detector.pt')  # ボール検出用の特別なモデル
        
        # コート領域の検出用パラメータ
        self.court_detector = cv2.createBackgroundSubtractorMOG2()
        
    def process_video(self, video_path: str) -> Dict:
        """動画を処理して統計データを抽出"""
        cap = cv2.VideoCapture(video_path)
        stats = {
            'rallies': [],
            'player_positions': [],
            'serve_data': [],
            'shot_types': []
        }
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # フレーム処理
            stats = self._process_frame(frame, stats)
            
        cap.release()
        return self._calculate_final_stats(stats)
    
    def _process_frame(self, frame: np.ndarray, stats: Dict) -> Dict:
        """各フレームを処理してデータを抽出"""
        # プレイヤーの検出と追跡
        player_results = self.player_detector(frame)
        player_positions = self._extract_player_positions(player_results)
        stats['player_positions'].append(player_positions)
        
        # ボールの検出と軌道追跡
        ball_results = self.ball_detector(frame)
        ball_trajectory = self._track_ball(ball_results)
        
        # ショットの種類を分類（サーブ、フォアハンド、バックハンド等）
        if ball_trajectory:
            shot_type = self._classify_shot_type(ball_trajectory, player_positions)
            stats['shot_types'].append(shot_type)
        
        return stats
    
    def _extract_player_positions(self, results) -> List[Tuple[int, int]]:
        """検出されたプレイヤーの位置を抽出"""
        positions = []
        for detection in results:
            if detection.label == 'person':
                x, y = detection.center
                positions.append((x, y))
        return positions
    
    def _track_ball(self, results) -> List[Tuple[int, int]]:
        """ボールの軌道を追跡"""
        ball_positions = []
        for detection in results:
            if detection.label == 'ball':
                x, y = detection.center
                ball_positions.append((x, y))
        return ball_positions
    
    def _classify_shot_type(self, ball_trajectory: List, player_positions: List) -> str:
        """ショットの種類を分類"""
        # 軌道とプレイヤーの位置関係から種類を判定
        # 実際の実装ではより複雑な分類ロジックが必要
        return 'forehand'  # 簡略化のため固定値を返す
    
    def _calculate_final_stats(self, stats: Dict) -> Dict:
        """収集したデータから最終的な統計を計算"""
        final_stats = {
            'total_shots': len(stats['shot_types']),
            'shot_distribution': pd.Series(stats['shot_types']).value_counts().to_dict(),
            'average_rally_length': np.mean([len(rally) for rally in stats['rallies']]),
            'court_coverage': self._calculate_court_coverage(stats['player_positions'])
        }
        return final_stats
    
    def _calculate_court_coverage(self, positions: List) -> float:
        """プレイヤーのコートカバー率を計算"""
        # コートを格子状に分割し、プレイヤーが訪れたセルの割合を計算
        court_grid = np.zeros((10, 10))
        for pos in positions:
            x, y = self._normalize_position(pos)
            court_grid[y][x] = 1
        return np.sum(court_grid) / court_grid.size
    
    def _normalize_position(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """座標をグリッド上の位置に正規化"""
        x, y = pos
        # 実際の実装では、コートの実寸とカメラアングルを考慮した変換が必要
        return (min(int(x/64), 9), min(int(y/64), 9))