import tensorflow as tf
from tensorflow.keras import applications, layers, optimizers, Model, callbacks
import numpy as np
import xml.etree.ElementTree as ET
import cv2
import os
from tensorflow.keras.utils import Sequence
import re
from typing import List, Tuple, Dict, Optional
import logging

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Config:
    """モデルとデータの設定を管理するクラス"""
    INPUT_SIZE = (299, 299)
    BATCH_SIZE = 8
    MAX_BOXES = 2
    CLASS_MAP = {'player': 0}
    EPOCHS = 100
    LEARNING_RATE = 1e-4

def create_model(input_shape: Tuple[int, int, int] = (299, 299, 3)) -> Model:
    """改善されたモデル作成関数"""
    base_model = applications.InceptionResNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # 転移学習の設定
    for layer in base_model.layers[:500]:
        layer.trainable = False
            
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)  # 過学習防止
    
    # バウンディングボックス出力
    bbox_output = layers.Dense(128, activation='relu')(x)
    bbox_output = layers.Dense(8, activation='linear')(bbox_output)
    bbox_output = layers.Reshape((2, 4), name='bbox')(bbox_output)
    
    # クラス分類出力
    class_output = layers.Dense(64, activation='relu')(x)
    class_output = layers.Dense(2, activation='sigmoid', name='class')(class_output)
    
    model = Model(inputs=base_model.input, outputs=[bbox_output, class_output])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        loss={
            'bbox': 'huber',  # MSEよりも外れ値に対して頑健
            'class': 'binary_crossentropy'
        },
        metrics={
            'bbox': 'mse',
            'class': ['accuracy', 'AUC']
        }
    )
    
    return model

def train_model(X: np.array, t: np.array):
    """モデルの学習を実行する関数"""
    try:
        # モデルの作成
        model = create_model()
        
        # コールバックの設定
        callbacks_list = [
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=3,
                min_lr=1e-6
            ),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            callbacks.ModelCheckpoint(
                'best_model.keras',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # モデルの学習
        history = model.fit(
            X,
            t,
            batch_size=64,
            epochs=Config.EPOCHS,
            callbacks=callbacks_list, 
            validation_split = 0.2
        )
        
        # モデルの保存
        model.save('player_detect_model.keras')
        
        return history
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise
    
import pickle
if __name__ == "__main__":
    with open('datasets.pkl', "rb") as f:
        images, targets = pickle.load(f)
    
    if 1: 
        try:
            history = train_model(images, targets)
            logging.info("Training completed successfully")
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")