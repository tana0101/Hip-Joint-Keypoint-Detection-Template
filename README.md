# Hip-Joint-Keypoint-Detection-Template

## Overview | 概述

這是一個使用 PyTorch 所實作的 **髖關節關鍵點偵測訓練模板**。 

## Features | 功能特色

- 使用 EfficientNet-V2 預訓練模型作為骨幹
- 自訂 Dataset，可輕鬆擴充至不同數量的關鍵點
- 自動計算像素誤差（Pixel Error）作為驗證指標
- 內建關鍵點可視化，方便確認標註與預測結果
- 自動繪製訓練與驗證的 Loss / Pixel Error 曲線並儲存
- 會自動保存驗證集表現最佳的模型

## Project Structure | 專案結構

```
├── train.py # 主訓練腳本
├── data/ # 存放影像與標註
│ ├── train/
│ │ ├── images/ # 訓練影像
│ │ └── annotations/ # 訓練 CSV 標註
│ └── val/
│ ├── images/ # 驗證影像
│ └── annotations/ # 驗證 CSV 標註
├── models/ # 保存訓練好的模型
└── logs/ # 訓練過程的圖表
```

## Annotation Format | 標註格式

- 每張圖片對應一個 `.csv`，內容類似：
```
"(x1,y1)","(x2,y2)",...,"(x12,y12)"
```

## Train | 訓練

可直接修改 `train.py` 的主函數設定：
```
if __name__ == "__main__":
    main(epochs=200, lr=1e-2, batch_size=32)
```
執行
```
python train.py
```
訓練過程中會自動：
- 顯示第一張圖的關鍵點位置
- 每個 epoch 輸出 Train / Val Loss 與 Pixel Error
- 保存最佳模型至 models/
- 繪製訓練曲線至 logs/

## Result | 結果
- 最佳模型會存為：
```
models/efficientnetv2s_keypoint_{epochs}_{lr}_{batch_size}_best.pth
```
- 訓練過程圖表會存為：
```
logs/efficientnetv2s_training_plot_{epochs}_{lr}_{batch_size}.png
```
