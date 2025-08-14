import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageOps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import initialize_model

# 初始化常數
IMAGE_SIZE = 224
LOGS_DIR = "logs"
MODELS_DIR = "models"
DATA_DIR = "data"
POINTS_COUNT = 12

# Dataset Class
# 能夠自訂義資料集，讀取圖片和對應的關鍵點標註
class KeypointDataset(Dataset):
    def __init__(self, img_dir, annotation_dir, transform=None): # 先初始化參數與方法
        self.img_dir = img_dir # 圖片目錄
        self.annotation_dir = annotation_dir # 標註目錄
        self.transform = transform # 圖片預處理方法
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')]) # 圖片列表
        self.annotations = sorted([f for f in os.listdir(annotation_dir) if f.endswith('.csv')]) # 標註列表

    def __len__(self): # 回傳資料集的大小
        return len(self.images)

    def __getitem__(self, idx): # 根據不同的 index 去讀取對應的圖片和標註
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert("L")
        original_width, original_height = image.size # 原始圖片的寬度和高度

        annotation_path = os.path.join(self.annotation_dir, self.annotations[idx])
        keypoints = pd.read_csv(annotation_path, header=None).values.flatten()
        keypoints = [float(coord) for point in keypoints for coord in point.strip('"()').split(",")] # 將標註檔案內的關鍵點轉換為陣列

        if self.transform:
            image = self.transform(image) # 對圖片進行預處理
            new_width = image.shape[1]
            new_height = image.shape[2]
            scale_x = new_width / original_width # 計算寬度縮放比例
            scale_y = new_height / original_height # 計算高度縮放比例
            keypoints = [coord * scale_x if i % 2 == 0 else coord * scale_y for i, coord in enumerate(keypoints)] # 將關鍵點的位置根據縮放比例進行調整

        return image, torch.tensor(keypoints, dtype=torch.float32), (original_width, original_height)

# Display Image
# 用來顯示圖片和關鍵點，方便檢查資料集是否正確
def display_image(dataset, index):
    # Get the nth image and keypoints
    image, keypoints, original_size = dataset[index]
    print(f"Displaying image {index}")
    print("Original size:", original_size)
    print("Image shape:", image.shape)
    
    # Convert the image to a NumPy array
    image_np = image.permute(1, 2, 0).numpy()

    # Display the image
    plt.imshow(image_np, cmap='gray')
    plt.title(f"Image {index} with Keypoints")
    
    # Plot the keypoints with numbering
    for i in range(0, len(keypoints), 2):
        x = keypoints[i].item()  
        y = keypoints[i + 1].item()  
        plt.scatter(x, y, c='red', s=20)
        plt.text(x, y, f'{i//2 + 1}', color='yellow', fontsize=12)  # Add number next to each point

    plt.show()

# Pixel Error
# 計算預測的關鍵點與實際標註的關鍵點之間的像素誤差
# 會先將預測和標註的關鍵點轉換為原始圖片大小，然後計算每個關鍵點的距離，最後取平均值
def calculate_pixel_error(preds, targets, img_size):
    preds = preds.reshape(POINTS_COUNT, 2)
    targets = targets.reshape(POINTS_COUNT, 2)
    original_width, original_height = img_size
    scale_x = original_width / IMAGE_SIZE
    scale_y = original_height / IMAGE_SIZE
    preds_scaled = preds * np.array([scale_x, scale_y])
    targets_scaled = targets * np.array([scale_x, scale_y])
    pixel_distances = np.linalg.norm(preds_scaled - targets_scaled, axis=1)
    return np.mean(pixel_distances)

# Plotting
# 用來繪製訓練過程中的損失和像素誤差
def plot_training_progress(train_losses, val_losses, train_pe, val_pe, loss_ylim=None, pixel_error_ylim=None):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss over epochs")
    if loss_ylim:
        plt.ylim(loss_ylim)
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(train_pe, label="Train Pixel Error")
    plt.plot(val_pe, label="Val Pixel Error")
    plt.title("Pixel Error over epochs")
    if pixel_error_ylim:
        plt.ylim(pixel_error_ylim)
    plt.legend()
    plt.tight_layout()

# Main Training
def main(epochs=200, lr=1e-2, batch_size=32, model_name="efficientnet_ms_cbam_3scales"):
    
    
    # ================================================================== #
    #                          1. 準備資料集
    # ================================================================== #
    
    # 對圖片進行預處理：要將圖片調整成模型可以接受的大小，並進行一些增強處理
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), # 將圖片轉換為灰階
        transforms.Lambda(lambda img: ImageOps.equalize(img)),# 對圖片進行直方圖均衡化
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # 調整圖片大小，變成模型可以接受的大小
        transforms.ToTensor(), # 將圖片轉換為 Tensor，模型的輸入格式
    ])
    # 準備訓練和驗證資料集
    train_dataset = KeypointDataset(os.path.join(DATA_DIR, 'train/images'),
                                    os.path.join(DATA_DIR, 'train/annotations'),
                                    transform=transform)
    val_dataset = KeypointDataset(os.path.join(DATA_DIR, 'val/images'),
                                  os.path.join(DATA_DIR, 'val/annotations'),
                                  transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=4)
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    display_image(train_dataset, 0)  # 顯示第一張訓練圖片和其關鍵點

    # ================================================================== #
    #                          2. 準備模型與訓練用的參數
    # ================================================================== #
    
    # 初始化模型、損失函數和優化器
    model = initialize_model(model_name, num_points=POINTS_COUNT) # 根據模型名稱初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 設置硬體設備
    model.to(device) # 將模型移動到 GPU 或 CPU (硬體設備)上
    
    criterion = nn.MSELoss() # Loss function: 計算預測的關鍵點與實際標註的關鍵點之間的均方誤差
    optimizer = optim.Adam(model.parameters(), lr=lr) # Optimizer: 使用 Adam 優化器來更新模型的權重

    # 用來記錄每個 epoch 的訓練和驗證損失
    epoch_losses, epoch_pixel_errors = [], []
    val_losses, val_pixel_errors = [], []
    best_val_loss = float('inf') 
    best_model_state = None # 用來儲存最佳模型的狀態

    # ================================================================== #
    #                          3. 開始訓練
    # ================================================================== #
    
    for epoch in range(epochs):
        
        # ----------------- 訓練階段 ----------------- #
        model.train()
        running_loss, pixel_error_values = 0, []
        for images, keypoints, original_sizes in train_loader: # 以 batch 為單位，從訓練資料集中讀取圖片和關鍵點，並進行訓練
            
            # 進行訓練的過程
            images, keypoints = images.to(device), keypoints.to(device) # 將圖片和關鍵點移動到 GPU 或 CPU (硬體設備)上
            optimizer.zero_grad() # 初始化將前一個 batch 的梯度清零
            outputs = model(images) # Forward pass: 將圖片輸入模型，得到預測的關鍵點
            loss = criterion(outputs, keypoints) # 計算損失，預測和標註關鍵點的差距
            loss.backward() # Backward pass: 計算梯度
            optimizer.step() # 更新模型的權重
            running_loss += loss.item() # 累加每個 batch 的損失

            # 計算每個 batch 的像素誤差
            preds = outputs.cpu().detach().numpy() # 預測的關鍵點
            targets = keypoints.cpu().numpy() # 實際標註的關鍵點
            
            # 將原始圖片的寬度和高度轉換為 numpy 陣列，讓其可以進行後續的計算
            widths, heights = original_sizes
            widths = widths.cpu().numpy()  
            heights = heights.cpu().numpy() 
            original_sizes = [(w, h) for w, h in zip(widths, heights)]
            
            # 計算每個 batch 的像素誤差
            # batch 中的每張圖片都會計算一次像素誤差
            for i in range(len(original_sizes)):
                img_size = original_sizes[i]
                pixel_error = calculate_pixel_error(preds[i], targets[i], img_size)
                pixel_error_values.append(pixel_error)

        epoch_losses.append(running_loss / len(train_loader))
        epoch_pixel_errors.append(np.mean(pixel_error_values))

        # ----------------- 驗證階段 ----------------- #
        # 在每個 epoch 訓練結束後，用該模型在驗證集上進行評估
        model.eval()
        val_loss, val_pixel_error_values = 0, []
        with torch.no_grad(): 
            for images, keypoints, original_sizes in val_loader: # 從驗證集中讀取圖片和關鍵點，並進行驗證
                # 進行驗證的過程
                images, keypoints = images.to(device), keypoints.to(device)
                outputs = model(images)
                
                # 計算損失，預測和標註關鍵點的差距。
                # 用以評估該模型在驗證集上的表現
                loss = criterion(outputs, keypoints) 
                val_loss += loss.item()
                preds = outputs.cpu().detach().numpy()
                targets = keypoints.cpu().numpy()
                
                widths, heights = original_sizes
                widths = widths.cpu().numpy()  
                heights = heights.cpu().numpy() 
                original_sizes = [(w, h) for w, h in zip(widths, heights)]
                
                for i in range(len(original_sizes)):
                    img_size = original_sizes[i]
                    pixel_error = calculate_pixel_error(preds[i], targets[i], img_size)
                    val_pixel_error_values.append(pixel_error)

        val_losses.append(val_loss / len(val_loader))
        val_pixel_errors.append(np.mean(val_pixel_error_values))

        # 輸出每個 epoch 的訓練和驗證結果
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {epoch_losses[-1]:.4f} Pixel Error: {epoch_pixel_errors[-1]:.2f} "
              f"| Val Loss: {val_losses[-1]:.4f} Pixel Error: {val_pixel_errors[-1]:.2f}")

        # 保存在驗證時表現最佳的模型
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_model_state = model.state_dict()
            print("Validation loss improved, saving model.")

    # ================================================================== #
    #                          4. 輸出結果
    # ================================================================== #
    
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    if best_model_state:
        model_path = f"{MODELS_DIR}/{model_name}_{epochs}_{lr}_{batch_size}_best.pth"
        torch.save(best_model_state, model_path)
        print(f"Best model saved to: {model_path}")

    plot_training_progress(epoch_losses, val_losses, epoch_pixel_errors, val_pixel_errors,
                           loss_ylim=(0, 300), pixel_error_ylim=(0, 200))
    plt.savefig(f"{LOGS_DIR}/{model_name}_{epochs}_{lr}_{batch_size}.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a keypoint detection model")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--model_name", type=str, default="efficientnet_ms_cbam_3scales", help="Model name")
    args = parser.parse_args()

    main(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, model_name=args.model_name) # 訓練模型，並且設置訓練的參數
    # python train.py --epochs 200 --lr 1e-2 --batch_size 32 --model_name efficientnet_ms_cbam_3scales