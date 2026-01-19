import os
from ultralytics import YOLO

def main():
    # 设置训练参数
    base_dir = "D:/VScode/xiangmu/recognize-main"
    yaml_path = os.path.join(base_dir, 'faces.yaml')

    # 加载 YOLOv8 模型
    model = YOLO('D:/VScode/xiangmu/recognize-main/yolov8n.pt')  # 加载预训练的基础模型

    results = model.train(
        data=yaml_path,       
        
        # --- 针对 FER2013 的核心调整 ---
        epochs=120,           # FER2013 很难收敛，需要更多轮次，设为 100-150 之间。
        imgsz=128,            # 【重要】原图只有48x48。设为128已经放大了2.5倍，足够了。
                            # 设太大会导致计算量浪费且引入插值噪声。
        
        batch=64,             # 【重要】因为图片变小了(128px)，显存占用很低。
                            # 请尽可能调大 batch（32, 64, 甚至 128），这能让梯度下降更稳定。
        
        # --- 针对 灰度图 的增强设置 ---
        hsv_h=0.0,            # 【关闭】色调增强。黑白图没有色调，开了只会干扰模型。
        hsv_s=0.0,            # 【关闭】饱和度增强。
        hsv_v=0.4,            # 【保留】明度（亮度）增强。模拟不同光照环境（这很关键）。
        
        # --- 几何增强 ---
        degrees=15.0,         # 允许 +/- 15度旋转（歪头是表情的一部分）。
        translate=0.1,        # 允许 10% 的平移。
        scale=0.2,            # 允许 20% 的缩放（模拟远近）。
        fliplr=0.5,           # 50% 水平翻转（左右脸对称，必须开）。
        flipud=0.0,           # 禁止垂直翻转。
        
        # --- 训练策略 ---
        lr0=0.01,             # 初始学习率
        lrf=0.01,             # 最终学习率
        patience=20,          # 早停机制
        optimizer='SGD',      # 对于小图、分类任务，SGD 通常比 AdamW 泛化性更好。
        
        # --- 其他 ---
        project='D:/VScode/xiangmu/recognize-main/runs/train',
        name='fer2013_optimized',
        save=True,
        pretrained=True       # 必须使用预训练权重
)

    print("训练完成！")

if __name__ == '__main__':
    main()
