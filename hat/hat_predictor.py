import cv2
import torch
import numpy as np
import torch.nn.functional as F

from hat.hat_arch import HAT


class HATPredictor:
    def __init__(self, model_path, device='cuda'):
        """
        初始化 HAT 超分模型
        :param model_path: .pth 权重文件的路径
        :param device: 运行设备 ('cuda' 或 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.upscale = 2
        self.window_size = 16

        # 1. 初始化模型网络 (参数严格对照提供的 YAML 文件中的 network_g)
        self.model = HAT(
            upscale=self.upscale,
            in_chans=3,
            img_size=64,
            window_size=self.window_size,
            compress_ratio=24,
            squeeze_factor=24,
            conv_scale=0.01,
            overlap_ratio=0.5,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=144,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection='1conv'
        )

        # 2. 加载预训练权重
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)

        # 根据 YAML 配置 param_key_g: 'params_ema'，优先读取 EMA 参数
        if 'params_ema' in checkpoint:
            state_dict = checkpoint['params_ema']
        elif 'params' in checkpoint:
            state_dict = checkpoint['params']
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.model.to(self.device)

    def predict(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        进行图像超分推理
        :param img_bgr: OpenCV 读取的原始 BGR 图像 (numpy array)
        :return: 超分后的 BGR 图像 (numpy array)
        """
        # --- 图像预处理 ---
        # OpenCV 默认读取的是 BGR，HAT 模型推理通常需要 RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # 归一化到 [0, 1] 并转换维度为 (1, C, H, W)
        img_tensor = torch.from_numpy(np.transpose(img_rgb, (2, 0, 1))).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # --- Padding 逻辑 ---
        # Window Attention 要求输入的宽高必须是 window_size 的整数倍
        h_old, w_old = img_tensor.shape[2:]
        pad_h = (self.window_size - h_old % self.window_size) % self.window_size
        pad_w = (self.window_size - w_old % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            # 使用 reflect 镜像填充，避免边缘产生黑边
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), 'reflect')

        # --- 前向推理 ---
        with torch.no_grad():
            output_tensor = self.model(img_tensor)

        # --- 移除 Padding 产生的多余部分 ---
        if pad_h > 0 or pad_w > 0:
            # 输出的尺寸已经被放大了 upscale 倍
            output_tensor = output_tensor[:, :, :h_old * self.upscale, :w_old * self.upscale]

        # --- 图像后处理 ---
        # 截断数据到 [0, 1] 区间，并转换为 numpy array
        output_arr = output_tensor.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_arr = np.transpose(output_arr, (1, 2, 0))
        output_arr = (output_arr * 255.0).round().astype(np.uint8)

        # 将 RGB 转回 BGR 供后续 OpenCV 管线使用
        output_bgr = cv2.cvtColor(output_arr, cv2.COLOR_RGB2BGR)

        return output_bgr


if __name__ == '__main__':
    weights_path = 'C:/Users/Admin/PycharmProjects/arrow_SR/Model_path/HAT_S_ftv5.pth'
    sr_model = HATPredictor(model_path=weights_path, device='cuda')

    # 2. 在处理流中读取/接收低分辨率图像
    lr_image = cv2.imread('C:/Users/Admin/Desktop/arrow/arrow/images_lr/Phyt_2024_236_114494_004_002_mol.png')

    # 3. 进行推理
    sr_image = sr_model.predict(lr_image)

    # 4. 继续后续的管线处理（例如保存、传入 OCR 引擎等）
    cv2.imwrite('output_sr.png', sr_image)
