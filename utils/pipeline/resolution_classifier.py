import os
import pickle
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, List
import numpy as np
import cv2

# 避免 joblib 在部分 Windows 环境调用系统命令探测物理核心时产生编码异常。
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

from utils.base.paths import PROJECT_ROOT
CLASSIFIER_PATH = PROJECT_ROOT / "model" / "classifier_fixed.pkl"
CROP_DIR = PROJECT_ROOT / "crop_dataset"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ResolutionClassifier:
    """
    分辨率分类器，用于评估图像是否为低清图像，以决定后续是否需要进行增强/超分处理。
    纯 CPU 运行，无需 CUDA 支持。
    """

    def __init__(self, classifier_path: str, low_threshold: float = 0.10):
        """
        :param classifier_path: 修复后的预训练分类器 (classifier_fixed.pkl) 的路径
        :param low_threshold: 低清判定阈值，预测高清概率低于此值将被判定为低清
        """
        self.classifier_path = classifier_path
        self.low_threshold = low_threshold
        self.model = None
        self.feature_keys = [
            'laplacian_var', 'gray_pixel_ratio', 'corner_density',
            'edge_integrity', 'true_edge_contrast', 'bg_artifacts',
            'edge_snr', 'gradient_coherence', 'skeleton_density',
            'hf_energy_ratio', 'image_entropy', 'intra_class_var', 'tenengrad_var'
        ]
        self._load_classifier()

    def _load_classifier(self):
        """加载 Pickle 格式的特征分类模型"""
        if not os.path.exists(self.classifier_path):
            raise FileNotFoundError(f"找不到分类器模型文件: {self.classifier_path}")

        # numpy 1.17+ 重构了随机数生成器，pickle 格式有破坏性变化：旧版本存下的
        # numpy.random._pickle.__generator_ctor 签名与新版不兼容，标准 pickle.load()
        # 会抛 TypeError。CompatUnpickler 在反序列化时拦截这些构造函数、跳过随机状态
        # 恢复——对已训练模型的 predict 无影响（random_state 只在 fit 阶段起作用）。
        try:
            with open(self.classifier_path, 'rb') as f:
                bundle = pickle.load(f)
        except (TypeError, ValueError) as e:
            if 'generator_ctor' in str(e) or 'state must be for' in str(e):
                print(f"[WARN] numpy.random 版本不兼容 "
                      f"(numpy {np.__version__}, scikit-learn {self._get_sklearn_version()})"
                      f"，改用 CompatUnpickler 加载")
                from utils.fix_classifier_pickle import CompatUnpickler
                with open(self.classifier_path, 'rb') as f:
                    bundle = CompatUnpickler(f).load()
            else:
                raise

        self.model = bundle['model']
        if 'feature_keys' in bundle:
            self.feature_keys = bundle['feature_keys']
        print(f"[OK] 成功加载分类器: {self.classifier_path} (特征数={len(self.feature_keys)})")

    @staticmethod
    def _get_sklearn_version():
        """获取 scikit-learn 版本"""
        try:
            import sklearn
            return sklearn.__version__
        except:
            return "unknown"

    @staticmethod
    def _sobel_mag(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算 Sobel 梯度幅值 (CPU)"""
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        return gx, gy, mag

    @staticmethod
    def _gaussian_blur(img: np.ndarray, ksize=(5, 5), sigma=0) -> np.ndarray:
        """计算高斯模糊 (CPU)"""
        return cv2.GaussianBlur(img, ksize, sigma).astype(np.float32)

    def extract_features(self, image_input: Union[str, Path, np.ndarray]) -> Optional[dict]:
        """
        提取单张图像的 13 维特征
        :param image_input: 图像路径 (str) 或 OpenCV 图像矩阵 (np.ndarray)
        :return: 包含特征的字典，提取失败返回 None
        """
        try:
            # 1. 图像加载与校验
            if isinstance(image_input, (str, Path)):
                img = cv2.imread(str(image_input), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"  [ERROR] 无法读取图像: {image_input}")
                    return None
            elif isinstance(image_input, np.ndarray):
                if image_input.ndim == 3 and image_input.shape[2] >= 3:
                    img = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
                else:
                    img = image_input.copy()
            else:
                raise TypeError("输入必须是图像路径 (str/Path) 或 numpy 数组 (np.ndarray)")

            # IMREAD_GRAYSCALE 对部分 PNG 仍返回 (H, W, 1)，下游按二维取 shape。
            if img.ndim == 3:
                img = img[:, :, 0]

            img_h, img_w = img.shape
            img_area = img_h * img_w
            feat = {}

            # ── 二值化 & 前景 ──────────────────────────────────────────
            thresh_val, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            fg_count = np.count_nonzero(thresh)

            # ── 梯度 (CPU) ─────────────────────────────────────────────
            gx, gy, mag = self._sobel_mag(img)

            # ── Canny 边缘 ─────────────────────────────────────────────
            edges = cv2.Canny(img, 50, 150)
            edge_px = np.count_nonzero(edges)

            # [A] 边缘完整性
            if edge_px > 0:
                num_labels, _, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
                avg_len = np.mean(stats[1:, cv2.CC_STAT_AREA]) if num_labels > 1 else 1.0
                feat['edge_integrity'] = 1.0 / (avg_len + 1e-5)
            else:
                feat['edge_integrity'] = 0.0

            # [B] 边缘梯度均值
            edge_mags = mag[edges > 0]
            feat['true_edge_contrast'] = float(np.mean(edge_mags)) if len(edge_mags) > 0 else 0.0

            # [C] 背景伪影
            kernel = np.ones((7, 7), np.uint8)
            dilated_fg = cv2.dilate(thresh, kernel, iterations=1)
            bg_mask = cv2.bitwise_not(dilated_fg)
            bg_mags = mag[bg_mask > 0]
            feat['bg_artifacts'] = float(np.std(bg_mags)) if len(bg_mags) > 0 else 0.0

            # [D] 边缘信噪比
            fg_dilated = cv2.dilate(thresh, np.ones((5, 5), np.uint8), iterations=2)
            non_edge_fg = (fg_dilated > 0) & (edges == 0)
            non_edge_mags = mag[non_edge_fg]
            non_edge_mean = float(np.mean(non_edge_mags)) if len(non_edge_mags) > 0 else 0.0
            edge_mean = float(np.mean(edge_mags)) if len(edge_mags) > 0 else 0.0
            feat['edge_snr'] = edge_mean / (non_edge_mean + 1e-3)

            # [E] 梯度方向一致性
            angle = np.arctan2(gy, gx) % np.pi
            sin2a = np.sin(2 * angle)
            cos2a = np.cos(2 * angle)
            edge_coords = np.argwhere(edges > 0)
            half_w = 2
            coh_vals = []
            for (r, c) in edge_coords[::3]:
                r0, r1 = max(0, r - half_w), min(img_h, r + half_w + 1)
                c0, c1 = max(0, c - half_w), min(img_w, c + half_w + 1)
                s = sin2a[r0:r1, c0:c1].ravel()
                cc = cos2a[r0:r1, c0:c1].ravel()
                coh_vals.append(np.sqrt(np.mean(s) ** 2 + np.mean(cc) ** 2))
            feat['gradient_coherence'] = float(np.mean(coh_vals)) if coh_vals else 0.0

            # [F] 骨架密度
            try:
                from cv2.ximgproc import thinning
                skeleton = thinning(thresh)
            except (ImportError, AttributeError):
                skeleton = thresh.copy()
                temp = np.zeros_like(thresh)
                el = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
                while True:
                    eroded = cv2.erode(skeleton, el)
                    opened = cv2.dilate(eroded, el)
                    subset = cv2.subtract(skeleton, opened)
                    temp = cv2.bitwise_or(temp, subset)
                    skeleton = eroded.copy()
                    if cv2.countNonZero(skeleton) == 0:
                        break
                skeleton = temp
            feat['skeleton_density'] = np.count_nonzero(skeleton) / (fg_count + 1e-5)

            # [G] 高频能量比 (CPU)
            blurred = self._gaussian_blur(img, (5, 5), 0)
            highfreq = np.abs(img.astype(np.float32) - blurred)
            fg_hf = highfreq[fg_dilated > 0]
            fg_orig = img.astype(np.float32)[fg_dilated > 0]
            total_energy = np.mean(fg_orig ** 2) + 1e-5
            feat['hf_energy_ratio'] = float(np.mean(fg_hf ** 2)) / total_energy

            # [H] 拉普拉斯方差 (模糊检测)
            feat['laplacian_var'] = cv2.Laplacian(img, cv2.CV_64F).var()

            # [I] 灰度过渡像素比例
            gray_pixels = np.count_nonzero((img > 15) & (img < 240))
            feat['gray_pixel_ratio'] = gray_pixels / (img_area + 1e-5)

            # [J] 角点密度
            corners = cv2.goodFeaturesToTrack(img, maxCorners=500, qualityLevel=0.01, minDistance=5)
            corner_count = len(corners) if corners is not None else 0
            feat['corner_density'] = corner_count / (fg_count + 1e-5)

            # [K] 图像信息熵
            hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
            hist_p = hist[hist > 0] / img_area
            feat['image_entropy'] = -np.sum(hist_p * np.log2(hist_p))

            # [L] Otsu 类内方差
            bg_pixels = img[img >= thresh_val]
            fg_pixels = img[img < thresh_val]
            var_bg = np.var(bg_pixels) if len(bg_pixels) > 0 else 0
            var_fg = np.var(fg_pixels) if len(fg_pixels) > 0 else 0
            feat['intra_class_var'] = (len(bg_pixels) * var_bg + len(fg_pixels) * var_fg) / img_area

            # [M] Tenengrad 梯度方差
            feat['tenengrad_var'] = np.var(mag ** 2)

            return feat

        except Exception as e:
            print(f"  [ERROR] 特征提取异常: {e}")
            return None

    def predict(self, image_input: Union[str, Path, np.ndarray]) -> dict:
        """
        判断图像是否需要处理（低于阈值判断为低清）。

        :param image_input: 图像路径 (str) 或 OpenCV 图像矩阵 (np.ndarray)
        :return: 包含分类结果和特征的字典
                 is_low_res: bool，True 表示低清（需要后续处理）
                 prob: float，模型判断为高清的原始概率值
                 features: dict，提取到的具体特征值
        """
        feat = self.extract_features(image_input)

        # 异常容错机制：如果无法提取特征，默认设为需要处理（防御性策略），你可以视业务需求修改
        if feat is None:
            return {
                'is_low_res': True,
                'prob': 0.0,
                'features': None,
                'error': 'Feature extraction failed'
            }

        # 构造输入张量
        x = np.array([[feat.get(k, 0.0) for k in self.feature_keys]], dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # 预测高清的概率
        prob = float(self.model.predict_proba(x)[0, 1])

        # 小于阈值记为低清
        is_low_res = prob < self.low_threshold

        return {
            'is_low_res': is_low_res,
            'prob': prob,
            'features': feat
        }


def classify_directory(
    dir_path: Union[str, Path] = CROP_DIR,
    classifier_path: Union[str, Path] = CLASSIFIER_PATH,
    low_threshold: float = 0.10,
) -> Dict[str, object]:
    """对目录下所有图片进行高/低清分类，逐文件打印并返回汇总。

    Returns:
        {'high': int, 'low': int, 'failed': int, 'details': [{name, prob, label}, ...]}
    """
    dir_path = Path(dir_path)
    if not dir_path.exists():
        print(f"[WARN] 目录不存在: {dir_path}")
        return {'high': 0, 'low': 0, 'failed': 0, 'details': []}

    files = sorted(p for p in dir_path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        print(f"[WARN] 目录为空（无图片）: {dir_path}")
        return {'high': 0, 'low': 0, 'failed': 0, 'details': []}

    classifier = ResolutionClassifier(
        classifier_path=str(classifier_path),
        low_threshold=low_threshold,
    )

    high = low = failed = 0
    details: List[dict] = []
    for f in files:
        result = classifier.predict(str(f))
        if result.get('features') is None:
            failed += 1
            label = "读取失败"
            print(f"[{f.name}] {label}")
            details.append({'name': f.name, 'prob': None, 'label': label})
            continue
        prob = result['prob']
        if result['is_low_res']:
            low += 1
            label = "低清"
        else:
            high += 1
            label = "高清"
        print(f"[{f.name}] prob={prob:.4f} → {label}")
        details.append({'name': f.name, 'prob': prob, 'label': label})

    print("=" * 30)
    print(f"高清图片: {high} 张")
    print(f"低清图片: {low} 张")
    if failed:
        print(f"读取失败: {failed} 张")

    return {'high': high, 'low': low, 'failed': failed, 'details': details}


# ================================================================
# 测试/调用示例代码 (可在此管线中直接实例化调用)
# ================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="对 crop_dataset/ 中的图片做高/低清分类")
    parser.add_argument(
        "dir",
        nargs="?",
        default=str(CROP_DIR),
        help="待分类图片目录，默认 crop_dataset/",
    )
    args = parser.parse_args()
    classify_directory(args.dir)
