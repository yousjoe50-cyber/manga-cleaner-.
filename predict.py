from cog import BasePredictor, Input, Path
import os
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from simple_lama_inpainting import SimpleLama
import gc

class Predictor(BasePredictor):
    def setup(self) -> None:
        """تحميل الموديلات في الذاكرة لتسريع المعالجة"""
        print("⏳ Loading Models...")
        # تحميل موديل YOLO إذا لم يكن موجوداً
        if not os.path.exists('/tmp/manga_model.pt'):
            os.system("wget -q https://huggingface.co/ogkalu/comic-text-segmenter-yolov8m/resolve/main/comic-text-segmenter.pt -O /tmp/manga_model.pt")

        original_load = torch.load
        def safe_load(*args, **kwargs):
            kwargs['weights_only'] = False 
            return original_load(*args, **kwargs)
        torch.load = safe_load

        self.manga_model = YOLO('/tmp/manga_model.pt')
        self.lama = SimpleLama()
        print("✅ Models Loaded!")

    def predict(
        self,
        image: Path = Input(description="Input Manga Page Image")
    ) -> Path:
        """معالجة الصورة المرفوعة"""
        original_img_pil = Image.open(str(image)).convert("RGB")
        W, H = original_img_pil.size
        num_slices = max(1, H // 2000)
        slice_h = H // num_slices
        overlap = 150 
        
        cv_original = cv2.cvtColor(np.array(original_img_pil), cv2.COLOR_RGB2BGR)
        cleaned_slices_data = []
        
        for i in range(num_slices):
            start_y = max(0, i * slice_h - (overlap if i > 0 else 0))
            end_y = min(H, (i + 1) * slice_h + (overlap if i < num_slices - 1 else 0))
            slice_cv_img = cv_original[start_y:end_y, 0:W]
            
            results = self.manga_model.predict(source=slice_cv_img, imgsz=1024, conf=0.15, verbose=False)
            bubble_count = len(results[0].boxes) if hasattr(results[0], 'boxes') and results[0].boxes is not None else 0

            if bubble_count > 0:
                mask_np = np.zeros((slice_cv_img.shape[0], slice_cv_img.shape[1]), dtype=np.uint8)
                
                if hasattr(results[0], 'masks') and results[0].masks is not None:
                    for m in results[0].masks.xy:
                        if len(m) > 0:
                            contour = np.array(m, dtype=np.int32)
                            cv2.drawContours(mask_np, [contour], -1, 255, -1)
                            
                if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                    for box in results[0].boxes.xyxy:
                        if len(box) >= 4:
                            x1, y1, x2, y2 = map(int, box[:4])
                            cv2.rectangle(mask_np, (x1, y1), (x2, y2), 255, -1)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                dilated_mask_np = cv2.dilate(mask_np, kernel, iterations=4)
                
                slice_pil = Image.fromarray(cv2.cvtColor(slice_cv_img, cv2.COLOR_BGR2RGB))
                mask_pil = Image.fromarray(dilated_mask_np).convert("L")
                
                cleaned_pil = self.lama(slice_pil, mask_pil)
                cleaned_slice_cv = cv2.cvtColor(np.array(cleaned_pil), cv2.COLOR_RGB2BGR)
            else:
                cleaned_slice_cv = slice_cv_img

            cleaned_slices_data.append((cleaned_slice_cv, start_y, end_y))

        final_img = Image.new("RGB", (W, H))
        for i, (cleaned_cv_slice, start_y, end_y) in enumerate(cleaned_slices_data):
            cleaned_pil_slice = Image.fromarray(cv2.cvtColor(cleaned_cv_slice, cv2.COLOR_BGR2RGB))
            if i == 0:
                final_img.paste(cleaned_pil_slice.crop((0, 0, W, cleaned_pil_slice.height - overlap // 2)), (0, 0))
            elif i == num_slices - 1:
                final_img.paste(cleaned_pil_slice.crop((0, overlap // 2, W, cleaned_pil_slice.height)), (0, start_y + overlap // 2))
            else:
                final_img.paste(cleaned_pil_slice.crop((0, overlap // 2, W, cleaned_pil_slice.height - overlap // 2)), (0, start_y + overlap // 2))
                
        output_path = "/tmp/cleaned_output.jpg"
        final_img.save(output_path, format='JPEG', quality=95, subsampling=0)
        
        del original_img_pil, cv_original, final_img
        gc.collect()

        return Path(output_path)

