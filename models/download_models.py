import os
import urllib.request
import sys

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

PROTOTXT_URLS = [
    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/voc/MobileNetSSD_deploy.prototxt",
    "https://raw.githubusercontent.com/PINTO0309/MobileNet-SSD-RealSense/master/caffemodel/MobileNetSSD/MobileNetSSD_deploy.prototxt",
]

CAFFEMODEL_URLS = [
    "https://github.com/PINTO0309/MobileNet-SSD-RealSense/raw/master/caffemodel/MobileNetSSD/MobileNetSSD_deploy.caffemodel",
    "https://raw.githubusercontent.com/PINTO0309/MobileNet-SSD-RealSense/master/caffemodel/MobileNetSSD/MobileNetSSD_deploy.caffemodel",
    "https://github.com/yeephycho/tensorflow-face-detection/raw/master/res10_300x300_ssd_iter_140000.caffemodel"
]

PROTOTXT_PATH = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.prototxt")
CAFFEMODEL_PATH = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.caffemodel")

# 20 PASCAL VOC / COCO classes supported by MobileNet-SSD
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]

def download_with_fallback(urls, destination):
    if os.path.exists(destination) and os.path.getsize(destination) > 10000:
        print(f"File already exists: {destination} ({os.path.getsize(destination)/1024:.1f} KB)")
        return True

    for url in urls:
        print(f"Attempting download from: {url}")
        try:
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response, open(destination, 'wb') as out_file:
                total_size = int(response.info().get('Content-Length', 0))
                block_size = 1024 * 16
                downloaded = 0
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    if total_size > 0:
                        percent = min(100.0, downloaded * 100 / total_size)
                        sys.stdout.write(f"\rDownloading: {percent:.1f}% ({downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB)")
                        sys.stdout.flush()
            print(f"\nSuccessfully downloaded to {destination}")
            return True
        except Exception as e:
            print(f"\nDownload failed from {url}: {e}")
            if os.path.exists(destination):
                try:
                    os.remove(destination)
                except Exception:
                    pass
    return False

def ensure_models_downloaded():
    """Ensure MobileNet-SSD model weights and prototxt files exist."""
    print("=== Model File Verification ===")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    proto_ok = download_with_fallback(PROTOTXT_URLS, PROTOTXT_PATH)
    caffe_ok = download_with_fallback(CAFFEMODEL_URLS, CAFFEMODEL_PATH)
    
    if proto_ok and caffe_ok:
        print("MobileNet-SSD models are ready.")
        return True
    else:
        print("MobileNet-SSD Caffe model not fully downloaded. Backup detector will be used.")
        return False

if __name__ == "__main__":
    ensure_models_downloaded()
