"""OpenCV와 PyAV의 macOS 네이티브 라이브러리 충돌을 피하기 위한 OCR 프로세스."""
import json
import sys

from rapidocr import RapidOCR, ModelType, OCRVersion


def extract(paths, model_dir):
    reader = RapidOCR(params={"Global.model_root_dir": model_dir, "Global.log_level": "error",
                              "Det.ocr_version": OCRVersion.PPOCRV5, "Det.model_type": ModelType.MOBILE,
                              "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.model_type": ModelType.MOBILE,
                              "Rec.lang_type": "korean"})
    return [line for path in paths for line in (reader(path).txts or [])]


if __name__ == "__main__":
    print(json.dumps({"lines": extract(sys.argv[2:], sys.argv[1])}, ensure_ascii=False))
