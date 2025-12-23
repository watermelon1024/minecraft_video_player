import json
import os
import shutil
import threading
import zipfile
from enum import Enum
from typing import Optional

import cv2


class PackMode(Enum):
    ZIP = 1
    FOLDER = 2


class PackGenerator:
    def __init__(self, output_path: str, template_path: Optional[str] = None, mode: PackMode = PackMode.ZIP):
        """
        :param output_path: Destination path (zip filename or folder path).
        :param template_path: Path to static template folder to include.
        """
        self.output_path = output_path
        self.template_path = template_path
        self._mode = mode
        self.zf = None
        # Lock for thread-safe zip writing
        self._lock = threading.Lock()

    def __enter__(self):
        if self._mode is PackMode.ZIP:
            # Use DEFLATED for general files; images will override this to STORED later.
            self.zf = zipfile.ZipFile(self.output_path, "w", zipfile.ZIP_DEFLATED)

            if self.template_path and os.path.exists(self.template_path):
                print(f"[PackGen] Writing template (Zip): {self.template_path}")
                for root, dirs, files in os.walk(self.template_path):
                    for file in files:
                        abs_path = os.path.join(root, file)
                        rel_path = os.path.relpath(abs_path, self.template_path)
                        rel_path = rel_path.replace(os.sep, "/")
                        self.zf.write(abs_path, rel_path)

        else:
            # Folder mode: Copy template if exists, else create empty dir.
            if self.template_path and os.path.exists(self.template_path):
                print(f"[PackGen] Copying template (Folder): {self.template_path}")
                if os.path.exists(self.output_path):
                    shutil.rmtree(self.output_path)
                shutil.copytree(self.template_path, self.output_path)
            else:
                os.makedirs(self.output_path, exist_ok=True)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.zf:
            with self._lock:
                self.zf.close()
            print(f"[PackGen] Closed Zip stream: {self.output_path}")

    def write_image(self, rel_path, image_data):
        """
        Encodes and writes an image (Thread-Safe).
        :param rel_path: Relative path in pack (e.g., assets/minecraft/textures/font/x.png).
        :param image_data: BGR numpy array.
        """
        rel_path = rel_path.replace("\\", "/")

        # 1. Encode (CPU intensive, parallelizable)
        success, encoded_img = cv2.imencode(".png", image_data)
        if not success:
            print(f"[Error] Encoding failed: {rel_path}")
            return
        bytes_data = encoded_img.tobytes()

        # 2. Write (IO / Critical Section)
        if self._mode is PackMode.ZIP:
            # Use STORED for images since PNG is already compressed.
            # This avoids double compression overhead.
            assert self.zf is not None, "Zip file not initialized"
            with self._lock:
                self.zf.writestr(rel_path, bytes_data, compress_type=zipfile.ZIP_STORED)
        else:
            # OS handles concurrent file creation; makedirs might race but exist_ok handles it.
            full_path = os.path.join(self.output_path, rel_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            with open(full_path, "wb") as f:
                f.write(bytes_data)

    def write_text(self, rel_path, content):
        """Thread-safe text write."""
        rel_path = rel_path.replace("\\", "/")

        if self._mode is PackMode.ZIP:
            assert self.zf is not None, "Zip file not initialized"
            with self._lock:
                self.zf.writestr(rel_path, content)
        else:
            full_path = os.path.join(self.output_path, rel_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)

    def write_json(self, rel_path, data_dict, **kwargs):
        json_str = json.dumps(data_dict, separators=(",", ":"), ensure_ascii=False, **kwargs)
        self.write_text(rel_path, json_str)

    def write_file_from_disk(self, rel_path, source_path):
        """Thread-safe file copy from disk."""
        rel_path = rel_path.replace("\\", "/")

        if self._mode is PackMode.ZIP:
            assert self.zf is not None, "Zip file not initialized"
            with self._lock:
                self.zf.write(source_path, rel_path)
        else:
            full_path = os.path.join(self.output_path, rel_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            shutil.copy2(source_path, full_path)
