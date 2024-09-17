import os

import numpy as np
import torch
import torchvision.transforms.v2 as T
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw

from utils.imgs import square_bbox
from utils.paths import ROOT_DIR


class NoFilterCriteriaError(Exception):
    def __init__(self, message="No images passed the filter criteria."):
        self.message = message
        super().__init__(self.message)


INSIGHTFACE_DIR = os.path.join(ROOT_DIR, "models", "insightface")

THRESHOLDS = {  # from DeepFace
    "VGG-Face": {"cosine": 0.68, "euclidean": 1.17, "L2_norm": 1.17},
    "Facenet": {"cosine": 0.40, "euclidean": 10, "L2_norm": 0.80},
    "Facenet512": {"cosine": 0.30, "euclidean": 23.56, "L2_norm": 1.04},
    "ArcFace": {"cosine": 0.68, "euclidean": 4.15, "L2_norm": 1.13},
    "Dlib": {"cosine": 0.07, "euclidean": 0.6, "L2_norm": 0.4},
    "SFace": {"cosine": 0.593, "euclidean": 10.734, "L2_norm": 1.055},
    "OpenFace": {"cosine": 0.10, "euclidean": 0.55, "L2_norm": 0.55},
    "DeepFace": {"cosine": 0.23, "euclidean": 64, "L2_norm": 0.64},
    "DeepID": {"cosine": 0.015, "euclidean": 45, "L2_norm": 0.17},
    "GhostFaceNet": {"cosine": 0.65, "euclidean": 35.71, "L2_norm": 1.10},
}


class InsightFace:
    def __init__(self, provider="CPU", name="buffalo_l"):
        self.face_analysis = FaceAnalysis(
            name=name,
            providers=[
                provider + "ExecutionProvider",
            ],
        )
        self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))
        self.thresholds = THRESHOLDS["ArcFace"]

    def get_face(self, image):
        for size in [(size, size) for size in range(640, 256, -64)]:
            self.face_analysis.det_model.input_size = size
            faces = self.face_analysis.get(image)
            if len(faces) > 0:
                return sorted(
                    faces,
                    key=lambda x: (x["bbox"][2] - x["bbox"][0])
                    * (x["bbox"][3] - x["bbox"][1]),
                    reverse=False,
                )
        return None

    def get_embeds(self, image):
        face = self.get_face(image)
        if face is not None:
            face = face[0].normed_embedding
        return face

    def get_bbox(self, image, padding=0, padding_percent=0, force_square=False):
        faces = self.get_face(np.array(image))
        if faces is None:
            faces = []
        img = []
        x = []
        y = []
        w = []
        h = []
        bbox = []
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            width = x2 - x1
            height = y2 - y1
            x1 = int(max(0, x1 - int(width * padding_percent) - padding))
            y1 = int(max(0, y1 - int(height * padding_percent) - padding))
            x2 = int(min(image.width, x2 + int(width * padding_percent) + padding))
            y2 = int(min(image.height, y2 + int(height * padding_percent) + padding))
            if force_square:
                x1, y1, x2, y2 = square_bbox(
                    [x1, y1, x2, y2], (image.height, image.width)
                )
            crop = image.crop((x1, y1, x2, y2))
            img.append(T.ToTensor()(crop).permute(1, 2, 0).unsqueeze(0))
            x.append(x1)
            y.append(y1)
            w.append(x2 - x1)
            h.append(y2 - y1)
            bbox.append([x1, y1, x2, y2])
        return (img, x, y, w, h, bbox)

    def get_keypoints(self, image):
        face = self.get_face(image)
        if face is not None:
            shape = face[0]["kps"]
            right_eye = shape[0]
            left_eye = shape[1]
            nose = shape[2]
            left_mouth = shape[3]
            right_mouth = shape[4]

            return [left_eye, right_eye, nose, left_mouth, right_mouth]
        return None

    def get_landmarks(self, image, extended_landmarks=False):
        face = self.get_face(image)
        if face is not None:
            shape = face[0]["landmark_2d_106"]
            landmarks = np.round(shape).astype(np.int64)

            main_features = landmarks[33:]
            left_eye = landmarks[87:97]
            right_eye = landmarks[33:43]
            eyes = landmarks[[*range(33, 43), *range(87, 97)]]
            nose = landmarks[72:87]
            mouth = landmarks[52:72]
            left_brow = landmarks[97:106]
            right_brow = landmarks[43:52]
            outline = landmarks[[*range(33), *range(48, 51), *range(102, 105)]]
            outline_forehead = outline

            return [
                landmarks,
                main_features,
                eyes,
                left_eye,
                right_eye,
                nose,
                mouth,
                left_brow,
                right_brow,
                outline,
                outline_forehead,
            ]
        return None


class FaceEmbedDistance:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "analysis_models": ("ANALYSIS_MODELS",),
                "reference": ("IMAGE",),
                "image": ("IMAGE",),
                "similarity_metric": (["L2_norm", "cosine", "euclidean"],),
                "filter_thresh": (
                    "FLOAT",
                    {"default": 100.0, "min": 0.001, "max": 100.0, "step": 0.001},
                ),
                "filter_best": (
                    "INT",
                    {"default": 0, "min": 0, "max": 4096, "step": 1},
                ),
                "generate_image_overlay": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("IMAGE", "distance")
    FUNCTION = "analize"
    CATEGORY = "FaceAnalysis"

    def analize(
        self,
        analysis_models,
        reference,
        image,
        similarity_metric,
        filter_thresh,
        filter_best,
    ):

        if filter_thresh == 0.0:
            filter_thresh = analysis_models.thresholds[similarity_metric]

        # you can send multiple reference images in which case the embeddings are averaged
        ref = []
        for i in reference:
            ref_emb = analysis_models.get_embeds(
                np.array(T.ToPILImage()(i.permute(2, 0, 1)).convert("RGB"))
            )
            if ref_emb is not None:
                ref.append(torch.from_numpy(ref_emb))

        if ref == []:
            raise Exception("No face detected in reference image")

        ref = torch.stack(ref)
        ref = np.array(torch.mean(ref, dim=0))

        out = []
        out_dist = []

        for i in image:
            img = np.array(T.ToPILImage()(i.permute(2, 0, 1)).convert("RGB"))

            img = analysis_models.get_embeds(img)

            if img is None:  # No face detected
                dist = 100.0
                norm_dist = 0
            else:
                if np.array_equal(ref, img):  # Same face
                    dist = 0.0
                    norm_dist = 0.0
                else:
                    if similarity_metric == "L2_norm":
                        # dist = euclidean_distance(ref, img, True)
                        ref = ref / np.linalg.norm(ref)
                        img = img / np.linalg.norm(img)
                        dist = np.float64(np.linalg.norm(ref - img))
                    elif similarity_metric == "cosine":
                        dist = np.float64(
                            1
                            - np.dot(ref, img)
                            / (np.linalg.norm(ref) * np.linalg.norm(img))
                        )
                        # dist = cos_distance(ref, img)
                    else:
                        # dist = euclidean_distance(ref, img)
                        dist = np.float64(np.linalg.norm(ref - img))

                    norm_dist = min(
                        1.0, 1 / analysis_models.thresholds[similarity_metric] * dist
                    )

            if dist <= filter_thresh:
                print(
                    f"\033[96mFace Analysis: value: {dist}, normalized: {norm_dist}\033[0m"
                )

                out.append(i)

                out_dist.append(dist)

        if not out:
            raise NoFilterCriteriaError("No image matches the filter criteria.")

        out = torch.stack(out)

        # filter out the best matches
        if filter_best > 0:
            filter_best = min(filter_best, len(out))
            out_dist, idx = torch.topk(
                torch.tensor(out_dist), filter_best, largest=False
            )
            out_dist = out_dist.cpu().numpy().tolist()

        return out_dist
