import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pycocotools.coco import COCO


def generate_white_noise_image(height, width):
    return np.random.rand(height, width, 3) * 255


def apply_segmentation_mask(image, annotations):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for ann in annotations:
        for seg in ann['segmentation']:
            poly = np.array(seg).reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [poly], 255)

    white_noise = generate_white_noise_image(image.shape[0], image.shape[1]).astype(np.uint8)
    foreground = cv2.bitwise_and(image, image, mask=mask)
    inverse_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(white_noise, white_noise, mask=inverse_mask)
    combined_image = cv2.add(foreground, background)
    return combined_image


def save_image(output_dir, category_name, image_id, image):
    category_dir = os.path.join(output_dir, category_name)
    if not os.path.exists(category_dir):
        os.makedirs(category_dir)

    image_path = os.path.join(category_dir, f'img_{image_id}.jpg')
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, image_bgr)


dataDir = './chicken/train/'
dataType = 'train'
annFile = f'{dataDir}_annotations.coco.json'
coco = COCO(annFile)

output_dir = './chicken/output_dir/'
imgIds = coco.getImgIds()

for image_id in imgIds:
    image_info = coco.loadImgs(image_id)[0]
    image_path = dataDir + image_info['file_name']
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    annotation_ids = coco.getAnnIds(imgIds=image_id)
    annotations = coco.loadAnns(annotation_ids)

    for ann in annotations:
        cat_id = ann['category_id']
        cat_name = coco.loadCats(cat_id)[0]['name']
        result_image = apply_segmentation_mask(image, [ann])
        save_image(output_dir, cat_name, image_id, result_image)

    print(f'Processed and saved images for image ID: {image_id}')