import torch
from torchvision import transforms
import numpy as np
from facenet_pytorch import fixed_image_standardization, MTCNN
from PIL import Image, ImageFont, ImageDraw
import os
import cv2


ABS_PATH = os.path.dirname(__file__)
#print(ABS_PATH, end='\n\n')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(keep_all=True, min_face_size=70, device=device)

PROTOTXT_PATH = os.path.join(ABS_PATH + '/caffe_model_data/deploy.prototxt')
CAFFEMODEL_PATH = os.path.join(ABS_PATH + '/caffe_model_data/weights.caffemodel')

caffe_model = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)

caffe_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
caffe_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

def diag(x1, y1, x2, y2):
    return np.linalg.norm([x2 - x1, y2 - y1])


def square(x1, y1, x2, y2):
    return abs(x2 - x1) * abs(y2 - y1)


def isOverlap(rect1, rect2):
    x1, x2 = rect1[0], rect1[2]
    y1, y2 = rect1[1], rect1[3]

    x1_, x2_ = rect2[0], rect2[2]
    y1_, y2_ = rect2[1], rect2[3]

    if x1 > x2_ or x2 < x1_: return False
    if y1 > y2_ or y2 < y1_: return False

    rght, lft = x1 < x1_ < x2, x1_ < x1 < x2_
    d1, d2 = 0, diag(x1_, y1_, x2_, y2_)
    threshold = 0.5

    if rght and y1 < y1_: d1 = diag(x1_, y1_, x2, y2)
    elif rght and y1 > y1_: d1 = diag(x1_, y2_, x2, y1)
    elif lft and y1 < y1_: d1 = diag(x2_, y1_, x1, y2)
    elif lft and y1 > y1_: d1 = diag(x2_, y2_, x1, y1)

    if d1 / d2 >= threshold and square(x1, y1, x2, y2) < square(x1_, y1_, x2_, y2_): return True
    return False

def draw_box(draw, boxes, names, probs, min_p=0.89):
    font = ImageFont.truetype(os.path.join(ABS_PATH, 'arial.ttf'), size=28)

    not_overlap_inds = []
    for i in range(len(boxes)):
        not_overlap = True
        for box2 in boxes:
            if np.all(boxes[i] == box2): continue
            not_overlap = not isOverlap(boxes[i], box2)
            if not not_overlap: break
        if not_overlap: not_overlap_inds.append(i)

    boxes = [boxes[i] for i in not_overlap_inds]
    probs = [probs[i] for i in not_overlap_inds]
    for box, name, prob in zip(boxes, names, probs):
        if prob >= min_p:
            draw.rectangle(box.tolist(), outline=(255, 255, 255), width=5)
            x1, y1, _, _ = box
            text_width, text_height = font.getbbox(f'{name}')[2:]#font.getsize(f'{name}')
            draw.rectangle(((x1, y1 - text_height), (x1 + text_width, y1)), fill='white')
            draw.text((x1, y1 - text_height), f'{name}: {prob:.2f}', (24, 12, 30), font)

    return boxes, probs

#

standard_transform = transforms.Compose([
                                transforms.Resize((256, 256)),#transforms.Resize((160, 160)),
                                np.float32,
                                transforms.ToTensor(),
                                fixed_image_standardization
])

def get_video_embedding(model, x):
    embeds = model(x.to(device))
    return embeds.detach().cpu().numpy()

def face_extract(model, clf, expressions, frame, boxes):
    names, prob = [], []
    if len(boxes):
        x = torch.stack([standard_transform(frame.crop(b)) for b in boxes])
        embeds = get_video_embedding(model, x)
        idx, prob = clf.predict(embeds), clf.predict_proba(embeds).max(axis=1)
        #names = [IDX_TO_CLASS[idx_] for idx_ in idx]
        names = [expressions[idx_] for idx_ in idx]
    return names, prob

def preprocess_image(detector, face_extractor, clf, expressions, path, transform=None):
    if not transform: transform = lambda x: x.resize((1280, 1280)) if (np.array(x.size) > 2000).all() else x
    capture = Image.open(path).convert('RGB')
    #capture = cv2.imread(path) doesn't work here
    # capture = Image.fromarray(path).convert('RGB')
    #i = 0

    #capture_rgb = cv2.cvtColor(capture, cv2.COLOR_BGR2RGB)

    # iframe = Image.fromarray(transform(np.array(capture)))
    iframe = transform(capture)

    #capture_pil = Image.fromarray(capture_rgb)
    #iframe = transform(capture_pil)#.convert('RGB'))
    #iframe = transform(Image.fromarray(capture_rgb))
    
    if detector == "MTCNN":
        #boxes, probs = detector.detect(iframe)
        boxes, probs = mtcnn.detect(iframe)
        if boxes is None: boxes, probs = [], []
    else:
        opencv_capture = np.array(capture)
        opencv_capture = opencv_capture[:, :, ::-1].copy() # Converting from rgb to bgr
        (h, w) = opencv_capture.shape[:2]
        #(h, w) = capture.shape[:2]

        capture_blob = cv2.dnn.blobFromImage(opencv_capture)
        #capture_blob = cv2.dnn.blobFromImage(np.array(capture))

        #detector.setInput(capture_blob)
        #detections = detector.forward()
        caffe_model.setInput(capture_blob)
        detections = caffe_model.forward()

        boxes, probs = [], []

        for i in range(0, detections.shape[2]):
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                boxes.append(box)
                probs.append(confidence)

    names, prob = face_extract(face_extractor, clf, expressions, iframe, boxes)

    frame_draw = iframe.copy()
    draw = ImageDraw.Draw(frame_draw)

    boxes, probs = draw_box(draw, boxes, names, probs)
    #return frame_draw.resize((620, 480), Image.BILINEAR)
    return frame_draw#.resize((1280, 720), Image.BILINEAR)
    #return frame_draw.resize((256, 256), Image.BILINEAR)


def preprocess_video(detector, face_extractor, clf, path, transform=None, k=3):
    frames = []
    if not transform: transform = lambda x: x.resize((1280, 1024)) if (np.array(x.shape) > 2000).all() else x
    capture = cv2.VideoCapture(path)
    i = 0
    while True:
        ret, frame = capture.read()
        if not ret: break

        iframe = Image.fromarray(transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        if (i + 1) % k:
            boxes, probs = detector.detect(iframe)
            if boxes is None: boxes, probs = [], []
            names, prob = face_extract(face_extractor, clf, iframe, boxes)

        frame_draw = iframe.copy()
        draw = ImageDraw.Draw(frame_draw)

        boxes, probs = draw_box(draw, boxes, names, probs)
        frames.append(frame_draw.resize((620, 480), Image.BILINEAR))
        i += 1

    print(f'Total frames: {i}')
    return frames

def framesToGif(frames, path):
    with imageio.get_writer(path, mode='I') as writer:
        for frame in tqdm.tqdm(frames):
            writer.append_data(np.array(frame))

#

