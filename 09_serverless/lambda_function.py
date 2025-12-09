from io import BytesIO
from urllib import request

import onnxruntime as ort

from torchvision import transforms
from PIL import Image


MODEL_PATH = "hair_classifier_empty.onnx"

model = ort.InferenceSession(MODEL_PATH)
input_name = model.get_inputs()[0].name


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict(url):
    img = download_image(url)
    img_resized = prepare_image(img, (200, 200))
    img_transformed = transform(img_resized)
    
    X = img_transformed.numpy()[None, :, :, :]

    output = model.run(None, {input_name: X})[0]
    return float(output[0][0])


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result
