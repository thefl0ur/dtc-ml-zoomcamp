FROM agrigorev/model-2025-hairstyle:v1

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install onnxruntime pillow

COPY lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]