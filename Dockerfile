FROM carlduke/eidos-base:latest
RUN pip3 install gdown albumentations torchstain sklearn tqdm
WORKDIR /app
COPY train.py /app/train.py
COPY utils.py /app/utils.py
COPY unitopatho.py /app/unitopatho.py
COPY data /app/data
ENTRYPOINT ["/app/train.py"]