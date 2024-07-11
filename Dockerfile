FROM python:3.9
ADD . /code
WORKDIR /code
RUN bash get_data.sh
RUN pip install -r requirements.txt
CMD python run.py --trace dataset/50_1.json --placers CloseRadiusFit FirstFit RandomFit CloseRadiusLB --p=0.95
