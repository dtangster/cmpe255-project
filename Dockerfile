FROM python:3.7.3-slim-stretch

COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt
COPY . /opt/project/

ENTRYPOINT ["/opt/project/project.py"]
#ENTRYPOINT ["sleep", "infinity"]
