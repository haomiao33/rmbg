# FROM python:3.10-slim

# WORKDIR /
# COPY requirements.txt /requirements.txt
# RUN pip install -r requirements.txt
# COPY rp_handler.py /

# # Start the container
# CMD ["python3", "-u", "rp_handler.py"]

FROM runpod/base:0.4.0-cuda12.1.1

# 设置工作目录
WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制源代码
COPY app.py .

# 运行入口
CMD ["python", "app.py"]
