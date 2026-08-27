# 用 slim 版本减小镜像体积，3.11 和本地开发环境版本保持一致，避免"本地能跑、容器里不行"的问题
FROM python:3.11-slim

WORKDIR /app

# 先只拷贝 requirements.txt 再装依赖，是 Docker 构建缓存的标准技巧：
# 只要 requirements.txt 没变，重新 build 时这一层会命中缓存，不用每次都重新下载几百MB的 torch，
# 只有真正改了依赖时才会触发重新安装，大幅加快日常开发时的重复构建速度
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 数据目录（向量库、上传文档、会话数据库、Agent 生成文件）会通过 docker-compose 里的 volume 挂载，
# 这样容器重启、重新构建时数据不会丢失
RUN mkdir -p /app/data

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
