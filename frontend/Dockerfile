# 第一阶段：构建。用 node 镜像把 Vue 项目编译成静态文件（HTML/JS/CSS）
FROM node:20-alpine AS build

WORKDIR /app

# 同样是先拷贝依赖清单再装依赖，利用构建缓存
COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

# 第二阶段：运行。用极轻量的 nginx 镜像只提供构建产物，
# 不需要 node 运行时了，最终镜像体积比"直接用 node 跑"小得多
FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
