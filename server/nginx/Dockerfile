FROM nginx:alpine3.18

COPY /conf/default.conf /etc/nginx/conf.d/
COPY /certs/nginx.crt /certs/nginx.key /etc/nginx/ssl/
COPY index.html /usr/share/nginx/html
COPY imgs/ data/images

RUN chmod 644 -R /etc/nginx/ssl/nginx.key && \
chown -R nginx:nginx /var/cache/nginx &&  \
touch /var/run/nginx.pid && chown -R nginx:nginx  /var/run/nginx.pid

USER nginx

CMD ["nginx","-g","daemon off;"]