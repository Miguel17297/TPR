FROM nginx:alpine3.18@sha256:b1cfc4e0e01b4dceca3265fd4ca97921569fca1a10919639bedfa8dad9127027

COPY /conf/default.conf /etc/nginx/conf.d/
COPY /certs/server_cert.pem /certs/server_key.pem /etc/nginx/ssl/
COPY index.html /usr/share/nginx/html
COPY imgs/ data/images

RUN chmod 644 -R /etc/nginx/ssl/server_key.pem && \
chown -R nginx:nginx /var/cache/nginx &&  \
touch /var/run/nginx.pid && chown -R nginx:nginx  /var/run/nginx.pid

USER nginx

CMD ["nginx","-g","daemon off;"]