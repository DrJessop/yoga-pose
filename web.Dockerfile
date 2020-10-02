# pull official base image
FROM node:13.12.0-alpine

# add `/app/node_modules/.bin` to $PATH
ENV PATH app/frontend/node_modules/.bin:$PATH

WORKDIR /app

COPY app/frontend/ /app/

# install app dependencies
RUN npm install --silent
RUN npm install react-scripts@3.4.1 -g --silent

CMD ["npm", "start"]
