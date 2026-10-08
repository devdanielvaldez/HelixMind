FROM node:20-alpine AS dependencies
WORKDIR /app
COPY package*.json ./
RUN npm install
FROM node:20-alpine AS build
WORKDIR /app
COPY --from=dependencies /app/node_modules ./node_modules
COPY . .
RUN npm run build
FROM node:20-alpine AS runtime
WORKDIR /app
ENV NODE_ENV=production
COPY --from=build /app/.next ./.next
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/package.json ./package.json
COPY --from=build /app/next.config.mjs ./next.config.mjs
RUN addgroup -S -g 10001 medai && adduser -S -u 10001 -G medai medai
USER 10001:10001
EXPOSE 3000
CMD ["npm","run","start"]
