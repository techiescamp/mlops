# Stage 1: Build
FROM public.ecr.aws/docker/library/node:22-bookworm-slim AS build

# Create app directory and set permissions
RUN mkdir -p /home/node/app/node_modules && chown -R node:node /home/node/app
WORKDIR /home/node/app

# Copy package.json and install dependencies
COPY ["package.json", "package-lock.json", "/home/node/app/"]
RUN chown node:node /home/node/app/package.json /home/node/app/package-lock.json
USER node
RUN npm install --ignore-scripts

# Copy application and public directory
COPY ["server.js", "/home/node/app"]
# Ensure /home/node/app/uploads exists and has proper permissions in the build stage
RUN mkdir -p /home/node/app/uploads && chown -R node:node /home/node/app/uploads

# Stage 2: Production
FROM gcr.io/distroless/nodejs22-debian12:nonroot

# Copy built application from the build stage
COPY --from=build /home/node/app /usr/src/app


USER nonroot
WORKDIR /usr/src/app

# Command to run the application
CMD ["server.js"]