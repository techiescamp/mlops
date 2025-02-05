# Stage 1: Build Stage
FROM node:18 AS build

# Set the working directory in the container
WORKDIR /app

# Copy package.json and package-lock.json
COPY package.json package-lock.json ./

# Install dependencies
RUN npm install

# Copy the entire project files to the working directory
COPY . .

# Build the React app
RUN npm run build

# Stage 2: Production Stage
FROM nginx:alpine


# Copy the built React app files from the build stage to the Nginx public folder
COPY --from=build /app/build /usr/share/nginx/html

# Remove default configuration
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom NGINX configuration (optional, if you need custom settings)
COPY nginx.conf /etc/nginx/conf.d/

# Expose the port Nginx will listen on
EXPOSE 80

# Start Nginx when the container runs
CMD ["nginx", "-g", "daemon off;"]