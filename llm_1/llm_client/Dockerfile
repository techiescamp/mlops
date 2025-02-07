# Use the official Node.js image as a base image
FROM node:20-alpine

# Set the working directory inside the container
WORKDIR /app

# Copy package.json and package-lock.json to install dependencies first
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the React app into the container
COPY . .

# Build the React app for production
RUN npm run build

# Install a simple HTTP server to serve the built app
RUN npm install -g serve

# Expose port 3000 for the app
EXPOSE 3000

# Serve the built app on port 3000
CMD ["serve", "-s", "build", "-l", "3000"]