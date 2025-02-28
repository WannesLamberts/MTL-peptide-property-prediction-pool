# Use Miniconda as base image
FROM continuumio/miniconda3

# Set working directory inside the container
WORKDIR /app

# Copy your environment file
COPY environment.yml .

# Create conda environment and clean up
RUN conda env create -f environment.yml && \
    conda clean --all -y

# Set shell to use conda environment
SHELL ["conda", "run", "-n", "my_env", "/bin/bash", "-c"]

# Copy the entire project into the container
COPY . .

# Set the default command (modify for your project)dd
CMD ["conda", "activate","MTL-pep-prop"]