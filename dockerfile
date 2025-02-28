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
SHELL ["conda", "run", "-n", "MTL-pep-prop", "/bin/bash", "-c"]

# Copy the entire project into the container
COPY . .

# Set the default command (this will run in the conda environment)
CMD ["python", "main.py"]  # Replace with the appropriate command for your project
