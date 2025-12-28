# -*- coding: utf-8 -*-
# modal_train.py

import modal
import os
import subprocess # If you need to run commands like git clone inside the container
import shutil # If you need to copy files inside the container

# --- Define Modal App and Image ---
stub = modal.Stub("audio-cnn-trainer")

# Define the Modal image, installing dependencies
# Ensure your requirements.txt includes torch, torchvision, torchaudio, librosa, pandas, scikit-learn, etc.
image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")

# Mount the volume containing your dataset
volume = modal.Volume.from_name("esc50-data", create_if_missing=True) # Use the name you created/uploaded to

# Optional: Mount a volume to save outputs/checkpoints/models
output_volume = modal.Volume.from_name("audio-cnn-outputs", create_if_missing=True)

@stub.function(
    image=image,
    gpu="any", # Or specify a specific GPU type like "A10G", "H100", etc., based on your needs and availability
    volumes={"/data": volume, "/outputs": output_volume}, # Mount volumes
    timeout=60 * 60 * 3, # Timeout for the entire training run (e.g., 3 hours), adjust as needed
    # cpu=4, # Specify CPU cores if needed
    # memory=32768, # Specify memory in MB if needed
)
def run_training(config_path: str):
    """
    Modal function to run the training script.

    Args:
        config_path: Path to the config YAML file *inside the container*,
                     e.g., 'configs/baseline.yaml' if copied there,
                     or '/root/configs/se_resnet.yaml' if placed there.
                     You might need to pass the config content or ensure the path is correct.
                     A more robust way is to pass the config name and have the script load it from a known location.
                         """
    print(f"Starting training with config: {config_path}")

    # --- Ensure Working Directory and Code ---
    # Option 1: Add your code during image build (recommended for stability)
    # image = image.add_local_python_source(".") # This adds your *entire* local directory, might be large
    # Or, more selectively:
    # image = image.add_local_dir("./src", remote_path="/root/src") # Add specific code dirs
    # For simplicity here, we assume the code is copied into the container via the image build or is available.
    # Let's assume the necessary files (trainer.py, data_loader.py, models/, utils.py, main.py)
    # and the configs/ folder are present in the container's working directory or a known path.
    # You might need to adjust the image build step to include these.

    # Example: Copy config file if it's in a mounted volume or passed differently
    # If configs are in a specific mounted path like /root/configs
    # shutil.copy(f"/root/configs/{config_name}", f"./configs/{config_name}")

    # --- Set Environment Variables (Optional but good practice) ---
    # os.environ['WANDB_MODE'] = 'offline' # Uncomment if you want to run W&B offline during Modal training
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Usually handled by Modal automatically

    # --- Run the training script ---
    # Assuming your main training entry point is in main.py and it accepts --config
    # and your code files (trainer.py, data_loader.py, etc.) are available in the container's PYTHONPATH/cwd
    try:
        # Ensure the output directory exists inside the container where it will be synced to the volume
        os.makedirs("/outputs/models", exist_ok=True)
        os.makedirs("/outputs/runs", exist_ok=True) # For TensorBoard logs if needed
        os.makedirs("/outputs/figures", exist_ok=True) # If trainer saves figures
        os.makedirs("/outputs/results", exist_ok=True) # If trainer saves results

        # Adjust the config path if necessary - it needs to be accessible from within the container.
        # If you mount configs separately, adjust the path.
        # If configs are in the cwd, it might be correct as is.
        # Example command assuming main.py is the entry point and configs are in ./configs relative to cwd:
        cmd = ["python", "main.py", "--config", config_path] # Use the passed config_path

        # Run the command
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd="/") # Or appropriate cwd if code is elsewhere

        print("Training completed successfully!")
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        # Optionally, sync the output volume explicitly (though Modal usually handles this)
        # subprocess.run(["modal", "volume", "sync", "audio-cnn-outputs"], check=True)

    except subprocess.CalledProcessError as e:
        print(f"Training failed with return code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise # Re-raise to mark the Modal function as failed
    except Exception as e:
        print(f"An unexpected error occurred during training: {e}")
        raise


# --- Optional: Local entrypoint to trigger the training ---
@stub.local_entrypoint()
def main(config_file: str = "configs/baseline.yaml"): # Default config, can be changed
    """
    Local entrypoint to start the Modal training function.
    """
    print(f"Triggering Modal training with config: {config_file}")
    # Ensure the config file path is accessible *inside* the Modal container.
    # If your image includes the 'configs' folder from your repo root, this path should work.
    # Otherwise, you might need to copy it or adjust the path.
    run_training.remote(config_file) # Pass the config file name/path

# Example: To run SE-ResNet-34 training
# @stub.local_entrypoint()
# def main_se():
#     print("Triggering Modal SE-ResNet-34 training")
#     run_training.remote("configs/se_resnet.yaml") # Pass the config file name/path
