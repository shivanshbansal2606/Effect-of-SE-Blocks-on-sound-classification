import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from torchcam.methods import CAM, GradCAM, ScoreCAM # Example, you can choose others
from torchcam.utils import overlay_mask

def generate_gradcam_visualization(model, input_tensor, target_class_idx, device, save_path):
    """"""
    Generates a Grad-CAM visualization for a given input and target class.
    Saves the combined image (input + heatmap) to save_path.
    """"""
    model.eval() # Set model to evaluation mode

    # Ensure input_tensor has batch dimension and is on the correct device
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0) # Add batch dimension if missing
    input_tensor = input_tensor.to(device)

    # --- Choose CAM method (GradCAM, ScoreCAM, etc.) ---
    cam_extractor = GradCAM(model, target_layer=model.layer4[-1]) # Example: use last layer of layer4
    # cam_extractor = ScoreCAM(model, target_layer=model.layer4[-1]) # Or use ScoreCAM

    # Forward pass
    out = model(input_tensor)
    # print(f""Model output shape: {out.shape}"") # Debug print

    # Get the CAM for the target class
    activation_map = cam_extractor(target_class_idx, out) # Pass the output tensor
    # print(f""Activation map shape: {activation_map.shape}"") # Debug print

    if activation_map is None:
        print(f""Warning: CAM returned None for target class {target_class_idx}. Skipping visualization."")
        return

    # Convert activation map to numpy array
    # The shape might be [B, H, W] or [H, W], depending on the CAM implementation
    # Assume [H, W] for single image after extraction
    activation_map = activation_map.squeeze(0).cpu().numpy() # Remove batch dimension and move to CPU

    # Ensure input_tensor is in the correct format for visualization (C, H, W) -> (H, W, C) if needed
    # Mel spectrograms are typically (1, H, W), so we take the first channel
    input_image = input_tensor.squeeze(0).cpu().numpy()[0] # Shape: (H, W)

    # --- Overlay heatmap on input image ---
    # The overlay_mask function expects the input image and activation map in specific formats
    # Often (H, W, C) for RGB or (H, W) for grayscale, and (H, W) for activation map
    # Input spectrogram is grayscale (H, W), activation map is (H, W)
    # Let's use matplotlib to create the overlay manually for more control

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot original spectrogram
    im1 = axes[0].imshow(input_image, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Input Spectrogram')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])

    # Plot activation map (CAM)
    im2 = axes[1].imshow(activation_map, aspect='auto', origin='lower', cmap='jet', alpha=0.7)
    axes[1].set_title('CAM Heatmap')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])

    # Plot overlay
    # Normalize the activation map to [0, 1] for blending
    cam_normalized = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-8)
    axes[2].imshow(input_image, aspect='auto', origin='lower', cmap='viridis')
    axes[2].imshow(cam_normalized, aspect='auto', origin='lower', cmap='jet', alpha=0.5) # Blend with alpha
    axes[2].set_title('Spectrogram + CAM Overlay')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150) # Save with higher resolution
    plt.close() # Close the plot to free memory
    print(f""Grad-CAM visualization saved to {save_path}"")

# Example usage function (call this from another script)
def run_example_visualization(model, dataloader, device, num_examples=3):
    """"""
    Runs Grad-CAM visualization on a few examples from the dataloader.
    """"""
    model.eval()
    count = 0
    with torch.no_grad():
        for data, target in dataloader:
            if count >= num_examples:
                break
            # Take the first item in the batch
            input_spec = data[0]
            true_label = target[0].item()

            # Predict the class (or use true label for visualization)
            with torch.enable_grad():
                output = model(input_spec.unsqueeze(0).to(device))
            predicted_label = output.argmax(dim=1).item()

            # Generate Grad-CAM for the *predicted* class (or true class)
            # Using predicted label often shows what the model *thinks* is important
            save_path = os.path.join(""./outputs/figures"", f""gradcam_example_{count}_pred_{predicted_label}_true_{true_label}.png"")
            generate_gradcam_visualization(model, input_spec, predicted_label, device, save_path)
            count += 1

# Note: You need to load your model and dataloader (e.g., test_loader) in the script where you call this
