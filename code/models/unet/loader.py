import segmentation_models_pytorch as smp

def get_model():
    """
    Returns a U-Net model with a ResNet34 encoder.
    """
    # Create a U-Net model with ResNet34 encoder
    model = smp.Unet(encoder_name="resnet34", 
                     encoder_weights="imagenet",        # Pre-trained on ImageNet
                     in_channels=1,                     # US image is grayscale
                     classes=7,                         # 7 muscle groups for segmentation    
                     activation="softmax")
    return model


