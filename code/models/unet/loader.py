import segmentation_models_pytorch as smp

def get_model():
    """
    Returns a U-Net model with a ResNet34 encoder.
    """
    # Create a U-Net model with ResNet34 encoder
    model = smp.Unet(encoder_name="resnet34", 
                     encoder_weights="imagenet",        # Pre-trained on ImageNet
                     in_channels=1,                     # US image is grayscale
                     out_channels=3,                    # 3 output channels for RGB mask
                     classes=8,                         # 7 muscle groups for segmentation + background
                     activation=None)
    return model


