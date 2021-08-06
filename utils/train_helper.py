from graphs.models.models import DeeplabNW

def get_model(args):
    model = DeeplabNW(num_classes=args.num_classes, backbone=args.backbone, pretrained=args.imagenet_pretrained)
    params = model.optim_parameters(args)
    args.numpy_transform = True
    return model, params