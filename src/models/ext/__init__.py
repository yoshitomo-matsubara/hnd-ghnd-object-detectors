from models.ext.backbone import ExtBackboneWithFPN


def get_ext_fpn_backbone(base_backbone, ext_config, freeze_layers):
    if freeze_layers:
        for name, parameter in base_backbone.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

    return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}
    in_channels_stage2 = base_backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = 256
    return ExtBackboneWithFPN(base_backbone, return_layers, in_channels_list, out_channels, ext_config)
