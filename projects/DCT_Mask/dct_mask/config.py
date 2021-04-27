
def add_dctmask_config(cfg):
    """
    Add config for DCT-Mask.
    """

    # For MaskRCNNDCTHead
    cfg.MODEL.ROI_MASK_HEAD.DCT_VECTOR_DIM = 300
    cfg.MODEL.ROI_MASK_HEAD.MASK_SIZE = 128
    cfg.MODEL.ROI_MASK_HEAD.DCT_LOSS_TYPE = "l1"
    cfg.MODEL.ROI_MASK_HEAD.MASK_LOSS_PARA = 1.0
