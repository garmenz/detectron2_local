def add_Cityscapes_config(cfg):
    '''
    Add config for training panoptic FPN on Cityscapes
    '''
    cfg.INPUT.CROP.COLOR_AUG_SSD = False