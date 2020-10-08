def patchify(images, patch_size, stride):
    # get all image windows of size (patch_size, patch_size) and stride (stride, stride)
    patches = images.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    patches = patches.permute(0, 2, 3, 4, 5, 1).contiguous()
    # patches.shape -> [batch, ... .... ... ..., ]
    # the size of flatten vector
    bs, pr, pc, h, w, ch = patches.shape[0], patches.shape[1], patches.shape[2], patches.shape[3], patches.shape[4], patches.shape[5]
    # bs->batch_size, rp->patches_row, pc->patches_col, h->patch_height, w->patch_width, w->patch_widht, ch->channels
    
    # dissolve it 
    patches = patches.view(bs, pr*pc, h*w*ch)
    
    return patches
    