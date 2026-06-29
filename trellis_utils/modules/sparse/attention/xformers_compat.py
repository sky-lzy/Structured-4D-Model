def get_block_diagonal_mask_cls(xops):
    if hasattr(xops.fmha, 'BlockDiagonalMask'):
        return xops.fmha.BlockDiagonalMask
    if hasattr(xops.fmha, 'attn_bias') and hasattr(xops.fmha.attn_bias, 'BlockDiagonalMask'):
        return xops.fmha.attn_bias.BlockDiagonalMask
    raise AttributeError("xformers BlockDiagonalMask is unavailable")
