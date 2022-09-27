from .model import SEM_model
from .lightning import build_lightning, Lightning
from .ema import EMA


def build_model(args):
    return SEM_model(
        backbone=args.backbone,
        in_chans=args.in_chans,
        depth=args.depth,
        pretrained=args.pretrained,
        upsample=args.upsample,
        n_query=args.n_query,
        d_model=args.d_model,
        n_head=args.n_head,
        d_ff=args.d_ff,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dropout=args.dropout
    )