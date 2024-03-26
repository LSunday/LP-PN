from dataset.utils import tprint

from model.LP_PN import LP_PN

def get_classifier(ebd_dim, args):
    tprint("Building classifier")
    

    model = LP_PN(ebd_dim,args)


    if args.cuda != -1:
        return model.cuda(args.cuda)
    else:
        return model