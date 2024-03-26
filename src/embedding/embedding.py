import datetime
from embedding.wordebd import WORDEBD
from model.model_FE import MODEL_FE


def get_embedding(vocab, args):
    print("{}, Building embedding".format(
        datetime.datetime.now()), flush=True)

    ebd = WORDEBD(vocab, args.finetune_ebd)
    model_FE = MODEL_FE(ebd, args)

    print("{}, Building embedding".format(
        datetime.datetime.now()), flush=True)

    if args.cuda != -1:
        model_FE= model_FE.cuda(args.cuda)
        return model_FE
    else:
        return model_FE