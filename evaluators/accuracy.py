from difflib import SequenceMatcher

def accuracy(pred, gt):
    return SequenceMatcher(None, pred.lower(), gt.lower()).ratio()