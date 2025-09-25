from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR, ExponentialLR, ConstantLR, LambdaLR, LinearLR

def get_scheduler(name):
    if name is None:
        name = 'step'
    return {
        "step": StepLR,
        "multi_step": MultiStepLR,
        "cosine_annealing": CosineAnnealingLR,
        "exponential": ExponentialLR,
        "constant": ConstantLR,
        "lambda": LambdaLR,
        "linear": LinearLR
    }[name]