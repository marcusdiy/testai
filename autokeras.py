import autokeras

autokeras.AutoModel(
    inputs,
    outputs,
    name="auto_model",
    max_trials=100,
    directory=None,
    objective="val_loss",
    tuner="greedy",
    overwrite=False,
    seed=None,
)