import numpy as np


def one_hot(labels, num_classes):
    labels = np.array(labels, dtype=int).reshape(-1)
    labels = np.where(labels >= num_classes, num_classes, labels)
    one_hot_encoded = np.zeros((len(labels), num_classes+1))
    one_hot_encoded[np.arange(len(labels)), labels] = 1.

    return one_hot_encoded


def predict_label(logits) -> np.ndarray:
    """Predict label from logits for a multi-label classification task.

    """
    return np.argmax(logits, axis=1).astype(np.uint8).reshape(-1) + 1


def mini_batch_gradient_descent(model: dict, lr: float):
    """One-step mini-batch SGD update of all model parameters.

    """
    for _, module in model.items():
        if hasattr(module, 'params'):
            for key, _ in module.params.items():
                grad = module.gradient[key]
                module.params[key] = module.params[key] - lr * grad
