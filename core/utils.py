import tensorflow as tf


def copy_model_parameters(from_scope, to_scope):
    """
    Copies the model's parameters of `from_model` to `to_model`.

    Args:
        from_model: model to copy the paramters from
        to_model:   model to copy the parameters to
    """
    from_model_paras = [
        v for v in tf.trainable_variables() if v.name.startswith(from_scope)
    ]
    from_model_paras = sorted(from_model_paras, key=lambda v: v.name)
    to_model_paras = [
        v for v in tf.trainable_variables() if v.name.startswith(to_scope)
    ]
    to_model_paras = sorted(to_model_paras, key=lambda v: v.name)
    update_ops = []
    for from_model_para, to_model_para in zip(from_model_paras,
                                              to_model_paras):
        op = to_model_para.assign(from_model_para)
        update_ops.append(op)
    return update_ops
