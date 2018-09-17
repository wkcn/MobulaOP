from .. import glue


def Custom(op_name):
    assert op_name in glue.CUSTOM_OP_LIST, KeyError(
        'Operator {} not found'.format(op_name))
    return glue.CUSTOM_OP_LIST[op_name]


def CustomList():
    return glue.CUSTOM_OP_LIST.keys()
