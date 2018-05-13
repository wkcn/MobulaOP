from .. import glue
def register(op_name):
    return glue.backend.register(op_name)
