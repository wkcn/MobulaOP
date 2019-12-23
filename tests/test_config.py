import mobula
from mobula import config
from nose.tools import assert_raises


def test_config():
    new_target_name = 'mobula_op_bak'
    config.TARGET = new_target_name
    assert config.TARGET == new_target_name, (config.TARGET, new_target_name)

    config.DEBUG = True
    assert config.DEBUG
    config.DEBUG = False

    assert_raises(AttributeError, lambda: config.UNKNOWN_ATTR)

    def assign_target_num():
        config.TARGET = 10
    assert_raises(TypeError, assign_target_num)
