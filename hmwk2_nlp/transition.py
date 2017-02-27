class Transition(object):
    """
    This class defines a set of transitions which are applied to a
    configuration to get the next configuration.
    """
    # Define set of transitions
    LEFT_ARC = 'LEFTARC'
    RIGHT_ARC = 'RIGHTARC'
    SHIFT = 'SHIFT'
    REDUCE = 'REDUCE'

    def __init__(self):
        raise ValueError('Do not construct this object!')

    @staticmethod
    def left_arc(conf, relation):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        # raise NotImplementedError('Please implement left_arc!')

        if not conf.buffer or not conf.stack:
            return -1
        idx_wi = conf.stack[-1]
        idx_wj = conf.buffer[0]

        # check pre-conditions:
        # (1) (*, *, s[-1]) is NOT in arcs (i.e. s[-1] has no head yet), and
        # (2) s[-1] is NOT 'TOP'
        if idx_wi == 0 or [ a for a in conf.arcs if a[2] == idx_wi ]:
            return -1

        conf.stack.pop(-1)
        conf.arcs.append((idx_wj, relation, idx_wi))

    @staticmethod
    def right_arc(conf, relation):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.buffer or not conf.stack:
            return -1

        # You get this one for free! Use it as an example.

        idx_wi = conf.stack[-1]
        idx_wj = conf.buffer.pop(0)

        conf.stack.append(idx_wj)
        conf.arcs.append((idx_wi, relation, idx_wj))

    @staticmethod
    def reduce(conf):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        # raise NotImplementedError('Please implement reduce!')

        if not conf.stack:
            return -1
        idx_wi = conf.stack[-1]

        # check pre-condition:
        # (*, *, s[-1]) MUST be in arcs (i.e. s[-1] has a head already)
        if not [ a for a in conf.arcs if a[2] == idx_wi ]:
            return -1

        conf.stack.pop(-1)

    @staticmethod
    def shift(conf):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        # raise NotImplementedError('Please implement shift!')

        if not conf.buffer:
            return -1

        conf.stack.append(conf.buffer.pop(0))
