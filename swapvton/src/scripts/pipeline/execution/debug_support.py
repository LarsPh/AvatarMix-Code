from ..core.constants import DEBUG_PORTS


def get_debug_port(stage_name):

    return DEBUG_PORTS.get(stage_name, 5678)
