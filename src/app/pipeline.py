"""Pipeline primitives and phase registrations (placeholders)."""

_registered_phases = {}


def register_phase(name):
    def _deco(fn):
        _registered_phases[name] = fn
        return fn
    return _deco


def get_phase(name):
    return _registered_phases.get(name)


# Example usage:
# @register_phase('index_source')
# def index_source(project_id):
#     pass
