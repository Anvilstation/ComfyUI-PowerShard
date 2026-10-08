if __package__:
    from .powershard.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    from .powershard.benchmark import install_if_requested
    from .powershard.web_api import register_routes
else:
    from powershard.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    from powershard.benchmark import install_if_requested
    from powershard.web_api import register_routes
install_if_requested()
register_routes()
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# --- Wan 2.1/2.2 (дополнение; строки выше не изменены) ---
if __package__:
    from .powershard.wan_nodes import WAN_NODE_CLASS_MAPPINGS, WAN_NODE_DISPLAY_NAME_MAPPINGS
else:
    from powershard.wan_nodes import WAN_NODE_CLASS_MAPPINGS, WAN_NODE_DISPLAY_NAME_MAPPINGS
NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **WAN_NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **WAN_NODE_DISPLAY_NAME_MAPPINGS}

# --- LTX-2 / 2.5, distributed Gemma, ускорители (дополнение; строки выше не изменены) ---
if __package__:
    from .powershard.ltx_nodes import LTX_NODE_CLASS_MAPPINGS, LTX_NODE_DISPLAY_NAME_MAPPINGS
else:
    from powershard.ltx_nodes import LTX_NODE_CLASS_MAPPINGS, LTX_NODE_DISPLAY_NAME_MAPPINGS
NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **LTX_NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **LTX_NODE_DISPLAY_NAME_MAPPINGS}
