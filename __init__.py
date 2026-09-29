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
