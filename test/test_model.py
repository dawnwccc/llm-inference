from model.model import BaseModelServer

server = BaseModelServer("demo", "demo", "cpu")
server.register_model()
