import server_robot_env
from ur_env.remote import RemoteEnvServer, RemoteEnvClient

host = "10.46.3.232"
port = "5555"
address = (host, port)

env = server_robot_env.RobotEnv()
server = RemoteEnvServer(env, address)

server.run()
