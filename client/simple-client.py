import socket
import ssl
import certifi

import h2.connection
import h2.events

SERVER_NAME = 'myserver.io'
SERVER_PORT = 8443

print(certifi.where())

socket.setdefaulttimeout(15)
ctx = ssl.create_default_context(cafile=certifi.where())
ctx.set_alpn_protocols(['h2'])

s = socket.create_connection((SERVER_NAME, SERVER_PORT))
s = ctx.wrap_socket(s, server_hostname=SERVER_NAME)

c = h2.connection.H2Connection()
c.initiate_connection()
s.sendall(c.data_to_send())

headers = [(':method', 'GET'),
           (':path', '/'),
           (':authority', SERVER_NAME),
           (':scheme', 'https'),
        ]

c.send_headers(1, headers, end_stream=True)
s.sendall(c.data_to_send())

body = b''
response_stream_ended = False
while not response_stream_ended:
    data = s.recv(65536 * 1024)
    if not data:
        break

    events = c.receive_data(data)
    for event in events:
        print(event)
        if isinstance(event, h2.events.DataReceived):
            c.acknowledge_received_data(event.flow_controlled_length, event.stream_id)
            body += event.data
        if isinstance(event, h2.events.StreamEnded):
            response_stream_ended = True
            break

    s.sendall(c.data_to_send())


print("Response fully received:")
print(body.decode())

c.close_connection()
s.sendall(c.data_to_send())

s.close()