# -*- coding: utf-8 -*-

import socket
import sys
import json
import numpy as np
import os

# TODO: Replace the string with your user ID
user_id = "aashish7k5"

# TODO: Change the filename of the output file.
# You should keep it in the format "speaker-data-<speaker>-#.csv"
filename="speaker-data-Abhinav-1.csv"#"speaker-data-HenryVIII-1.csv"

# TODO: Change the label to match the speaker; it must be numeric
label = 0

data_dir = "data"

if not os.path.exists(data_dir):
    os.mkdir(data_dir)
  

#################   Server Connection Code  ####################

msg_request_id = "ID"
msg_authenticate = "ID,{}\n"
msg_acknowledge_id = "ACK"

def authenticate(sock):
    """
    Authenticates the user by performing a handshake with the data collection server.
    
    If it fails, it will raise an appropriate exception.
    """
    message = sock.recv(256).strip().decode('ascii')
    if (message == msg_request_id):
        print("Received authentication request from the server. Sending authentication credentials...")
        sys.stdout.flush()
    else:
        print("Authentication failed!")
        raise Exception("Expected message {} from server, received {}".format(msg_request_id, message))
    sock.send(msg_authenticate.format(user_id).encode('utf-8'))

    try:
        message = sock.recv(256).strip().decode('ascii')
    except:
        print("Authentication failed!")
        raise Exception("Wait timed out. Failed to receive authentication response from server.")
        
    if (message.startswith(msg_acknowledge_id)):
        ack_id = message.split(",")[1]
    else:
        print("Authentication failed!")
        raise Exception("Expected message with prefix '{}' from server, received {}".format(msg_acknowledge_id, message))
    
    if (ack_id == user_id):
        print("Authentication successful.")
        sys.stdout.flush()
    else:
        print("Authentication failed!")
        raise Exception("Authentication failed : Expected user ID '{}' from server, received '{}'".format(user_id, ack_id))

try:

    '''
    This socket is used to receive data from the data collection server
    '''
    receive_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    receive_socket.connect(("none.cs.umass.edu", 8888))
    # ensures that after 1 second, a keyboard interrupt will close
    receive_socket.settimeout(1.0)

    print("Authenticating user for receiving data...")
    sys.stdout.flush()
    authenticate(receive_socket)
    
    print("Successfully connected to the server! Waiting for incoming data...")
    sys.stdout.flush()
        
    previous_json = ''
    labelled_data = []
        
    while True:
        try:
            message = receive_socket.recv(1024).strip().decode('ascii')
            json_strings = message.split("\n")
            json_strings[0] = previous_json + json_strings[0]
            for json_string in json_strings:
                try:
                    data = json.loads(json_string)
                except:
                    previous_json = json_string
                    continue
                previous_json = '' # reset if all were successful
                sensor_type = data['sensor_type']
                if (sensor_type == u"SENSOR_AUDIO"):
                    t = data['data']['t']
                    audio_buffer=data['data']['values']
                    print("Receiving audio data")                    
                    labelled_instance = [t]
                    labelled_instance.extend(audio_buffer)
                    labelled_instance.append(label)
                    labelled_data.append(labelled_instance)
                    
            sys.stdout.flush()

        except KeyboardInterrupt: 
            # occurs when the user presses Ctrl-C
            print("User Interrupt. Saving labelled data...")
            labelled_data = np.asarray(labelled_data)
            with open(os.path.join(data_dir, filename), "wb") as f:
                np.savetxt(f, labelled_data, delimiter=",")
            break

        except Exception as e:
            # ignore exceptions, such as parsing the json
            # if a connection timeout occurs, also ignore and try again. Use Ctrl-C to stop
            # but make sure the error is displayed so we know what's going on
            if (str(e) != "timed out"):  # ignore timeout exceptions completely       
                print(e)
            pass

except KeyboardInterrupt: 
    # occurs when the user presses Ctrl-C
    print("User Interrupt. Quitting...")
    quit()
    
finally:
    print('Closing socket for receiving data')
    receive_socket.shutdown(socket.SHUT_RDWR)
    receive_socket.close()