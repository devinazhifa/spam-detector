import zmq

# ZeroMQ setup
context = zmq.Context()
socket = context.socket(zmq.REQ)  # REQ for Request
socket.connect("tcp://localhost:5555")

print("Client is running...")

while True:
    # Input message
    message = input("Enter your message (type 'exit' to quit): ")
    if message.lower() == 'exit':
        break

    # Send message to server
    socket.send_string(message)

    # Receive and display result
    result = socket.recv_string()
    print(f"Result: {result}")
