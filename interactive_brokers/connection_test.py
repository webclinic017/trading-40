import time
from ibapi.client import EClient
from ibapi.wrapper import EWrapper


# https://algotrading101.com/learn/interactive-brokers-python-api-native-guide/
class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)


# Socket port - TWS settings
port = 7497

# Identifies this script to the API - any unique positive integer.
client_id = 123

app = IBapi()
app.connect(host="127.0.0.1", port=port, clientId=client_id)
app.run()

time.sleep(2.0)
app.disconnect()
