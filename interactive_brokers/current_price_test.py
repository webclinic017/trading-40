from ibapi.client import EClient, MarketDataTypeEnum
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import threading
import time


class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

    def tickPrice(self, reqId, tickType, price, attrib):
        if tickType == 2 and reqId == 1:
            print(f"current ask price = {price}")

        # TMP
        elif reqId == 1:
            print(f"tmp current ask price = {price}")


def run_loop():
    app.run()


# Socket port - TWS settings
port = 7497

# Identifies this script to the API - any unique positive integer.
client_id = 123

app = IBapi()
app.connect(host="127.0.0.1", port=port, clientId=client_id)

# Start the socket in a thread.
api_thread = threading.Thread(target=run_loop, daemon=True)
api_thread.start()

# Allow time to connect to server.
time.sleep(1.0)


# Create contract object
contract = Contract()

# Chevron
# contract.symbol = "CVX"
# contract.secType = "STK"
# contract.exchange = "SMART"
# contract.currency = "USD"

# Micro Crude Oil
# contract.symbol = "MCL"
# contract.secType = "FUT"
# contract.exchange = "NYMEX"
# contract.currency = "USD"
# contract.localSymbol = "MCLG3"

# Mini Natural Gas
contract.symbol = "QG"
contract.secType = "FUT"
contract.exchange = "NYMEX"
contract.currency = "USD"
contract.localSymbol = "QGG3"

# Request data in delayed mode: -15 minutes.
app.reqMarketDataType(MarketDataTypeEnum.DELAYED)

# Request market data.
# app.reqMktData(1, contract, "", False, False, [])
# `genericTickList` was `tickType`
app.reqMktData(reqId=1, contract=contract, genericTickList="", snapshot=False, regulatorySnapshot=False, mktDataOptions=[])

# Allow time for incoming price data.
# time.sleep(10.0)
time.sleep(50.0)
app.disconnect()
