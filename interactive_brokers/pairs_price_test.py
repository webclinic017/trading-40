from ibapi.client import EClient, MarketDataTypeEnum
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from typing import List
import threading
import time


class IBapi(EWrapper, EClient):
    def __init__(self, req_ids: List[int]):
        self.req_ids = req_ids
        EClient.__init__(self, self)

    def tickPrice(self, reqId, tickType, price, attrib):
        if tickType == 2 and reqId in self.req_ids:
            print(f"current ask price = {price}")

        # TMP
        else:
            print(f"req_id {reqId}: current ask price = {price}")


def run_loop():
    app.run()


# Socket port - TWS settings
port = 7497

# Identifies this script to the API - any unique positive integer.
client_id = 123

# Req ID per asset - determines which asset price data is received via sockets.
req_ids = [1, 2]
app = IBapi(req_ids=req_ids)
app.connect(host="127.0.0.1", port=port, clientId=client_id)

# Start the socket in a thread.
api_thread = threading.Thread(target=run_loop, daemon=True)
api_thread.start()

# Allow time to connect to server.
time.sleep(1.0)

# Micro Crude Oil
contract0 = Contract()
contract0.symbol = "MCL"
contract0.secType = "FUT"
contract0.exchange = "NYMEX"
contract0.currency = "USD"
contract0.localSymbol = "MCLG3"

# Mini Natural Gas
contract1 = Contract()
contract1.symbol = "QG"
contract1.secType = "FUT"
contract1.exchange = "NYMEX"
contract1.currency = "USD"
contract1.localSymbol = "QGG3"

# Request data in delayed mode: -15 minutes.
app.reqMarketDataType(MarketDataTypeEnum.DELAYED)

# Request market data.
app.reqMktData(reqId=req_ids[0], contract=contract0, genericTickList="", snapshot=False, regulatorySnapshot=False, mktDataOptions=[])
app.reqMktData(reqId=req_ids[1], contract=contract1, genericTickList="", snapshot=False, regulatorySnapshot=False, mktDataOptions=[])

# Allow time for incoming price data.
# time.sleep(10.0)
time.sleep(5.0)
app.disconnect()
