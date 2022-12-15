import backtrader as bt


# Introduces -inf in reward.
class FixedReverser(bt.Sizer):
    params = (("stake", 1),)

    # https://www.backtrader.com/docu/sizers/sizers/#practical-sizer-applicability
    def _getsizing(self, comminfo, cash, data, isbuy):
        position = self.broker.getposition(data)
        size = self.p.stake * (1 + (position.size != 0))
        print(size)
        return size
