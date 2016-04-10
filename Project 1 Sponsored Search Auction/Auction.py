class Auction:
	'This class represents an auction of multiple ad slots to multiple advertisers'
	query = ""
	bids = []

	def __init__(self, term, bids1=[]):
		self.query = term
		
		# sort
		for b in bids1:
			j = 0
# 			print len(self.bids)
			while j < len(self.bids) and float(b.value) < float(self.bids[j].value):
				j += 1
			self.bids.insert(j, b)

	'''
	This method accepts a vector of slots and fills it with the results
	of a VCG auction. The competition for those slots is specified in the bids vector.
	@param slots a vector of Slots, which (on entry) specifies only the clickThruRates
	and (on exit) also specifies the name of the bidder who won that slot,
	the price said bidder must pay, and the expected profit for the bidder.  
	'''

	def executeVCG(self, slots):		
		# set the range and the base price
		if len(slots) >= len(self.bids):
			length = len(self.bids)  # if available bids are fewer than available slots
			slots[length - 1].price = 0  # set the base price
		else:
			length = len(slots)  # if available slots are fewer than available bids
			slots[-1].price = slots[-1].clickThruRate * self.bids[length].value  # set the base price

		# price_VCG_i = (Click_i-Click_(i+1))*Bid_(i+1) + price_VCG_(i+1)
		for i in range(length - 1, 0, -1):
			# testing
# 			print "\nprice_%d now is %d" % (i - 1, slots[i - 1].price)
# 			print "price_%d now is %d" % (i, slots[i].price)
# 			print "CTR_%d now is %d" % (i - 1, slots[i - 1].clickThruRate)
# 			print "CTR_%d now is %d" % (i, slots[i].clickThruRate)
# 			print "value_%d now is %d" % (i, self.bids[i].value)
			slots[i - 1].price = (slots[i - 1].clickThruRate - slots[i].clickThruRate) * self.bids[i].value + slots[i].price
# 			print "\nprice_%d now is %d" % (i - 1, slots[i - 1].price) 
			
		# profit
		for i in range(0, length):
			slots[i].profit = slots[i].clickThruRate * self.bids[i].value - slots[i].price
			
		# bidder
		for i in range(0, length):
			slots[i].bidder = self.bids[i].name