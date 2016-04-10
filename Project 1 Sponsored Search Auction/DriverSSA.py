from Bid import Bid
from Slot import Slot
from Auction import Auction

'''
reads a file, constructs an Auction from it,
	prints the bids and slots info, calls the auction.executeVCG() method,
	and prints results
	@param fName	the name of the file to read data from
'''
def runAndPrint(filename):
	print ("Loading data from file %s..." % filename)
	count = 0
	slots = []  # slot object array
	bids = []  # bid object array
	with open(filename) as infile:
			for line in infile:
				line = line.strip()
				count += 1
				# read auction term
				if count == 1:
					term = line
				# read click through rate
				if count == 2:
					parts = line.split(' ')
					for ctr in parts:
						slot = Slot(ctr)
						slots.append(slot)
				# read bids information
				if count > 2:
					bid = Bid(line)
					bids.append(bid)
				
	
	print ("\nAuction for \"%s\" with %d slots and %d bidders:" % (term, len(slots), len(bids)))
	for slot in slots:
		print("slot: %6.2f %8.2f %8.2f   %s" % (float(slot.clickThruRate), float(slot.price), float(slot.profit), slot.bidder))
	print ("	<-- click through rates")
	print(" ")

	auction = Auction(term, bids)	
	
	# print bid information
	print "Bids sorted from high to low:"
	for b in auction.bids:
# 		print ("%s\t%s" % (b.value, b.name))
		print ("bid:%6.2f %s" % (float(b.value), b.name))

	auction.executeVCG(slots)
	
	print(" ")
	print("%12s %8s %8s %8s\n" % ("clicks", "price", "profit", "bidder"))
	# print slots
	for slot in slots:
		print("slot: %6.2f %8.2f %8.2f   %s" % (float(slot.clickThruRate), float(slot.price), float(slot.profit), slot.bidder))
	# print sums
	cls = 0; rev = 0; val = 0;
	for s in slots:
		cls += float(s.clickThruRate);
		rev += float(s.price);
		val += float(s.profit);
	print ("sums: %6.2f %8.2f %8.2f\n" % (cls, rev, val))
	
if __name__ == '__main__':
	runAndPrint("burgers.data.txt")
# 	runAndPrint("etaMeson.data.txt")
# 	runAndPrint("bicycleParts.data.txt")
# 	runAndPrint("bicyclePartsDup.data.txt")
# 	runAndPrint("jewelers5.data.txt")
# 	runAndPrint("jewelers8.data.txt")
