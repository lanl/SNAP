

def parse(filename):
	events = {}
	iline = 0
	start_time = 0
	stop_time = 0
	ifile = open(filename)
	for line in ifile:
		if line.find("<sweep_prof>:"):
			res = line.split("<sweep_prof>:")[1].replace(" ","")
			[info_s,time_s] = res.split("=")
			time = int(time_s)

			if iline == 0:
				start_time = time
				stop_time = time
			start_time = min(time,start_time)
			stop_time = max(time,stop_time)
			iline += 1

	ifile.close()
	print (stop_time- start_time)/1.0e6

parse("legion.log")
