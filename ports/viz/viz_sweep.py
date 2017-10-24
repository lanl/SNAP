import numpy as np
import mayavi.mlab as mlab
import time as time
import  moviepy.editor as mpy
mlab.options.offscreen = True



class Cell:
	def __init__(self,_i,_j,_k,_ioct):
		self.i=_i
		self.j=_j
		self.k=_k
		self.ioct=_ioct
	def __hash__(self):
		return (self.i + 2*(self.j + 4* (self.k + 4*self.ioct)))

	def __eq__(self, other):
		return (self.i, self.j,self.k,self.ioct) == (other.i, other.j,other.k,other.ioct)

class Event:
	def __init__(self):
		self.starts = []
		self.stops = []
	def add_start(self,time):
		self.starts.append(time)
	def add_stop(self,time):
		self.stops.append(time)


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
			time -= start_time
			stop_time = max(time,stop_time)
			iline += 1
			info_s = info_s.strip("[")
			info_s = info_s.strip("]")
			info = info_s.split(",")
			print info
			istart = int(info[0])
			i = int(info[1])
			j = int(info[2])
			k = int(info[3])
			ioct = int(info[4])
			print istart, i, j, k, ioct, time
			newcell = Cell(i,j,k,ioct)
			if(newcell not in events):
				print "Adding Cell to events", i, j, k, istart
				events[newcell] = Event()
			
			temp = events[newcell]
			if istart == 1:
				temp.add_stop(time)
			else:
				temp.add_start(time)

			events[newcell] = temp

	ifile.close()
	return (0,stop_time,events)

def check_on(events,i,j,k,ioct,tmin,tmax):
	event = events[Cell(i,j,k,ioct)]
	for it in range(len(event.starts)):
		if (tmin >= event.starts[it] or tmax >= event.starts[it]) and (tmin < event.stops[it] or tmax < event.stops[it]):
			return True;

	return False;
		
nx = 8;
ny = 8;
nz = 8;
fig_myv = mlab.figure(size=(2048,2048), bgcolor=(1,1,1))

(start_time, stop_time,events) = parse("legion.log")
duration = 10.0; #video length in seconds
fps = 60.0
dt = duration/fps
X, Y , Z =np.linspace(0,nx,nx), np.linspace(0,ny,ny), np.linspace(0,nz,nz)
XX, YY, ZZ = np.meshgrid(X,Y,Z)
data = []
XX 
def get_data(t):
	t = (t)*duration/(duration-0.5) - 0.1
	s = np.zeros(shape=(nx,ny,nz));
	tmin = int((t)*stop_time/duration)
	tmax = int((t+0.4*dt)*stop_time/duration)
	for ioct in range(8):	
		for k in range(nz):
			for j in range(ny):
				for i in range(nx):
					if check_on(events,i,j,k,ioct,tmin,tmax):
						s[i][j][k] = ioct+1.0
	return s

#def make_frame(t):
#	mlab.clf()
#	s = np.zeros(shape=(nx,ny,nz));
#	t2 = int(t*stop_time/duration)
#	for ioct in range(8):	
#		for k in range(nz):
#			for j in range(ny):
#				for i in range(nx):
#					if check_on(events,i,j,k,ioct,t2):
#						s[i][j][k] = ioct+1.0
#			
#		
#	mlab.barchart(s,vmin=0,vmax=8,scale_mode='none',line_width=5.0,lateral_scale=1.0,transparent=True)
#	return mlab.screenshot(antialiased=True)

frames = []
for t in np.linspace(0.,(duration),int(fps*duration)):
	data.append(get_data(t))

temp_view = data[0];
bar = mlab.barchart(temp_view,vmin=0.0,vmax=9.0,scale_mode='none',line_width=2.0,lateral_scale=0.99,auto_scale=False)
lut = bar.module_manager.scalar_lut_manager.lut.table.to_array()

#for i in range(8):
#	lut[i+1] = lut[255*i/9]
lut[0] = [0,0,0,10]
bar.module_manager.scalar_lut_manager.lut.table = lut

data2 = []
for view in data:
	data2.append(np.reshape(view,np.product(view.shape)))

#for view in data:
#	bar.mlab_source.set(scalars=np.reshape(view,np.product(view.shape)))


def make_frame(t):

	view = data2[int(t*fps)]
	bar.mlab_source.set(scalars=view)
	return mlab.screenshot(antialiased=True)
	
animation = mpy.VideoClip(make_frame,duration=duration)
animation.write_videofile("my_animation.mp4", fps=48) # export as video
animation.write_gif("my_animation.gif", fps=24) # export as GIF (slow)


















