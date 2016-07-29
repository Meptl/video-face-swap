CXX=g++
CFLAGS=`pkg-config opencv --cflags --libs` -Wall

all:
	$(CXX) $(CFLAGS) face-replace.cpp -o video-face-swap
clean:
	rm video-face-swap
