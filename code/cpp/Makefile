CFLAGS=`pkg-config --cflags eigen3`
all: kadai1

kadai1: kadai1.cpp
	g++ -o $@ $< ${CFLAGS}

clean: kadai1
	rm -rf $<