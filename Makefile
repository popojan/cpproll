all: .DEPS
	mkdir -p build; \
	cd build; \
	cmake ..;
	$(MAKE) -C build;

.DEPS:
	$(MAKE) -C deps;

profile:
	valgrind --tool=callgrind build/roll -b 18 -I q*k,q*c,q*u,z*x -v info -s data/text.vw

test:
	mkdir -p data
	example/test.sh
