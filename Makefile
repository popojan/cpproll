all: .DEPS
	mkdir -p build; \
	cd build; \
	cmake ..;
	$(MAKE) -C build;

.DEPS:
	$(MAKE) -C deps;

profile:
	valgrind --tool=callgrind build/roll -b 18 -v info --l2 0.01 --passes 1 -B 2 --log "o\^" -T 1500 --standardize -I "q*t,s*s,s*Q" data/pla/train.vw

test:
	mkdir -p data
	example/test.sh
