all: deps

clipp:
	git clone https://github.com/muellan/clipp.git

spdlog:
	git clone https://github.com/gabime/spdlog.git

boost-bloom-filters:
	git clone https://github.com/queertypes/boost-bloom-filters.git

# osklive, potreba ubalit a dat do zavislosti:

liblbfgs:
	git clone https://github.com/chokkan/liblbfgs.git
	cd liblbfgs; \
    ./autogen.sh; \
    ./configure
	$(MAKE) -C liblbfgs

deps: spdlog boost-bloom-filters clipp liblbfgs

