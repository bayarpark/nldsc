.PHONY: build ubuntu-install-dependencies


build :
	@cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=ninja -G Ninja ./src/ldscore/ -B ./src/ldscore/build
	@cmake --build ./src/ldscore/build --target ldscore
	@cp ./src/ldscore/build/ldscore*.so ./src/ldscore/ldscore.so

ubuntu-install-dependencies : 
	@sudo apt update
	@sudo apt -y install cmake gfortran libblas-dev liblapack-dev

