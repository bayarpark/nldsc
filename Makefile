.PHONY: build buildi clean


build:
	@cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ -DCMAKE_MAKE_PROGRAM=ninja -G Ninja ./nldsc/ldscore/_ldscore -B ./nldsc/ldscore/_ldscore/build
	@cmake --build ./nldsc/ldscore/_ldscore/build --target _ldscore
	@cp ./nldsc/ldscore/_ldscore/build/_ldscore*.so ./nldsc/ldscore/_ldscore.so

buildi:
	@cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx -DCMAKE_MAKE_PROGRAM=ninja -G Ninja ./nldsc/ldscore/_ldscore -B ./nldsc/ldscore/_ldscore/build
	@cmake --build ./nldsc/ldscore/_ldscore/build --target _ldscore
	@cp ./nldsc/ldscore/_ldscore/build/_ldscore*.so ./nldsc/ldscore/_ldscore.so

clean:
	@rm -rf ./src/ldscore/_ldscore/build
	@rm ./src/ldscore/ldscore.so

