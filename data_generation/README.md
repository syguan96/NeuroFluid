# Generate fluid data (video + particles)

We tested on SPlisHSPlasH V2.4.0 and Blender 2.9. Thank Huayu for this part. 
 
## Dependencies
### Install SPlisHSPlasH V2.4.0
- Step 1: Download the source code of SPlisHSPlasH **V2.4.0** from the [release page](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH/releases).
- Step 2: Build the source code. 
```bash
sudo apt install git cmake xorg-dev freeglut3-dev build-essential
cd SPlisHSPlasH-2.4.0
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DUSE_PYTHON_BINDINGS=On ..
make -j 8
```
- Step 3: set the path of DynamicBoundarySimulator (pathto/SPlisHSPlasH-2.4.0/bin/DynamicBoundarySimulator) to `SIMULATOR_BIN` in splishsplash_config.py.

### Install pyopenvdb
We follow [this repo](https://github.com/theNewFlesh/docker_pyopenvdb) to install pyopenvdb.
```bash
pip install pyopenvdb
find / | grep -P 'pyopenvdb\.so'
export LD_LIBRARY_PATH=[parent directory]
```
If you meet this error: `ImportError: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory`, fix it as follows:
```bash
find / | grep -P 'libpython3\.7m\.so\.1\.0'
sudo cp pathto/libpython3.7m.so.1.0 /usr/lib
```

### Config Blender
Install [Stop-motion-OBJ](https://github.com/neverhood311/Stop-motion-OBJ), which is an add-on of blender. 

## Get Started
The procedure of generating data is (1) simulation with SPlisHSPlasH; (2) Load it with Stop-motion-OBJ; (3) Rendering with Blender engine.

### Simulation
```bash
bash run.sh YOUR_SAVEDIR
```
eg: bash run.sh ./data/default  
Please check arguments to specify fluid types, fluid shape, viscosity, density and scale. Generated mesh will be save under './data/default/sim_0001/mesh' directory.

### Load Mesh and Rendering
- Step 1: Download [example.blend](https://drive.google.com/file/d/1ytAg4EZtEAaNBaPEvK3ujYDMB2URIN_U/view?usp=sharing)
- Step 2: See this [instruction video](https://drive.google.com/file/d/1XEaC5LFzpZNQpg8pwn1EomY4X8Mr17MI/view?usp=sharing) to know how to load generated mesh and render the meshes.