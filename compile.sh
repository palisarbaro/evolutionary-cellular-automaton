export LLVM_BUILD=~/pr/llvm/build
export PATH=$LLVM_BUILD/bin:$PATH
export LD_LIBRARY_PATH=$LLVM_BUILD/lib:$LD_LIBRARY_PATH
src=$(find  src/*.cpp)
echo "Files to compile: "$src
clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda -lglut -lGLU -lGL  $src
if [ $? -eq 0 ]; then
    ./a.out 
fi