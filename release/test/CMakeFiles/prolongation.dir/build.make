# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/liaowang/Documents/master-thesis/code

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/liaowang/Documents/master-thesis/code/release

# Include any dependencies generated for this target.
include test/CMakeFiles/prolongation.dir/depend.make

# Include the progress variables for this target.
include test/CMakeFiles/prolongation.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/prolongation.dir/flags.make

test/CMakeFiles/prolongation.dir/test_prolongation.cpp.o: test/CMakeFiles/prolongation.dir/flags.make
test/CMakeFiles/prolongation.dir/test_prolongation.cpp.o: ../test/test_prolongation.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liaowang/Documents/master-thesis/code/release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/prolongation.dir/test_prolongation.cpp.o"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/prolongation.dir/test_prolongation.cpp.o -c /home/liaowang/Documents/master-thesis/code/test/test_prolongation.cpp

test/CMakeFiles/prolongation.dir/test_prolongation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/prolongation.dir/test_prolongation.cpp.i"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liaowang/Documents/master-thesis/code/test/test_prolongation.cpp > CMakeFiles/prolongation.dir/test_prolongation.cpp.i

test/CMakeFiles/prolongation.dir/test_prolongation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/prolongation.dir/test_prolongation.cpp.s"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liaowang/Documents/master-thesis/code/test/test_prolongation.cpp -o CMakeFiles/prolongation.dir/test_prolongation.cpp.s

test/CMakeFiles/prolongation.dir/__/LagrangeO1/HE_LagrangeO1.cpp.o: test/CMakeFiles/prolongation.dir/flags.make
test/CMakeFiles/prolongation.dir/__/LagrangeO1/HE_LagrangeO1.cpp.o: ../LagrangeO1/HE_LagrangeO1.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liaowang/Documents/master-thesis/code/release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object test/CMakeFiles/prolongation.dir/__/LagrangeO1/HE_LagrangeO1.cpp.o"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/prolongation.dir/__/LagrangeO1/HE_LagrangeO1.cpp.o -c /home/liaowang/Documents/master-thesis/code/LagrangeO1/HE_LagrangeO1.cpp

test/CMakeFiles/prolongation.dir/__/LagrangeO1/HE_LagrangeO1.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/prolongation.dir/__/LagrangeO1/HE_LagrangeO1.cpp.i"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liaowang/Documents/master-thesis/code/LagrangeO1/HE_LagrangeO1.cpp > CMakeFiles/prolongation.dir/__/LagrangeO1/HE_LagrangeO1.cpp.i

test/CMakeFiles/prolongation.dir/__/LagrangeO1/HE_LagrangeO1.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/prolongation.dir/__/LagrangeO1/HE_LagrangeO1.cpp.s"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liaowang/Documents/master-thesis/code/LagrangeO1/HE_LagrangeO1.cpp -o CMakeFiles/prolongation.dir/__/LagrangeO1/HE_LagrangeO1.cpp.s

test/CMakeFiles/prolongation.dir/__/Pum_WaveRay/HE_FEM.cpp.o: test/CMakeFiles/prolongation.dir/flags.make
test/CMakeFiles/prolongation.dir/__/Pum_WaveRay/HE_FEM.cpp.o: ../Pum_WaveRay/HE_FEM.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liaowang/Documents/master-thesis/code/release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object test/CMakeFiles/prolongation.dir/__/Pum_WaveRay/HE_FEM.cpp.o"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/prolongation.dir/__/Pum_WaveRay/HE_FEM.cpp.o -c /home/liaowang/Documents/master-thesis/code/Pum_WaveRay/HE_FEM.cpp

test/CMakeFiles/prolongation.dir/__/Pum_WaveRay/HE_FEM.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/prolongation.dir/__/Pum_WaveRay/HE_FEM.cpp.i"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liaowang/Documents/master-thesis/code/Pum_WaveRay/HE_FEM.cpp > CMakeFiles/prolongation.dir/__/Pum_WaveRay/HE_FEM.cpp.i

test/CMakeFiles/prolongation.dir/__/Pum_WaveRay/HE_FEM.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/prolongation.dir/__/Pum_WaveRay/HE_FEM.cpp.s"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liaowang/Documents/master-thesis/code/Pum_WaveRay/HE_FEM.cpp -o CMakeFiles/prolongation.dir/__/Pum_WaveRay/HE_FEM.cpp.s

test/CMakeFiles/prolongation.dir/__/Pum_WaveRay/PUM_WaveRay.cpp.o: test/CMakeFiles/prolongation.dir/flags.make
test/CMakeFiles/prolongation.dir/__/Pum_WaveRay/PUM_WaveRay.cpp.o: ../Pum_WaveRay/PUM_WaveRay.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liaowang/Documents/master-thesis/code/release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object test/CMakeFiles/prolongation.dir/__/Pum_WaveRay/PUM_WaveRay.cpp.o"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/prolongation.dir/__/Pum_WaveRay/PUM_WaveRay.cpp.o -c /home/liaowang/Documents/master-thesis/code/Pum_WaveRay/PUM_WaveRay.cpp

test/CMakeFiles/prolongation.dir/__/Pum_WaveRay/PUM_WaveRay.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/prolongation.dir/__/Pum_WaveRay/PUM_WaveRay.cpp.i"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liaowang/Documents/master-thesis/code/Pum_WaveRay/PUM_WaveRay.cpp > CMakeFiles/prolongation.dir/__/Pum_WaveRay/PUM_WaveRay.cpp.i

test/CMakeFiles/prolongation.dir/__/Pum_WaveRay/PUM_WaveRay.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/prolongation.dir/__/Pum_WaveRay/PUM_WaveRay.cpp.s"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liaowang/Documents/master-thesis/code/Pum_WaveRay/PUM_WaveRay.cpp -o CMakeFiles/prolongation.dir/__/Pum_WaveRay/PUM_WaveRay.cpp.s

test/CMakeFiles/prolongation.dir/__/Pum/HE_PUM.cpp.o: test/CMakeFiles/prolongation.dir/flags.make
test/CMakeFiles/prolongation.dir/__/Pum/HE_PUM.cpp.o: ../Pum/HE_PUM.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liaowang/Documents/master-thesis/code/release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object test/CMakeFiles/prolongation.dir/__/Pum/HE_PUM.cpp.o"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/prolongation.dir/__/Pum/HE_PUM.cpp.o -c /home/liaowang/Documents/master-thesis/code/Pum/HE_PUM.cpp

test/CMakeFiles/prolongation.dir/__/Pum/HE_PUM.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/prolongation.dir/__/Pum/HE_PUM.cpp.i"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liaowang/Documents/master-thesis/code/Pum/HE_PUM.cpp > CMakeFiles/prolongation.dir/__/Pum/HE_PUM.cpp.i

test/CMakeFiles/prolongation.dir/__/Pum/HE_PUM.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/prolongation.dir/__/Pum/HE_PUM.cpp.s"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liaowang/Documents/master-thesis/code/Pum/HE_PUM.cpp -o CMakeFiles/prolongation.dir/__/Pum/HE_PUM.cpp.s

test/CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_EdgeMat.cpp.o: test/CMakeFiles/prolongation.dir/flags.make
test/CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_EdgeMat.cpp.o: ../ExtendPum/ExtendPUM_EdgeMat.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liaowang/Documents/master-thesis/code/release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object test/CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_EdgeMat.cpp.o"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_EdgeMat.cpp.o -c /home/liaowang/Documents/master-thesis/code/ExtendPum/ExtendPUM_EdgeMat.cpp

test/CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_EdgeMat.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_EdgeMat.cpp.i"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liaowang/Documents/master-thesis/code/ExtendPum/ExtendPUM_EdgeMat.cpp > CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_EdgeMat.cpp.i

test/CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_EdgeMat.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_EdgeMat.cpp.s"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liaowang/Documents/master-thesis/code/ExtendPum/ExtendPUM_EdgeMat.cpp -o CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_EdgeMat.cpp.s

test/CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_EdgeVector.cpp.o: test/CMakeFiles/prolongation.dir/flags.make
test/CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_EdgeVector.cpp.o: ../ExtendPum/ExtendPUM_EdgeVector.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liaowang/Documents/master-thesis/code/release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object test/CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_EdgeVector.cpp.o"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_EdgeVector.cpp.o -c /home/liaowang/Documents/master-thesis/code/ExtendPum/ExtendPUM_EdgeVector.cpp

test/CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_EdgeVector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_EdgeVector.cpp.i"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liaowang/Documents/master-thesis/code/ExtendPum/ExtendPUM_EdgeVector.cpp > CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_EdgeVector.cpp.i

test/CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_EdgeVector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_EdgeVector.cpp.s"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liaowang/Documents/master-thesis/code/ExtendPum/ExtendPUM_EdgeVector.cpp -o CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_EdgeVector.cpp.s

test/CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_ElementMatrix.cpp.o: test/CMakeFiles/prolongation.dir/flags.make
test/CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_ElementMatrix.cpp.o: ../ExtendPum/ExtendPUM_ElementMatrix.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liaowang/Documents/master-thesis/code/release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object test/CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_ElementMatrix.cpp.o"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_ElementMatrix.cpp.o -c /home/liaowang/Documents/master-thesis/code/ExtendPum/ExtendPUM_ElementMatrix.cpp

test/CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_ElementMatrix.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_ElementMatrix.cpp.i"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liaowang/Documents/master-thesis/code/ExtendPum/ExtendPUM_ElementMatrix.cpp > CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_ElementMatrix.cpp.i

test/CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_ElementMatrix.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_ElementMatrix.cpp.s"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liaowang/Documents/master-thesis/code/ExtendPum/ExtendPUM_ElementMatrix.cpp -o CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_ElementMatrix.cpp.s

test/CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_ElemVector.cpp.o: test/CMakeFiles/prolongation.dir/flags.make
test/CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_ElemVector.cpp.o: ../ExtendPum/ExtendPUM_ElemVector.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liaowang/Documents/master-thesis/code/release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object test/CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_ElemVector.cpp.o"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_ElemVector.cpp.o -c /home/liaowang/Documents/master-thesis/code/ExtendPum/ExtendPUM_ElemVector.cpp

test/CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_ElemVector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_ElemVector.cpp.i"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liaowang/Documents/master-thesis/code/ExtendPum/ExtendPUM_ElemVector.cpp > CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_ElemVector.cpp.i

test/CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_ElemVector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_ElemVector.cpp.s"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liaowang/Documents/master-thesis/code/ExtendPum/ExtendPUM_ElemVector.cpp -o CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_ElemVector.cpp.s

test/CMakeFiles/prolongation.dir/__/ExtendPum/HE_ExtendPUM.cpp.o: test/CMakeFiles/prolongation.dir/flags.make
test/CMakeFiles/prolongation.dir/__/ExtendPum/HE_ExtendPUM.cpp.o: ../ExtendPum/HE_ExtendPUM.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liaowang/Documents/master-thesis/code/release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object test/CMakeFiles/prolongation.dir/__/ExtendPum/HE_ExtendPUM.cpp.o"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/prolongation.dir/__/ExtendPum/HE_ExtendPUM.cpp.o -c /home/liaowang/Documents/master-thesis/code/ExtendPum/HE_ExtendPUM.cpp

test/CMakeFiles/prolongation.dir/__/ExtendPum/HE_ExtendPUM.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/prolongation.dir/__/ExtendPum/HE_ExtendPUM.cpp.i"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liaowang/Documents/master-thesis/code/ExtendPum/HE_ExtendPUM.cpp > CMakeFiles/prolongation.dir/__/ExtendPum/HE_ExtendPUM.cpp.i

test/CMakeFiles/prolongation.dir/__/ExtendPum/HE_ExtendPUM.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/prolongation.dir/__/ExtendPum/HE_ExtendPUM.cpp.s"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liaowang/Documents/master-thesis/code/ExtendPum/HE_ExtendPUM.cpp -o CMakeFiles/prolongation.dir/__/ExtendPum/HE_ExtendPUM.cpp.s

test/CMakeFiles/prolongation.dir/__/ExtendPum_WaveRay/ExtendPUM_WaveRay.cpp.o: test/CMakeFiles/prolongation.dir/flags.make
test/CMakeFiles/prolongation.dir/__/ExtendPum_WaveRay/ExtendPUM_WaveRay.cpp.o: ../ExtendPum_WaveRay/ExtendPUM_WaveRay.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liaowang/Documents/master-thesis/code/release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object test/CMakeFiles/prolongation.dir/__/ExtendPum_WaveRay/ExtendPUM_WaveRay.cpp.o"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/prolongation.dir/__/ExtendPum_WaveRay/ExtendPUM_WaveRay.cpp.o -c /home/liaowang/Documents/master-thesis/code/ExtendPum_WaveRay/ExtendPUM_WaveRay.cpp

test/CMakeFiles/prolongation.dir/__/ExtendPum_WaveRay/ExtendPUM_WaveRay.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/prolongation.dir/__/ExtendPum_WaveRay/ExtendPUM_WaveRay.cpp.i"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liaowang/Documents/master-thesis/code/ExtendPum_WaveRay/ExtendPUM_WaveRay.cpp > CMakeFiles/prolongation.dir/__/ExtendPum_WaveRay/ExtendPUM_WaveRay.cpp.i

test/CMakeFiles/prolongation.dir/__/ExtendPum_WaveRay/ExtendPUM_WaveRay.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/prolongation.dir/__/ExtendPum_WaveRay/ExtendPUM_WaveRay.cpp.s"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liaowang/Documents/master-thesis/code/ExtendPum_WaveRay/ExtendPUM_WaveRay.cpp -o CMakeFiles/prolongation.dir/__/ExtendPum_WaveRay/ExtendPUM_WaveRay.cpp.s

test/CMakeFiles/prolongation.dir/__/utils/utils.cpp.o: test/CMakeFiles/prolongation.dir/flags.make
test/CMakeFiles/prolongation.dir/__/utils/utils.cpp.o: ../utils/utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liaowang/Documents/master-thesis/code/release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object test/CMakeFiles/prolongation.dir/__/utils/utils.cpp.o"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/prolongation.dir/__/utils/utils.cpp.o -c /home/liaowang/Documents/master-thesis/code/utils/utils.cpp

test/CMakeFiles/prolongation.dir/__/utils/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/prolongation.dir/__/utils/utils.cpp.i"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liaowang/Documents/master-thesis/code/utils/utils.cpp > CMakeFiles/prolongation.dir/__/utils/utils.cpp.i

test/CMakeFiles/prolongation.dir/__/utils/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/prolongation.dir/__/utils/utils.cpp.s"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liaowang/Documents/master-thesis/code/utils/utils.cpp -o CMakeFiles/prolongation.dir/__/utils/utils.cpp.s

test/CMakeFiles/prolongation.dir/__/utils/HE_solution.cpp.o: test/CMakeFiles/prolongation.dir/flags.make
test/CMakeFiles/prolongation.dir/__/utils/HE_solution.cpp.o: ../utils/HE_solution.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liaowang/Documents/master-thesis/code/release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object test/CMakeFiles/prolongation.dir/__/utils/HE_solution.cpp.o"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/prolongation.dir/__/utils/HE_solution.cpp.o -c /home/liaowang/Documents/master-thesis/code/utils/HE_solution.cpp

test/CMakeFiles/prolongation.dir/__/utils/HE_solution.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/prolongation.dir/__/utils/HE_solution.cpp.i"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liaowang/Documents/master-thesis/code/utils/HE_solution.cpp > CMakeFiles/prolongation.dir/__/utils/HE_solution.cpp.i

test/CMakeFiles/prolongation.dir/__/utils/HE_solution.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/prolongation.dir/__/utils/HE_solution.cpp.s"
	cd /home/liaowang/Documents/master-thesis/code/release/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liaowang/Documents/master-thesis/code/utils/HE_solution.cpp -o CMakeFiles/prolongation.dir/__/utils/HE_solution.cpp.s

# Object files for target prolongation
prolongation_OBJECTS = \
"CMakeFiles/prolongation.dir/test_prolongation.cpp.o" \
"CMakeFiles/prolongation.dir/__/LagrangeO1/HE_LagrangeO1.cpp.o" \
"CMakeFiles/prolongation.dir/__/Pum_WaveRay/HE_FEM.cpp.o" \
"CMakeFiles/prolongation.dir/__/Pum_WaveRay/PUM_WaveRay.cpp.o" \
"CMakeFiles/prolongation.dir/__/Pum/HE_PUM.cpp.o" \
"CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_EdgeMat.cpp.o" \
"CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_EdgeVector.cpp.o" \
"CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_ElementMatrix.cpp.o" \
"CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_ElemVector.cpp.o" \
"CMakeFiles/prolongation.dir/__/ExtendPum/HE_ExtendPUM.cpp.o" \
"CMakeFiles/prolongation.dir/__/ExtendPum_WaveRay/ExtendPUM_WaveRay.cpp.o" \
"CMakeFiles/prolongation.dir/__/utils/utils.cpp.o" \
"CMakeFiles/prolongation.dir/__/utils/HE_solution.cpp.o"

# External object files for target prolongation
prolongation_EXTERNAL_OBJECTS =

test/prolongation: test/CMakeFiles/prolongation.dir/test_prolongation.cpp.o
test/prolongation: test/CMakeFiles/prolongation.dir/__/LagrangeO1/HE_LagrangeO1.cpp.o
test/prolongation: test/CMakeFiles/prolongation.dir/__/Pum_WaveRay/HE_FEM.cpp.o
test/prolongation: test/CMakeFiles/prolongation.dir/__/Pum_WaveRay/PUM_WaveRay.cpp.o
test/prolongation: test/CMakeFiles/prolongation.dir/__/Pum/HE_PUM.cpp.o
test/prolongation: test/CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_EdgeMat.cpp.o
test/prolongation: test/CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_EdgeVector.cpp.o
test/prolongation: test/CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_ElementMatrix.cpp.o
test/prolongation: test/CMakeFiles/prolongation.dir/__/ExtendPum/ExtendPUM_ElemVector.cpp.o
test/prolongation: test/CMakeFiles/prolongation.dir/__/ExtendPum/HE_ExtendPUM.cpp.o
test/prolongation: test/CMakeFiles/prolongation.dir/__/ExtendPum_WaveRay/ExtendPUM_WaveRay.cpp.o
test/prolongation: test/CMakeFiles/prolongation.dir/__/utils/utils.cpp.o
test/prolongation: test/CMakeFiles/prolongation.dir/__/utils/HE_solution.cpp.o
test/prolongation: test/CMakeFiles/prolongation.dir/build.make
test/prolongation: /home/liaowang/.hunter/_Base/6c9b2bc/252be92/d0f91ad/Install/lib/libboost_filesystem-mt-x64.a
test/prolongation: /home/liaowang/.hunter/_Base/6c9b2bc/252be92/d0f91ad/Install/lib/libboost_system-mt-x64.a
test/prolongation: /home/liaowang/.hunter/_Base/6c9b2bc/252be92/d0f91ad/Install/lib/libboost_program_options-mt-x64.a
test/prolongation: /home/liaowang/.hunter/_Base/6c9b2bc/252be92/d0f91ad/Install/lib/liblf.mesh.a
test/prolongation: /home/liaowang/.hunter/_Base/6c9b2bc/252be92/d0f91ad/Install/lib/liblf.mesh.utils.a
test/prolongation: /home/liaowang/.hunter/_Base/6c9b2bc/252be92/d0f91ad/Install/lib/liblf.mesh.test_utils.a
test/prolongation: /home/liaowang/.hunter/_Base/6c9b2bc/252be92/d0f91ad/Install/lib/liblf.mesh.hybrid2d.a
test/prolongation: /home/liaowang/.hunter/_Base/6c9b2bc/252be92/d0f91ad/Install/lib/liblf.refinement.a
test/prolongation: /home/liaowang/.hunter/_Base/6c9b2bc/252be92/d0f91ad/Install/lib/liblf.assemble.a
test/prolongation: /home/liaowang/.hunter/_Base/6c9b2bc/252be92/d0f91ad/Install/lib/liblf.io.a
test/prolongation: /home/liaowang/.hunter/_Base/6c9b2bc/252be92/d0f91ad/Install/lib/liblf.uscalfe.a
test/prolongation: /home/liaowang/.hunter/_Base/6c9b2bc/252be92/d0f91ad/Install/lib/liblf.base.a
test/prolongation: /home/liaowang/.hunter/_Base/6c9b2bc/252be92/d0f91ad/Install/lib/libgtest_main.a
test/prolongation: /home/liaowang/.hunter/_Base/6c9b2bc/252be92/d0f91ad/Install/lib/libgtest.a
test/prolongation: /home/liaowang/.hunter/_Base/6c9b2bc/252be92/d0f91ad/Install/lib/liblf.assemble.a
test/prolongation: /home/liaowang/.hunter/_Base/6c9b2bc/252be92/d0f91ad/Install/lib/liblf.mesh.a
test/prolongation: /home/liaowang/.hunter/_Base/6c9b2bc/252be92/d0f91ad/Install/lib/liblf.mesh.utils.a
test/prolongation: /home/liaowang/.hunter/_Base/6c9b2bc/252be92/d0f91ad/Install/lib/liblf.mesh.a
test/prolongation: /home/liaowang/.hunter/_Base/6c9b2bc/252be92/d0f91ad/Install/lib/liblf.mesh.utils.a
test/prolongation: /home/liaowang/.hunter/_Base/6c9b2bc/252be92/d0f91ad/Install/lib/liblf.geometry.a
test/prolongation: /home/liaowang/.hunter/_Base/6c9b2bc/252be92/d0f91ad/Install/lib/liblf.quad.a
test/prolongation: /home/liaowang/.hunter/_Base/6c9b2bc/252be92/d0f91ad/Install/lib/liblf.base.a
test/prolongation: /home/liaowang/.hunter/_Base/6c9b2bc/252be92/d0f91ad/Install/lib/libspdlog.a
test/prolongation: /home/liaowang/.hunter/_Base/6c9b2bc/252be92/d0f91ad/Install/lib/libfmt.a
test/prolongation: test/CMakeFiles/prolongation.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/liaowang/Documents/master-thesis/code/release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Linking CXX executable prolongation"
	cd /home/liaowang/Documents/master-thesis/code/release/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/prolongation.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/prolongation.dir/build: test/prolongation

.PHONY : test/CMakeFiles/prolongation.dir/build

test/CMakeFiles/prolongation.dir/clean:
	cd /home/liaowang/Documents/master-thesis/code/release/test && $(CMAKE_COMMAND) -P CMakeFiles/prolongation.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/prolongation.dir/clean

test/CMakeFiles/prolongation.dir/depend:
	cd /home/liaowang/Documents/master-thesis/code/release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/liaowang/Documents/master-thesis/code /home/liaowang/Documents/master-thesis/code/test /home/liaowang/Documents/master-thesis/code/release /home/liaowang/Documents/master-thesis/code/release/test /home/liaowang/Documents/master-thesis/code/release/test/CMakeFiles/prolongation.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/prolongation.dir/depend
