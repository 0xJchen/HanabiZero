# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_SOURCE_DIR = /home/game/revert/lsy/lst_best_confirm_copy/EfficientZero/envs/hanabi

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/game/revert/lsy/lst_best_confirm_copy/EfficientZero/envs/hanabi/build

# Include any dependencies generated for this target.
include CMakeFiles/game_example.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/game_example.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/game_example.dir/flags.make

CMakeFiles/game_example.dir/examples/game_example.cc.o: CMakeFiles/game_example.dir/flags.make
CMakeFiles/game_example.dir/examples/game_example.cc.o: ../examples/game_example.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/game/revert/lsy/lst_best_confirm_copy/EfficientZero/envs/hanabi/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/game_example.dir/examples/game_example.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/game_example.dir/examples/game_example.cc.o -c /home/game/revert/lsy/lst_best_confirm_copy/EfficientZero/envs/hanabi/examples/game_example.cc

CMakeFiles/game_example.dir/examples/game_example.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/game_example.dir/examples/game_example.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/game/revert/lsy/lst_best_confirm_copy/EfficientZero/envs/hanabi/examples/game_example.cc > CMakeFiles/game_example.dir/examples/game_example.cc.i

CMakeFiles/game_example.dir/examples/game_example.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/game_example.dir/examples/game_example.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/game/revert/lsy/lst_best_confirm_copy/EfficientZero/envs/hanabi/examples/game_example.cc -o CMakeFiles/game_example.dir/examples/game_example.cc.s

CMakeFiles/game_example.dir/examples/game_example.cc.o.requires:

.PHONY : CMakeFiles/game_example.dir/examples/game_example.cc.o.requires

CMakeFiles/game_example.dir/examples/game_example.cc.o.provides: CMakeFiles/game_example.dir/examples/game_example.cc.o.requires
	$(MAKE) -f CMakeFiles/game_example.dir/build.make CMakeFiles/game_example.dir/examples/game_example.cc.o.provides.build
.PHONY : CMakeFiles/game_example.dir/examples/game_example.cc.o.provides

CMakeFiles/game_example.dir/examples/game_example.cc.o.provides.build: CMakeFiles/game_example.dir/examples/game_example.cc.o


# Object files for target game_example
game_example_OBJECTS = \
"CMakeFiles/game_example.dir/examples/game_example.cc.o"

# External object files for target game_example
game_example_EXTERNAL_OBJECTS =

../examples/game_example: CMakeFiles/game_example.dir/examples/game_example.cc.o
../examples/game_example: CMakeFiles/game_example.dir/build.make
../examples/game_example: hanabi_lib/libhanabi.a
../examples/game_example: CMakeFiles/game_example.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/game/revert/lsy/lst_best_confirm_copy/EfficientZero/envs/hanabi/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../examples/game_example"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/game_example.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/game_example.dir/build: ../examples/game_example

.PHONY : CMakeFiles/game_example.dir/build

CMakeFiles/game_example.dir/requires: CMakeFiles/game_example.dir/examples/game_example.cc.o.requires

.PHONY : CMakeFiles/game_example.dir/requires

CMakeFiles/game_example.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/game_example.dir/cmake_clean.cmake
.PHONY : CMakeFiles/game_example.dir/clean

CMakeFiles/game_example.dir/depend:
	cd /home/game/revert/lsy/lst_best_confirm_copy/EfficientZero/envs/hanabi/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/game/revert/lsy/lst_best_confirm_copy/EfficientZero/envs/hanabi /home/game/revert/lsy/lst_best_confirm_copy/EfficientZero/envs/hanabi /home/game/revert/lsy/lst_best_confirm_copy/EfficientZero/envs/hanabi/build /home/game/revert/lsy/lst_best_confirm_copy/EfficientZero/envs/hanabi/build /home/game/revert/lsy/lst_best_confirm_copy/EfficientZero/envs/hanabi/build/CMakeFiles/game_example.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/game_example.dir/depend

