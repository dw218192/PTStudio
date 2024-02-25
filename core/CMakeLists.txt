cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(lib_name "core")
set(core_util "core_util")
set(src_dir "${CMAKE_CURRENT_SOURCE_DIR}/src")
#GLOB source files 
file(GLOB_RECURSE src_files
	${src_dir}/*.*
	${src_dir}/imgui/*.*
	${src_dir}/include/*.*
	${src_dir}/include/imgui/*.*
)

#GLOB resource files
file(GLOB_RECURSE res_files
	${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.*
)

#Whole library
add_library(${lib_name} STATIC ${src_files})

#Util library to bundle all the utility libraries
add_library(${core_util} INTERFACE)
add_library(${lib_name}::util ALIAS ${core_util})
target_link_libraries(${core_util} INTERFACE
	$<BUILD_INTERFACE:developer_flags>
	expected
	span
	fmt-header-only
	nlohmann_json
	glm
)

#Sub-modules
function(create_core_submodule module_name)
	set(module_src_path "${src_dir}/${module_name}")
	set(module_include_path "${src_dir}/include/${module_name}")
	file(GLOB module_sources "${module_src_path}/*.cpp")
	file(GLOB module_headers "${module_include_path}/*.hpp" "${module_include_path}/*.h")


	add_library(core_${module_name} ${module_sources} ${module_headers})
	add_library(core::${module_name} ALIAS core_${module_name})
	target_include_directories(core_${module_name} PUBLIC
		${module_include_path}
		${src_dir}/include
	)
	target_link_libraries(core_${module_name} PUBLIC
		$<BUILD_INTERFACE:developer_flags>
		${core_util}
	)
endfunction()

# automatically create submodules
# for any folder in src
file(GLOB submodules "${src_dir}/*")
foreach(submodule ${submodules})
	if(IS_DIRECTORY ${submodule})
		get_filename_component(submodule_name ${submodule} NAME)
		# skip include itself
		if(${submodule_name} STREQUAL "include")
			continue()
		endif()
		create_core_submodule(${submodule_name})
		message(STATUS "Created submodule: ${lib_name}::${submodule_name}")
	endif()
endforeach()

#Resources
cmrc_add_resource_library(${lib_name}_resources
	ALIAS ${lib_name}::rc
	${res_files}
)

#Libs linked to the executable
target_link_libraries(${lib_name}
PUBLIC
	$<BUILD_INTERFACE:developer_flags>
	${lib_name}::rc
	${core_util}
	tinyobjloader
	stb
	imgui
	nativefiledialog
)

set(PT_CORE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/include PARENT_SCOPE)

target_include_directories(${lib_name} PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/src/include
)

# Build unit tests if enabled
if (CORE_BUILD_TESTS)
	add_subdirectory(tests)
endif()