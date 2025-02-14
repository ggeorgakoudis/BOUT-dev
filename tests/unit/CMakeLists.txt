bout_update_submodules()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/externalpackages/googletest/CMakeLists.txt")
  set(BOUT_ENABLE_UNIT_TESTS OFF PARENT_SCOPE)
  message(WARNING "googletest not found! Have you disabled the git submodules (GIT_SUBMODULE)?")
  return()
endif()

set(BOUT_ENABLE_UNIT_TESTS ON PARENT_SCOPE)

# disable gtest pthreads for proper execution of Death tests (on some platforms)
add_definitions(-DGTEST_HAS_PTHREAD=0)
set(gtest_disable_pthreads ON CACHE BOOL "" FORCE)

add_subdirectory("${PROJECT_SOURCE_DIR}/externalpackages/googletest"
  "externalpackages/googletest")

if(NOT TARGET gtest)
  message(FATAL_ERROR "googletest not found! Have you disabled the git submodules (GIT_SUBMODULE)?")
endif()

# Some unit tests require GMOCK, so make sure we build it
set(BUILD_GMOCK ON)

mark_as_advanced(
  BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
  gmock_build_tests gtest_build_samples gtest_build_tests
  gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
  )

set(serial_tests_source
  ./bout_test_main.cxx
  ./field/test_field.cxx
  ./field/test_field2d.cxx
  ./field/test_field3d.cxx
  ./field/test_field_factory.cxx
  ./field/test_fieldgroup.cxx
  ./field/test_fieldperp.cxx
  ./field/test_initialprofiles.cxx
  ./field/test_vector2d.cxx
  ./field/test_vector3d.cxx
  ./field/test_where.cxx
  ./include/bout/test_array.cxx
  ./include/bout/test_assert.cxx
  ./include/bout/test_bout_enum_class.cxx
  ./include/bout/test_deriv_store.cxx
  ./include/bout/test_generic_factory.cxx
  ./include/bout/test_macro_for_each.cxx
  ./include/bout/test_monitor.cxx
  ./include/bout/test_petsc_indexer.cxx
  ./include/bout/test_petsc_matrix.cxx
  ./include/bout/test_petsc_setters.cxx
  ./include/bout/test_petsc_vector.cxx
  ./include/bout/test_region.cxx
  ./include/bout/test_single_index_ops.cxx
  ./include/bout/test_stencil.cxx
  ./include/bout/test_template_combinations.cxx
  ./include/bout/test_traits.cxx
  ./include/test_cyclic_reduction.cxx
  ./include/test_derivs.cxx
  ./include/test_mask.cxx
  ./invert/test_fft.cxx
  ./invert/laplace/test_laplace_petsc3damg.cxx
  ./invert/laplace/test_laplace_cyclic.cxx
  ./mesh/data/test_gridfromoptions.cxx
  ./mesh/parallel/test_shiftedmetric.cxx
  ./mesh/test_boundary_factory.cxx
  ./mesh/test_boutmesh.cxx
  ./mesh/test_coordinates.cxx
  ./mesh/test_coordinates_accessor.cxx
  ./mesh/test_interpolation.cxx
  ./mesh/test_mesh.cxx
  ./mesh/test_paralleltransform.cxx
  ./solver/test_fakesolver.cxx
  ./solver/test_fakesolver.hxx
  ./solver/test_solver.cxx
  ./solver/test_solverfactory.cxx
  ./sys/test_boutexception.cxx
  ./sys/test_expressionparser.cxx
  ./sys/test_msg_stack.cxx
  ./sys/test_options.cxx
  ./sys/test_options_fields.cxx
  ./sys/test_options_netcdf.cxx
  ./sys/test_optionsreader.cxx
  ./sys/test_output.cxx
  ./sys/test_range.cxx
  ./sys/test_timer.cxx
  ./sys/test_type_name.cxx
  ./sys/test_utils.cxx
  ./sys/test_variant.cxx
  ./sys/test_raja.cxx
  ./test_extras.cxx
  ./test_extras.hxx
  ./src/test_bout++.cxx)

if(BOUT_HAS_HYPRE)
   list(APPEND serial_tests_source ./include/bout/test_hypre_interface.cxx)
   list(APPEND serial_tests_source ./invert/laplace/test_laplace_hypre3d.cxx)
endif ()

add_executable(serial_tests ${serial_tests_source})

target_include_directories(serial_tests PUBLIC bout++
   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
   $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../../include>
   ${CMAKE_INSTALL_PREFIX}/include
  )
set_target_properties(serial_tests PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(serial_tests gtest gmock bout++::bout++)
add_test(NAME serial_tests COMMAND serial_tests --gtest_brief=1)
set_target_properties(serial_tests PROPERTIES FOLDER tests/unit)
add_dependencies(build-check-unit-tests serial_tests)

if (BOUT_HAS_CUDA)
   set_source_files_properties(${serial_tests_source} PROPERTIES LANGUAGE CUDA )
   set_target_properties(serial_tests PROPERTIES CUDA_STANDARD 14)
endif ()   

