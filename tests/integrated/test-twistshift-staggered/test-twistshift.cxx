#include "bout/bout.hxx"
#include "bout/field_factory.hxx"

int main(int argc, char** argv) {
  BoutInitialise(argc, argv);

  using bout::globals::mesh;

  Field3D test = FieldFactory::get()->create3D("test", nullptr, mesh, CELL_YLOW);

  Field3D test_aligned = toFieldAligned(test);

  // zero guard cells to check that communication is doing something
  for (int x = 0; x < mesh->LocalNx; x++) {
    for (int z = 0; z < mesh->LocalNz; z++) {
      for (int y = 0; y < mesh->ystart; y++) {
        test_aligned(x, y, z) = 0.;
      }
      for (int y = mesh->yend + 1; y < mesh->LocalNy; y++) {
        test_aligned(x, y, z) = 0.;
      }
    }
  }

  mesh->communicate(test_aligned);

  Options::root()["check"] =
      FieldFactory::get()->create3D("check", nullptr, mesh, CELL_YLOW);

  Options::root()["test"] = test;
  Options::root()["test_aligned"] = test_aligned;

  bout::writeDefaultOutputFile(Options::root());

  BoutFinalise();
}
