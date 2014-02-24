#include "volumes/box.h"
#include "management/volume_factory.h"

namespace vecgeom {


VPlacedVolume* VolumeFactory::CreateSpecializedVolume(
    LogicalVolume const &logical_volume,
    TransformationMatrix const &matrix) const {

  const TranslationCode trans_code = matrix.GenerateTranslationCode();
  const RotationCode rot_code = matrix.GenerateRotationCode();

  // All shapes must be implemented here. Better solution?

  VPlacedVolume *placed = NULL;

  if (UnplacedBox const *const box =
      dynamic_cast<UnplacedBox const *const>(
        &logical_volume.unplaced_volume()
      )) {
    placed = CreateByTransformation<SpecializedBox>(logical_volume, matrix,
                                                    trans_code, rot_code);
  }

  // Will return null if the passed shape isn't implemented here. Maybe throw an
  // exception instead?

  return placed;

}

template<typename VolumeType>
VPlacedVolume* VolumeFactory::CreateByTransformation(
    LogicalVolume const &logical_volume, TransformationMatrix const &matrix,
    const TranslationCode trans_code, const RotationCode rot_code) const {

  if (trans_code == 0 && rot_code == 0x1b1) {
    return Create<VolumeType, 0, 0x1b1>(unplaced, matrix);
  }
  if (trans_code == 1 && rot_code == 0x1b1) {
    return Create<VolumeType, 1, 0x1b1>(unplaced, matrix);
  }
  if (trans_code == 0 && rot_code == 0x18e) {
    return Create<VolumeType, 0, 0x18e>(unplaced, matrix);
  }
  if (trans_code == 1 && rot_code == 0x18e) {
    return Create<VolumeType, 1, 0x18e>(unplaced, matrix);
  }
  if (trans_code == 0 && rot_code == 0x076) {
    return Create<VolumeType, 0, 0x076>(unplaced, matrix);
  }
  if (trans_code == 1 && rot_code == 0x076) {
    return Create<VolumeType, 1, 0x076>(unplaced, matrix);
  }
  if (trans_code == 0 && rot_code == 0x16a) {
    return Create<VolumeType, 0, 0x16a>(unplaced, matrix);
  }
  if (trans_code == 1 && rot_code == 0x16a) {
    return Create<VolumeType, 1, 0x16a>(unplaced, matrix);
  }
  if (trans_code == 0 && rot_code == 0x155) {
    return Create<VolumeType, 0, 0x155>(unplaced, matrix);
  }
  if (trans_code == 1 && rot_code == 0x155) {
    return Create<VolumeType, 1, 0x155>(unplaced, matrix);
  }
  if (trans_code == 0 && rot_code == 0x0ad) {
    return Create<VolumeType, 0, 0x0ad>(unplaced, matrix);
  }
  if (trans_code == 1 && rot_code == 0x0ad) {
    return Create<VolumeType, 1, 0x0ad>(unplaced, matrix);
  }
  if (trans_code == 0 && rot_code == 0x0dc) {
    return Create<VolumeType, 0, 0x0dc>(unplaced, matrix);
  }
  if (trans_code == 1 && rot_code == 0x0dc) {
    return Create<VolumeType, 1, 0x0dc>(unplaced, matrix);
  }
  if (trans_code == 0 && rot_code == 0x0e3) {
    return Create<VolumeType, 0, 0x0e3>(unplaced, matrix);
  }
  if (trans_code == 1 && rot_code == 0x0e3) {
    return Create<VolumeType, 1, 0x0e3>(unplaced, matrix);
  }
  if (trans_code == 0 && rot_code == 0x11b) {
    return Create<VolumeType, 0, 0x11b>(unplaced, matrix);
  }
  if (trans_code == 1 && rot_code == 0x11b) {
    return Create<VolumeType, 1, 0x11b>(unplaced, matrix);
  }
  if (trans_code == 0 && rot_code == 0x0a1) {
    return Create<VolumeType, 0, 0x0a1>(unplaced, matrix);
  }
  if (trans_code == 1 && rot_code == 0x0a1) {
    return Create<VolumeType, 1, 0x0a1>(unplaced, matrix);
  }
  if (trans_code == 0 && rot_code == 0x10a) {
    return Create<VolumeType, 0, 0x10a>(unplaced, matrix);
  }
  if (trans_code == 1 && rot_code == 0x10a) {
    return Create<VolumeType, 1, 0x10a>(unplaced, matrix);
  }
  if (trans_code == 0 && rot_code == 0x046) {
    return Create<VolumeType, 0, 0x046>(unplaced, matrix);
  }
  if (trans_code == 1 && rot_code == 0x046) {
    return Create<VolumeType, 1, 0x046>(unplaced, matrix);
  }
  if (trans_code == 0 && rot_code == 0x062) {
    return Create<VolumeType, 0, 0x062>(unplaced, matrix);
  }
  if (trans_code == 1 && rot_code == 0x062) {
    return Create<VolumeType, 1, 0x062>(unplaced, matrix);
  }
  if (trans_code == 0 && rot_code == 0x054) {
    return Create<VolumeType, 0, 0x054>(unplaced, matrix);
  }
  if (trans_code == 1 && rot_code == 0x054) {
    return Create<VolumeType, 1, 0x054>(unplaced, matrix);
  }
  if (trans_code == 0 && rot_code == 0x111) {
    return Create<VolumeType, 0, 0x111>(unplaced, matrix);
  }
  if (trans_code == 1 && rot_code == 0x111) {
    return Create<VolumeType, 1, 0x111>(unplaced, matrix);
  }
  if (trans_code == 0 && rot_code == 0x200) {
    return Create<VolumeType, 0, 0x200>(unplaced, matrix);
  }
  if (trans_code == 1 && rot_code == 0x200) {
    return Create<VolumeType, 1, 0x200>(unplaced, matrix);
  }

  // No specialization
  return Create<1, 0>(unplaced, matrix);

}

} // End namespace vecgeom