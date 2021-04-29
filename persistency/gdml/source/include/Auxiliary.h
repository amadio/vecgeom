//! \file Auxiliary.h

#ifndef VGDMLAuxiliary_h
#define VGDMLAuxiliary_h

#include <string>
#include <memory>
#include <vector>

namespace vgdml {

/**
 * \brief Class representing a GDML auxiliary tag
 *
 * The GDML schema provides the `<auxiliary/>` tag to allow attachment of additional,
 * user specified data to . It has the general form in XML
 *
 * ```xml
 * <auxiliary auxtype="foo" auxvalue="bar" auxunit="optional">
 *   <!-- Zero to N child auxiliaries -->
 *   <auxiliary ...>
 *   ...
 * </auxiliary>
 * ```
 *
 * Auxiliary tags may be added either as child nodes of the `<volume>` or `<userinfo>` tags.
 *
 * In this class, the attributes of the auxiliary tags are stored as `std::string` and it is left to
 * the client to interpret their values as appropriate. The list of child auxiliary tags, if any, are
 * exposed as a vector and again left to the client interpret this subtree and the leaf auxiliary tags
 * and attributes.
 */
class Auxiliary {
public:
  /// Convenience type alias for collection of child Auxiliary tags
  using AuxiliaryList = std::vector<Auxiliary>;

  /// Default constructor
  /// Initializes all attributes to the empty string, with zero child nodes
  Auxiliary() = default;

  /// Destructor
  ~Auxiliary() = default;

  /// Copy constructor
  Auxiliary(const Auxiliary &rhs)
      : type(rhs.type), value(rhs.value), unit(rhs.unit), children(new AuxiliaryList{*(rhs.children)})
  {
  }

  /// Move Constructor
  Auxiliary(Auxiliary &&rhs) : Auxiliary() { swap(*this, rhs); }

  /// Copy assignment operator
  Auxiliary &operator=(const Auxiliary &rhs)
  {
    Auxiliary tmp(rhs);
    swap(*this, tmp);
    return *this;
  }

  /// Move assignment operator
  Auxiliary &operator=(Auxiliary &&rhs)
  {
    swap(*this, rhs);
    return *this;
  }

  /// Return the value of the auxiliary `type` attribute
  const std::string &GetType() const { return type; }

  /// Return the value of the auxiliary `value` attribute
  const std::string &GetValue() const { return value; }

  /// Return the value of the auxiliary `unit` attribute
  const std::string &GetUnit() const { return unit; }

  /// Return the collection of child auxiliary tags
  const AuxiliaryList &GetChildren() const { return *children; }

  /// Set the value for the `type` attribute
  void SetType(const std::string &s) { type = s; }

  /// Set the value for the `value` attribute
  void SetValue(const std::string &s) { value = s; }

  /// Set the value for the `unit` attribute
  void SetUnit(const std::string &s) { unit = s; }

  /// Add an auxiliary object as a child of this one
  void AddChild(const Auxiliary &a) { children->push_back(a); }

  /// Implementation of swap to enable copy-and-swap
  /// Copy-and-swap used as simplest way to copy/move the held unique_ptr
  /// whilst ensuring moved-from object is left with a new, empty auxiliary
  /// list. This is held in a unique_ptr to avoid an incomplete type.
  friend void swap(Auxiliary &lhs, Auxiliary &rhs)
  {
    using std::swap;
    swap(lhs.type, rhs.type);
    swap(lhs.value, rhs.value);
    swap(lhs.unit, rhs.unit);
    swap(lhs.children, rhs.children);
  }

private:
  std::string type  = "";                                     ///< value of `type` attribute
  std::string value = "";                                     ///< value of `value` attribute
  std::string unit  = "";                                     ///< value of `unit` attribute
  std::unique_ptr<AuxiliaryList> children{new AuxiliaryList}; ///< collection of child Auxiliary tags
};

/// Return true if input Auxiliary instances are equal
/// Equality means that all attributes are equal, and that both instances
/// have the same child nodes all the way down the tree.
inline bool operator==(const Auxiliary &lhs, const Auxiliary &rhs)
{
  if (lhs.GetType() != rhs.GetType()) return false;
  if (lhs.GetValue() != rhs.GetValue()) return false;
  if (lhs.GetUnit() != rhs.GetUnit()) return false;
  if (lhs.GetChildren() != rhs.GetChildren()) return false;

  return true;
}

/// Return true if input Auxiliary instances are not equal
inline bool operator!=(const Auxiliary &lhs, const Auxiliary &rhs)
{
  return !(lhs == rhs);
}

} // namespace vgdml

#endif