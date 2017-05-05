// modified from stl map
// https://gcc.gnu.org/onlinedocs/gcc-4.9.3/libstdc++/api/a01255.html
//
// removed allocators
// removed inverted iterators
// added CUDA annotations
// added select1st struct

#ifndef VECCORE_MAP_H
#define VECCORE_MAP_H

#include "RBTree.h"
namespace vecgeom {
// This fails because of the commas, we would need to use another (new) macro
// VECGEOM_DEVICE_FORWARD_DECLARE(template <class _Key, class _Tp, class _Compare>  class map; );
#ifndef VECCORE_CUDA
namespace cuda {
template <class _key>
struct less;
template <class _Key, class _Tp, class _Compare = cuda::less<_Key>>
class map;
}
#endif

inline namespace VECGEOM_IMPL_NAMESPACE {

template <typename P>
struct select1st {
  VECCORE_ATT_HOST_DEVICE
  typename P::first_type const &operator()(P const &p) const { return p.first; }
};

template <class _Key, class _Tp, class _Compare = less<_Key>>
class map;

template <class _Key, class _Tp, class _Compare>
VECCORE_ATT_HOST_DEVICE
inline bool operator==(const map<_Key, _Tp, _Compare> &__x, const map<_Key, _Tp, _Compare> &__y);

template <class _Key, class _Tp, class _Compare>
VECCORE_ATT_HOST_DEVICE
inline bool operator<(const map<_Key, _Tp, _Compare> &__x, const map<_Key, _Tp, _Compare> &__y);

template <class _Key, class _Tp, class _Compare>
class map {
public:
  // typedefs:

  typedef _Key key_type;
  typedef _Tp data_type;
  typedef _Tp mapped_type;
  typedef vecgeom::pair<_Key, _Tp> value_type;
  typedef _Compare key_compare;

  template <class Key, class T, class Compare>
  class value_compare {
    friend class map;

  protected:
    Compare comp;
    VECCORE_ATT_HOST_DEVICE
    value_compare(Compare c) : comp(c) {}
  public:
    typedef bool result_type;
    typedef value_type first_argument_type;
    typedef value_type second_argument_type;
    VECCORE_ATT_HOST_DEVICE
    bool operator()(const value_type &x, const value_type &y) const { return comp(x.first, y.first); }
  };

private:
  typedef _Rb_tree<key_type, value_type, select1st<value_type>, key_compare> _Rep_type;
  // std::_Select1st<value_type>, key_compare> _Rep_type;
  _Rep_type _M_t; // red-black tree representing map
public:
  typedef typename _Rep_type::pointer pointer;
  typedef typename _Rep_type::const_pointer const_pointer;
  typedef typename _Rep_type::reference reference;
  typedef typename _Rep_type::const_reference const_reference;
  typedef typename _Rep_type::iterator iterator;
  typedef typename _Rep_type::const_iterator const_iterator;
  // typedef typename _Rep_type::reverse_iterator reverse_iterator;
  // typedef typename _Rep_type::const_reverse_iterator const_reverse_iterator;
  typedef typename _Rep_type::size_type size_type;
  typedef typename _Rep_type::difference_type difference_type;

  // Constructors
  VECCORE_ATT_HOST_DEVICE
  map() : _M_t(_Compare()) {}
  VECCORE_ATT_HOST_DEVICE
  map(const key_type /*_key*/) : _M_t() {} // TEST
  VECCORE_ATT_HOST_DEVICE
  map(const value_type *__first, const value_type *__last) : _M_t(_Compare()) { _M_t.insert_unique(__first, __last); }
  VECCORE_ATT_HOST_DEVICE
  map(const value_type *__first, const value_type *__last, const _Compare &__comp) : _M_t(__comp)
  {
    _M_t.insert_unique(__first, __last);
  }
  VECCORE_ATT_HOST_DEVICE
  map(const_iterator __first, const_iterator __last) : _M_t(_Compare()) { _M_t.insert_unique(__first, __last); }
  VECCORE_ATT_HOST_DEVICE
  map(const_iterator __first, const_iterator __last, const _Compare &__comp) : _M_t(__comp)
  {
    _M_t.insert_unique(__first, __last);
  }

  VECCORE_ATT_HOST_DEVICE
  map(const map<_Key, _Tp, _Compare> &__x) : _M_t(__x._M_t) {}
  VECCORE_ATT_HOST_DEVICE
  map<_Key, _Tp, _Compare> &operator=(const map<_Key, _Tp, _Compare> &__x)
  {
    _M_t = __x._M_t;
    return *this;
  }

  // key/value compare funtions
  VECCORE_ATT_HOST_DEVICE
  key_compare key_comp() const { return _M_t.key_comp(); }
  VECCORE_ATT_HOST_DEVICE
  value_compare<_Key, _Tp, _Compare> value_comp() const { return value_compare<_Key, _Tp, _Compare>(_M_t.key_comp()); }

  // iterators
  VECCORE_ATT_HOST_DEVICE
  iterator begin() { return _M_t.begin(); }
  VECCORE_ATT_HOST_DEVICE
  const_iterator begin() const { return _M_t.begin(); }
  VECCORE_ATT_HOST_DEVICE
  iterator end() { return _M_t.end(); }
  VECCORE_ATT_HOST_DEVICE
  const_iterator end() const { return _M_t.end(); }
  /*
    reverse_iterator rbegin() { return _M_t.rbegin(); }
    const_reverse_iterator rbegin() const { return _M_t.rbegin(); }
    reverse_iterator rend() { return _M_t.rend(); }
    const_reverse_iterator rend() const { return _M_t.rend(); }
  */
  VECCORE_ATT_HOST_DEVICE
  bool empty() const { return _M_t.empty(); }
  VECCORE_ATT_HOST_DEVICE
  size_type size() const { return _M_t.size(); }
  VECCORE_ATT_HOST_DEVICE
  size_type max_size() const { return _M_t.max_size(); }
  VECCORE_ATT_HOST_DEVICE
  _Tp &operator[](const key_type &__k)
  {
    iterator __i = lower_bound(__k);
    // __i->first is greater than or equivalent to __k.
    if (__i == end() || key_comp()(__k, (*__i).first)) __i = insert(__i, value_type(__k, _Tp()));
    return (*__i).second;
  }
  VECCORE_ATT_HOST_DEVICE
  const _Tp &at(const key_type &__k) const
  {
    const_iterator __i = lower_bound(__k);
    // __i->first is greater than or equivalent to __k.
    if (__i == end() || key_comp()(__k, (*__i).first)) {
      printf("at(): key out of range \n");
    }
    return (*__i).second;
  }
  // void swap(map<_Key,_Tp,_Compare,_Alloc>& __x) { _M_t.swap(__x._M_t); }

  // insert/erase
  VECCORE_ATT_HOST_DEVICE
  pair<iterator, bool> insert(const value_type &__x) { return _M_t.insert_unique(__x); }
  VECCORE_ATT_HOST_DEVICE
  iterator insert(iterator position, const value_type &__x) { return _M_t.insert_unique(position, __x); }
  VECCORE_ATT_HOST_DEVICE
  void insert(const value_type *__first, const value_type *__last) { _M_t.insert_unique(__first, __last); }
  VECCORE_ATT_HOST_DEVICE
  void insert(const_iterator __first, const_iterator __last) { _M_t.insert_unique(__first, __last); }

  VECCORE_ATT_HOST_DEVICE
  void erase(iterator __position) { _M_t.erase(__position); }
  VECCORE_ATT_HOST_DEVICE
  size_type erase(const key_type &__x) { return _M_t.erase(__x); }
  VECCORE_ATT_HOST_DEVICE
  void erase(iterator __first, iterator __last) { _M_t.erase(__first, __last); }
  VECCORE_ATT_HOST_DEVICE
  void clear() { _M_t.clear(); }

  // map operations:
  VECCORE_ATT_HOST_DEVICE
  iterator find(const key_type &__x) { return _M_t.find(__x); }
  VECCORE_ATT_HOST_DEVICE
  const_iterator find(const key_type &__x) const { return _M_t.find(__x); }
  VECCORE_ATT_HOST_DEVICE
  size_type count(const key_type &__x) const { return _M_t.find(__x) == _M_t.end() ? 0 : 1; }
  VECCORE_ATT_HOST_DEVICE
  iterator lower_bound(const key_type &__x) { return _M_t.lower_bound(__x); }
  VECCORE_ATT_HOST_DEVICE
  const_iterator lower_bound(const key_type &__x) const { return _M_t.lower_bound(__x); }
  VECCORE_ATT_HOST_DEVICE
  iterator upper_bound(const key_type &__x) { return _M_t.upper_bound(__x); }
  VECCORE_ATT_HOST_DEVICE
  const_iterator upper_bound(const key_type &__x) const { return _M_t.upper_bound(__x); }
  VECCORE_ATT_HOST_DEVICE
  pair<iterator, iterator> equal_range(const key_type &__x) { return _M_t.equal_range(__x); }
  VECCORE_ATT_HOST_DEVICE
  pair<const_iterator, const_iterator> equal_range(const key_type &__x) const { return _M_t.equal_range(__x); }
};

template <class _Key, class _Tp, class _Compare>
VECCORE_ATT_HOST_DEVICE
inline bool operator==(const map<_Key, _Tp, _Compare> &__x, const map<_Key, _Tp, _Compare> &__y)
{
  return __x._M_t == __y._M_t;
}

template <class _Key, class _Tp, class _Compare>
VECCORE_ATT_HOST_DEVICE
inline bool operator<(const map<_Key, _Tp, _Compare> &__x, const map<_Key, _Tp, _Compare> &__y)
{
  return __x._M_t < __y._M_t;
}

template <class _Key, class _Tp, class _Compare>
VECCORE_ATT_HOST_DEVICE
inline bool operator!=(const map<_Key, _Tp, _Compare> &__x, const map<_Key, _Tp, _Compare> &__y)
{
  return !(__x == __y);
}

template <class _Key, class _Tp, class _Compare>
VECCORE_ATT_HOST_DEVICE
inline bool operator>(const map<_Key, _Tp, _Compare> &__x, const map<_Key, _Tp, _Compare> &__y)
{
  return __y < __x;
}

template <class _Key, class _Tp, class _Compare>
VECCORE_ATT_HOST_DEVICE
inline bool operator<=(const map<_Key, _Tp, _Compare> &__x, const map<_Key, _Tp, _Compare> &__y)
{
  return !(__y < __x);
}

template <class _Key, class _Tp, class _Compare>
VECCORE_ATT_HOST_DEVICE
inline bool operator>=(const map<_Key, _Tp, _Compare> &__x, const map<_Key, _Tp, _Compare> &__y)
{
  return !(__x < __y);
}
/*
template <class _Key, class _Tp, class _Compare>
inline void swap(map<_Key,_Tp,_Compare>& __x,
                 map<_Key,_Tp,_Compare>& __y) {
  __x.swap(__y);
}
*/
}
}
#endif
