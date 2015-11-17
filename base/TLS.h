#ifndef BASE_TLS
#define BASE_TLS

#if defined(__APPLE__)       /* MacOS X support, initially following FreeBSD */
#   include <AvailabilityMacros.h>
#   define R__MACOSX
#   if defined(__clang__) && defined(MAC_OS_X_VERSION_10_7) && (defined(__x86_64__) || defined(__i386__))
#     define R__HAS___THREAD
#   elif !defined(R__HAS_PTHREAD)
#     define R__HAS_PTHREAD
#   endif
#endif // __APPLE__

#if defined(linux) && defined(__i386__)
#   define R__LINUX
#endif

#ifdef _AIX
#   define R__AIX
#endif

#if defined(R__LINUX) || defined(R__AIX)
#  define R__HAS___THREAD
#endif

#if defined(__FCC_VERSION)    /* Solaris with Fujitsu compiler */
#   define R__SOLARIS
#endif

#if defined(__sun) && defined(__SVR4) && !(defined(linux) || defined(__FCC_VERSION))
#   define R__SOLARIS
#endif

#if defined(R__SOLARIS) && !defined(R__HAS_PTHREAD)
#  define R__HAS_PTHREAD
#endif

#ifdef _WIN32
#   define R__WIN32
#   define R__HAS_DECLSPEC_THREAD
#   define thread_local static __declspec(thread)
#endif

// Clang 3.4 also support SD-6 (feature test macros __cpp_*), but no thread local macro
#  if defined(__clang__)

#    if __has_feature(cxx_thread_local)
     // thread_local was added in Clang 3.3
     // Still requires libstdc++ from GCC 4.8
     // For that __GLIBCXX__ isn't good enough
     // Also the MacOS build of clang does *not* support thread local yet.
#      define R__HAS_THREAD_LOCAL
#    else
#      define R__HAS___THREAD
#    endif

#  elif defined(__GNUG__) && (__GNUC__ <= 4 && __GNUC_MINOR__ < 8)
    // The C++11 thread_local keyword is supported in GCC only since 4.8
#    define R__HAS___THREAD
#  else
#    define R__HAS_THREAD_LOCAL
#  endif // __clang__

#ifdef __cplusplus
// Note that the order is relevant, more than one of the flag might be
// on at the same time and we want to use 'best' option available.
#ifdef R__HAS_THREAD_LOCAL
#  define THREAD_TLS(type) thread_local type
#  pragma message("TLS using thread_local")

#elif defined(R__HAS___THREAD)
#  define THREAD_TLS(type)  static __thread type
#  pragma message("TLS using static __thread")

#elif defined(R__HAS_DECLSPEC_THREAD)
#  define THREAD_TLS(type) static __declspec(thread) type
#  pragma message("TLS using static __declspec(thread)")

#elif defined(R__HAS_PTHREAD)

#include <assert.h>
#include <pthread.h>
template <typename type> class ThreadTLSWrapper
{
private:
   pthread_key_t  fKey;
   type           fInitValue;

   static void key_delete(void *arg) {
      assert (NULL != arg);
      delete (type*)(arg);
   }

public:

   ThreadTLSWrapper() : fInitValue() {

      pthread_key_create(&(fKey), key_delete);
   }

   ThreadTLSWrapper(const type &value) : fInitValue(value) {

      pthread_key_create(&(fKey), key_delete);
   }

   ~ThreadTLSWrapper() {
      pthread_key_delete(fKey);
   }

   type& get() {
      void *ptr = pthread_getspecific(fKey);
      if (!ptr) {
         ptr = new type(fInitValue);
         assert (NULL != ptr);
         (void) pthread_setspecific(fKey, ptr);
      }
      return *(type*)ptr;
   }

   type& operator=(const type &in) {
      type &ptr = get();
      ptr = in;
      return ptr;
   }

   operator type&() {
      return get();
   }

};

#  define THREAD_TLS(type) static TThreadTLSWrapper<type>
#  pragma message("TLS using pthread wrapper")

#else

#error "No Thread Local Storage (TLS) technology for this platform specified."

#endif

#endif // __cplusplus

#include <atomic>

inline namespace BaseTLS {

  static int gNumThreads = 0;
  static std::atomic_flag gLock(ATOMIC_FLAG_INIT);

  inline int ThreadId() {
    // Utility getting a thread local id
#ifdef VECGEOM_ROOT
   TTHREAD_TLS(Int_t) tid = -1;
#else
   THREAD_TLS(int) tid = -1;
#endif 
    int ttid = tid;
    if (ttid > -1) return ttid;
    while (BaseTLS::gLock.test_and_set(std::memory_order_acquire))
      ;
    tid = gNumThreads;
    ttid = gNumThreads++;
    BaseTLS::gLock.clear(std::memory_order_release);
    return ttid;
  }
  
  inline void ClearThreadId() {
    BaseTLS::gNumThreads = 0;
  }  
};

#endif // BASE_TLS
