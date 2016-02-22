// first version of file added by Federico.Carminati@cern.ch
#ifndef MESSAGELOGGER_H
#define MESSAGELOGGER_H

#include "base/Global.h"
#include <stdio.h>
#include <stdarg.h>
#include <string>
#include <mutex>
#include <iostream>
#ifndef VECGEOM_NVCC
#include <map>
using std::map;
#else
#include "base/Map.h"
using vecgeom::map;
#endif
using std::string;
using std::mutex;
using std::endl;

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

  /**
   * a simple singleton message logger class
   */
   class MessageLogger {
   public:
      
      typedef enum {kInfo=0, kWarning, kError, kFatal, kDebug} logging_severity;
      
      static MessageLogger* I() {
        static mutex mtx;
        mtx.lock();
        if (!gMessageLogger)
          gMessageLogger = new MessageLogger();
        mtx.unlock();
        return gMessageLogger;
      }
      std::ostream &message(std::ostream &os, const char *classname, const char *methodname, logging_severity sev,
                            const char *const fmt, ...) {
        static mutex mtx;
        va_list ap;
	va_start(ap, fmt);
        char line[1024];
        vsnprintf(line, 1023, fmt, ap);
	va_end(ap);
        line[1023] = '\0';
        mtx.lock();
        os << sevname[sev] << "=>" << classname << "::" << methodname << ": " << line;
        os.flush();
        gMessageCount[sev][string(classname) + "::" + methodname][line] += 1;
        mtx.unlock();
        return os;
      }

      void summary(std::ostream &os, const string opt) {
        if (opt.find("a") != string::npos) {
          os << string("\n================================== Detailed summary of messages "
                       "=======================================\n");
          for (map<logging_severity, map<string, map<string, int>>>::iterator it = gMessageCount.begin();
               it != gMessageCount.end(); ++it)
            for (map<string, map<string, int>>::iterator jt = it->second.begin(); jt != it->second.end(); ++jt)
              for (map<string, int>::iterator kt = jt->second.begin(); kt != jt->second.end(); ++kt) {
                os << sevname[it->first] << "=>" << jt->first << ":" << kt->first << " # ";
                os << kt->second << endl;
              }
        }
        os << string("================================================================================================="
                     "======\n");
      }

    private:
      const char *const sevname[5] = {"Information", "Warning    ", "Error      ", "Fatal      ", "Debug      "};
      MessageLogger() {}                              // private
      MessageLogger(const MessageLogger&); // not implemented
      MessageLogger& operator=(const MessageLogger&); // not implemented
      static MessageLogger* gMessageLogger;
      static map<logging_severity,map<string,map<string,int> > > gMessageCount;

   };

}
}

#define log_information(os,...) vecgeom::MessageLogger::I()->message(os,ClassName(),__func__,vecgeom::MessageLogger::kInfo,__VA_ARGS__)
#define log_warning(os,...) vecgeom::MessageLogger::I()->message(os,ClassName(),__func__,vecgeom::MessageLogger::kWarning,__VA_ARGS__)
#define log_error(os,...) vecgeom::MessageLogger::I()->message(os,ClassName(),__func__,vecgeom::MessageLogger::kError,__VA_ARGS__)
#define log_fatal(os,...) vecgeom::MessageLogger::I()->message(os,ClassName(),__func__,vecgeom::MessageLogger::kFatal,__VA_ARGS__)
#define log_debug(os,...) vecgeom::MessageLogger::I()->message(os,ClassName(),__func__,vecgeom::MessageLogger::kDebug,__VA_ARGS__)

#endif
