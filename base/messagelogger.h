#ifndef MESSAGELOGGER_H
#define MESSAGELOGGER_H

#include "base/Global.h"
#include <stdio.h>
#include <map>
#include <string>
#include <mutex>
#include <iostream>

using std::map;
using std::string;
using std::mutex;
using std::endl;

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

   class messagelogger {
   public:
      
      typedef enum {kInfo=0, kWarning, kError, kFatal, kDebug} logging_severity;
      
      static messagelogger* I() {
	 static mutex mtx;
	 mtx.lock();
	 if(!gMessageLogger) gMessageLogger = new messagelogger();
	 mtx.unlock();
	 return gMessageLogger;
      }
      std::ostream & message(std::ostream & os, const char* classname,
			     const char* methodname,
			     logging_severity sev,
			     const char *const fmt, ...) {
	 static mutex mtx;
	 va_list ap;
	 char line[256];
	 snprintf(line,255,fmt,ap);
	 line[255]='\0';
	 mtx.lock();
	 os << sevname[sev] << "=>" << classname 
	    << "::" << methodname << ": " << line;
	 os.flush();
	 gMessageCount[sev][string(classname)+"::"+methodname][line]+=1;
	 mtx.unlock();
	 return os;}
		
      void summary(std::ostream &os, const string opt) {
	 if(opt.find("a")!=string::npos) {
	    os << string("\n================================== Detailed summary of messages =======================================\n");
	    for(map<logging_severity,map<string,map<string,int> > >::iterator it=gMessageCount.begin(); 
		it != gMessageCount.end(); ++it)
	       for(map<string,map<string,int> >::iterator jt=it->second.begin(); jt!=it->second.end(); ++jt)
		  for(map<string,int>::iterator kt=jt->second.begin(); kt!=jt->second.end(); ++kt) {
		     os << sevname[it->first] << "=>" << jt->first << ":" << kt->first << " # ";
		     os << kt->second << endl;
		  }
	 }
	 os << string("=======================================================================================================\n");
      }

   private:
      const char *const sevname[5] = {"Information",
				      "Warning    ",
				      "Error      ", 
				      "Fatal      ",
				      "Debug      "};
      messagelogger() {} // private
      messagelogger(const messagelogger&); // not implemented
      messagelogger& operator=(const messagelogger&); // not implemented
      static messagelogger* gMessageLogger;
      static map<logging_severity,map<string,map<string,int> > > gMessageCount;

   };

}
}

#define log_information(os,...) vecgeom::messagelogger::I()->message(os,ClassName(),__func__,vecgeom::messagelogger::kInfo,__VA_ARGS__)
#define log_warning(os,...) vecgeom::messagelogger::I()->message(os,ClassName(),__func__,vecgeom::messagelogger::kWarning,__VA_ARGS__)
#define log_error(os,...) vecgeom::messagelogger::I()->message(os,ClassName(),__func__,vecgeom::messagelogger::kError,__VA_ARGS__)
#define log_fatal(os,...) vecgeom::messagelogger::I()->message(os,ClassName(),__func__,vecgeom::messagelogger::kFatal,__VA_ARGS__)
#define log_debug(os,...) vecgeom::messagelogger::I()->message(os,ClassName(),__func__,vecgeom::messagelogger::kDebug,__VA_ARGS__)

#endif
