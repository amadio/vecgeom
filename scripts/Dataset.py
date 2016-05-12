from ROOT import TGraph, TGraphErrors
#import sys
import csv
from datetime import date

class Datapoint:
    def __init__(self,x,y,dy):
        self._x = x
        self._y = y
        self._dy = dy

class Dataset:
    """ Reads performance data from a table: "version algo source timing uncertainty"
    """
    def addPoint(self,x,y,dy):
        self._data.append( Datapoint(x,y,dy) )
        if x < self._minkey : self._minkey = x
        if x > self._maxkey : self._maxkey = x
        if y < self._minvalue : self._minvalue = y
        if y > self._maxvalue : self._maxvalue = y

    def __init__(self, shape, algo, impl):
        """ constructor creates the data series"""
        self._shape = shape
        self._algo = algo
        self._impl = impl
        self._data = []
        self._minkey = 1.0e16
        self._maxkey = -1.0e16
        self._minvalue = 1.0e16
        self._maxvalue = -1.0e16

    def tgraph(self):
        """ returns data points in a TGraphErrors object."""
        if(len(self._data)==0):
            print "***** Dataset.tgraph(): ERROR: no data available for", self._algo+'_'+self._impl
            return TGraphErrors()
        result = TGraphErrors( len(self._data) )
        i = 0
        #for k in sorted(self._data.iterkeys()):
        for k in self._data:
            x = k._x
            y = k._y
            #if y < self._minvalue : self._minvalue = y
            #if y > self._maxvalue : self._maxvalue = y
            print "raw %s: i=%d vers=%d value=%f minVal=%f maxVal=%f" % (self._name, i, k, v, self._minvalue, self._maxvalue)
            result.SetPoint( i, x, y )
            result.SetPointError( i, 0, k._dy )
            i = i + 1;
        return result


    def tgraphNorm(self):
        """ produces a timing series normalized to value=1.0 at the first data point"""
        if(len(self._data)==0):
            print "***** Dataset.tgraphNorm(): ERROR: no data available for", self._algo+'_'+self._impl
            return TGraph()
        result = TGraph( len(self._data) )
        normFactor = 1. / self._data[0]._y
        i = 0
        for k in self._data:
            normValue = k._y * normFactor
            if normValue < self._minvalue : self._minvalue = normValue
            if normValue > self._maxvalue : self._maxvalue = normValue
            #print "Normalizing %s: i=%d days=%d val=%f normFactor=%f stored=%f minVal=%f maxVal=%f" % (self._name, i, k, self.serie[k], normFactor, normValue, self._minvalue, self._maxvalue)
            result.SetPoint( i, k._x, normValue )
            i = i + 1;
        return result

    def tgraphNormError(self):
        """ returns a TGraphErrors object normalized to value=1.0 at the first data point"""
        if(len(self._data)==0):
            print "***** Dataset.tgraphNormError(): ERROR: no data available for", self._algo+'_'+self._impl
            return TGraphErrors()
        result = TGraphErrors( len(self._data) )
        normFactor = 1. / self._data[0]._y
        i = 0
        for k in self._data:
            normValue = k._y * normFactor
            if normValue < self._minvalue : self._minvalue = normValue
            if normValue > self._maxvalue : self._maxvalue = normValue
            #print "Normalizing %s: i=%d days=%d val=%f normFactor=%f stored=%f minVal=%f maxVal=%f" % (self._name, i, k, self.serie[k], normFactor, normValue, self._minvalue, self._maxvalue)
            result.SetPoint( i, k._x, normValue )
            result.SetPointError( i, 0, k._dy*normFactor )
            i = i + 1;
        return result


    def tgraphRatio(self, refDataset):
        """ produces timing ratio graphs"""
        if(len(self._data)==0):
            print "***** Dataset.tgraphNorm(): ERROR: no data available for", self._algo+'_'+self._impl
            return TGraphErrors()

        result = TGraph( len(self._data) )
        for i in range(len(self._data)):
            ratioValue = self._data[i]._y / refDataset._data[i]._y
            result.SetPoint( i, self._data[i]._x, ratioValue )
            if ratioValue < self._minvalue : self._minvalue = ratioValue
            if ratioValue > self._maxvalue : self._maxvalue = ratioValue
            #print "Normalizing %s: i=%d days=%d val=%f normFactor=%f stored=%f minVal=%f maxVal=%f" % (self._name, i, k, self.serie[k], normFactor, normValue, self._minvalue, self._maxvalue)
        return result

    def tgraphSpeedup(self, refDataset):
        """ produces speed-up graphs"""
        if(len(self._data)==0):
            print "***** Dataset.tgraphNorm(): ERROR: no data available for", self._algo+'_'+self._impl
            return TGraphErrors()

        result = TGraph( len(self._data) )
        #print "Speed-up: sizes: refDataset=%i self=%i" % (len(refDataset._data), len(self._data))
        for i in range(len(self._data)):
            speedup = refDataset._data[i]._y / self._data[i]._y
            result.SetPoint( i, self._data[i]._x, speedup )
            if speedup < self._minvalue : self._minvalue = speedup
            if speedup > self._maxvalue : self._maxvalue = speedup
        return result

    def getEarliestVersion(self):
        """ return earliest data point in a time series"""
        return self._minkey

    def getLatestVersion(self):
        """ return latest data point in a time series"""
        return self._maxkey

    def getLowestValue(self):
        """ return lowest data point in a time series"""
        if self._maxvalue<=0 and self._minvalue>=99999:
            print "ERROR in getLowestValue:",self._name,": make sure to call tgraphNorm() first!!!"
        return self._minvalue

    def getHighestValue(self):
        """ return highest value in a time series"""
        if self._maxvalue<=0 and self._minvalue>=99999:
            print "ERROR in getHighestValue:",self._name,": make sure to call tgraphNorm() first!!!"
        return self._maxvalue

    def getLastValue(self):
        return ( self.serie[self._maxkey] / self.serie[self._minkey] )

    def getFirstValue(self):
        return ( self.serie[self._minkey] / self.serie[self._minkey] )

### Testing

if __name__ == "__main__":

    ### these are the full series (without normalization)
    algoNames = ["inside", "distToIn", "safetyToIn", "contains", "distToOut", "safetyToOut"]
    implNames = ["root", "usolids", "unspec", "vector", "spec"]

    ### read in the data points from external file
    datasets = {}
    for algo in algoNames:
        for impl in implNames:
            datasets[ algo+impl ] = Dataset( algo, impl )

    ok = True
    try:
        file = open("trap-perf-hist.dat", "rt")
    except:
        print "probs reading file?  file=", filename
        ok=False
        pass

    if ok:
        try:
            reader = csv.reader(file, delimiter=' ')
            first=False
            for row in reader :
                if not first :
                    #print 'read: <%s>' % row
                    #print "row[0] = <%s>" % row[0]
                    #print "row[0][6:]=",row[0][6:]
                    #print "row[0][3:5]=",row[0][3:5]
                    #print "row[0][0:2]=",row[0][0:2]
                    vers = int(row[0])
                    algo = row[1]
                    impl = row[2]
                    perf = float(row[3])
                    error = float(row[4])
                    datasets[algo+impl].addPoint(vers,perf,error)
                    #print vers,algo,impl,perf,error
                else :
                    #.. just skip reading header line in file
                    #print "len(row)=",len(row),":",row
                    if len(row)>0 and row[0]=='Data': first=False
                    pass
        finally:
            file.close()
            print len(datasets),"datasets read."


#    #.. show how to loop over data
#    for key,val in datasets.iteritems():
#        print val._algo, val._impl, len(val._data),val._minkey,val._maxkey
#        for i in range(len(val._data)):
#            print "   ",val._data[i]._x,val._data[i]._y,val._data[i]._dy

    #.. Make plots
    from ROOT import TCanvas, gROOT, TGraphErrors, TPaveText, TLatex, TH1F, TProfile, TFile, gStyle

    ### Plot 1

    c1 = TCanvas('c1','c1', 900, 600 )
    c1.Divide(3,2)

    ### graphs contain the normalized datasets
    graphs = []
    for i in range(6): graphs.append([])  # 6 empty lists, one for each algo

    #.. separate the data into the 6 different algorithms
    for key,value in datasets.iteritems() :
        #print "Dataset %s: %d elements between %s and %s" % (key, len(value._data), value._minkey, value._maxkey )
        for ialgo in range(6):
            ilen = len(algoNames[ialgo])
            if key[0:ilen] == algoNames[ialgo]:
                normGraph = value.tgraphNorm()
                for impl in range(6):
                    icolor = impl+2
                    if icolor>=5: icolor+=1
                    if key[ilen:] == implNames[impl]:
                        normGraph.SetLineColor(icolor)
                        normGraph.SetMarkerStyle(20+icolor)
                        normGraph.SetMarkerColor(icolor)
                        break
                #graphs[ialgo].append( normGraph )
                graphs[ialgo].append(  )

    #.. now plot the 6 separate groups of datasets
    for ialgo in range(len(graphs)):
        ipad = ialgo + 1
        ymin = 1.1
        ymax = 0.9
        for igraph in range(len(graphs[ialgo])):
            tmp = graphs[ialgo][igraph]
            npts = tmp.GetN()
            x = tmp.GetX()
            y = tmp.GetY()
            for i in range(npts):
                #print "point %i: (%f; %f)" % (i,x[i],y[i])
                if y[i]<ymin: ymin = y[i]
                if y[i]>ymax: ymax = y[i]
            #print "ialgo=%i, igraph=%i, ymin=%f, ymax=%f" % (ialgo,igraph,ymin,ymax)

        thePad = c1.cd(ipad)
        thePad.SetGrid(1,1)
        frame1 = thePad.DrawFrame(0, 0.95*ymin, 10, 1.05*ymax)
        frame1.SetTitle(algoNames[ialgo])
        frame1.GetXaxis().SetTitle("Version")
        frame1.GetYaxis().SetTitle("Time [sec]")

        for igra in range(len(graphs[ialgo])):
                tmp = graphs[ialgo][igra]
                #tmp.SetLineColor(ialgo+1)
                if(tmp.GetN()>0): tmp.Draw("pl")

    c1.Update()
    c1.SaveAs("last.png")
