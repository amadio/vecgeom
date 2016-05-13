#!/usr/bin/python
#
# File: plotSpeedups.py
#
from ROOT import TCanvas, gROOT, TGraphErrors, TPaveText, TLatex, TFile, gStyle
from PerfHistoryTools import readPerformanceData

### these are the full series

commits   = ["c4154901", "1f673e0d", "458e6f08"]
shapes    = ["Box", "Trapezoid", "Trd"]
algoNames = ["inside", "distToIn", "safeToIn", "contains", "distToOut", "safeToOut"]
implNames = ["vect", "unspec", "spec", "root", "usolids", "geant4"]
#implNames = ["spec", "vect", "cuda"]
refNames  = ["insidespec", "distToInspec","safeToInspec","containsspec","distToOutspec","safeToOutspec"]

### read in the data points from external files

#.. loop over available commits, filling perf data into datasets
datasets = {}
i = 0
while i < len(commits):
    commit = commits[i]
    filename = commit+"/"+commit+"-perf.dat"
    #print "commit: <%s> and file=<%s>" % (commit, filename)
    readPerformanceData( i+1, filename, datasets )
    i = i + 1

#.. show how to loop over data
#for key,val in datasets.iteritems():
#    print val._shape, val._algo, val._impl, len(val._data), val._minkey, val._maxkey
#    for i in range(len(val._data)):
#        print "   ",val._data[i]._x,val._data[i]._y,val._data[i]._dy


#.. Prepare plotting canvas
c1 = TCanvas('c1','c1', 900, 600 )
c1.Divide(3,2)

#.. loop over all shapes requested
for shname in shapes:
    shoffset = len(shname)

    ###.. define references for speedups
    refGraphs = []
    for name in refNames:
        for key,value in datasets.iteritems():
            if key==shname+name:
                #print "Dataset %s: %d datasets between %s and %s" % (key, len(value._data), value._minkey, value._maxkey )
                refGraphs.append( value )

    ### speedupGraphs will contain the normalized datasets
    speedupGraphs = []
    for i in range(6): speedupGraphs.append([])  # 6 empty lists for each shape, one list per algo

    #.. loop over perf data loaded into datasets, to separate the data into the 6 distinct algorithms
    for key,dset in datasets.iteritems() :
        if dset._shape != shname: continue
        algo = dset._algo
        impl = dset._impl
        #print "shname=%s - Dataset %s %s %s: %d elements between %s and %s" % (shname, dset._shape, algo, impl, len(dset._data), dset._minkey, dset._maxkey )

        #.. search perf data for each algorithm requested
        for ialgo in range(6):
            ilen = shoffset + len(algoNames[ialgo])
            if key[shoffset:ilen] == algoNames[ialgo]:
                #evoGraph = dset.tgraphNormError()
                speedupGraph = dset.tgraphSpeedup( refGraphs[ialgo] )

                #.. search perf data for each implementation requested
                for iimpl in range(6):
                    icolor = iimpl+1
                    if icolor>4: icolor+=1
                    #ilen2 = len(implNames[iimpl])
                    if key[ilen:] == implNames[iimpl]:
                        #print "found:",iimpl, implNames[iimpl]
                        speedupGraph.SetLineColor(icolor)
                        speedupGraph.SetMarkerStyle(20+iimpl)
                        speedupGraph.SetMarkerColor(icolor)
                        break
                speedupGraphs[ialgo].append( speedupGraph )

    #.. now plot the 6 separate groups of datasets
    for ialgo in range(len(speedupGraphs)):
        ipad = ialgo + 1

        #.. scan data for ymin,ymax scale adjustment
        ymin = 999999.
        ymax = -ymin
        for igraph in range(len(speedupGraphs[ialgo])):
            tmp = speedupGraphs[ialgo][igraph]
            npts = tmp.GetN()
            x = tmp.GetX()
            y = tmp.GetY()
            for i in range(npts):
                #print "point %i: (%f; %f)" % (i,x[i],y[i])
                if y[i]<ymin: ymin = y[i]
                if y[i]>ymax: ymax = y[i]
            #print "ialgo=%i, igraph=%i, ymin=%f, ymax=%f" % (ialgo,igraph,ymin,ymax)

        gPad = c1.cd(ipad)
        gPad.SetGrid(1,1)
        #gPad.SetLogx(1)
        #gPad.SetLogy(1)
        frame1 = gPad.DrawFrame(0, 0.01, 1+len(commits), 1.05*ymax)
        frame1.SetTitle(algoNames[ialgo])
        frame1.GetXaxis().SetTitle("N tracks")
        frame1.GetYaxis().SetTitle("Speed-up")

        for igra in range(len(speedupGraphs[ialgo])):
            tmp = speedupGraphs[ialgo][igra]
            #tmp.SetLineColor(ialgo+1)
            if(tmp.GetN()>0): tmp.Draw("ELP")

    c1.SetTitle("Speed-ups")
    c1.Update()
    c1.SaveAs(shname+".png")
