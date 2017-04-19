#!/usr/bin/python
#
# File: plotEvolution.py
#
from ROOT import TCanvas, gROOT, TGraphErrors, TPaveText, TLatex, TFile, gStyle
from PerfHistoryTools import readPerformanceData

### these are the full series

commits = ["68c48276","4e5928c1","019d29a0","b356bad1","2c007289",
           "298df455","a860eea3","e569cfb7","9e47e5fd","0877406d"]
#           "143d582e","b92b66ba","3af4b3f2","b9437a06","e4c54450","c2f9b8cc","ec7c80a5","1047750f","30635340"]
#commits = ["143d582e","b92b66ba","3af4b3f2","b9437a06","e4c54450","c2f9b8cc","ec7c80a5","1047750f","30635340"]
shapes    = ["Trapezoid"]
algoNames = ["inside", "distToIn", "safeToIn", "contains", "distToOut", "safeToOut"]
implNames = ["vect", "unspec", "spec", "root", "usolids", "geant4"]

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

### Plot 1

c1 = TCanvas('c1','c1', 900, 600 )
c1.Divide(3,2)

#.. loop over all shapes requested
for shname in shapes:
    shoffset = len(shname)

    ### perfGraphs will contain the proc.times per dataset
    perfGraphs = []
    for i in range(6): perfGraphs.append([])  # 6 empty lists for each shape, one list per algo

    #.. loop over perf data loaded into datasets
    for key,dset in datasets.iteritems() :
        if dset._shape != shname: continue
        #print "shname=%s - Dataset %s %s %s: %d elements between %s and %s" \
        #   % (shname, dset._shape, dset._algo, dset._impl, len(dset._data), dset._minkey, dset._maxkey )

        #.. search perf data for each algorithm requested
        for ialgo in range(6):
            ilen = shoffset + len(algoNames[ialgo])
            if key[shoffset:ilen] == algoNames[ialgo]:
                #.. convert dataset into root TGraphs with errors
                tmpGraph = dset.tgraph()

                #.. search perf data for each implementation requested
                for iimpl in range(6):
                    icolor = iimpl+1
                    if icolor>4: icolor=icolor+1
                    #ilen2 = len(implNames[iimpl])
                    if key[ilen:] == implNames[iimpl]:
                        #.. here for a shape/algo/implementation combination
                        #print "found:",impl, implNames[iimpl] #,perfGraphs[ialgo].GetNumberOfElements()
                        tmpGraph.SetLineColor(icolor)
                        tmpGraph.SetMarkerStyle(20+iimpl)
                        tmpGraph.SetMarkerColor(icolor)
                        break
                perfGraphs[ialgo].append( tmpGraph )

    #.. now plot the 6 separate groups of datasets
    for ialgo in range(len(perfGraphs)):
        ipad = ialgo + 1

        #.. scan data for ymin,ymax scale adjustment
        ymin = 999999.
        ymax = -ymin
        for igraph in range(len(perfGraphs[ialgo])):
            tmp = perfGraphs[ialgo][igraph]
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
        frame1 = gPad.DrawFrame(0, 0.95*ymin, 1+len(commits), 1.05*ymax)
        frame1.SetTitle(algoNames[ialgo])
        frame1.GetXaxis().SetTitle("Version")
        frame1.GetYaxis().SetTitle("CPU time [s]")

        for igra in range(len(perfGraphs[ialgo])):
            tmp = perfGraphs[ialgo][igra]
            #tmp.SetLineColor(ialgo+1)
            if(tmp.GetN()>0): tmp.Draw("ELP")

    c1.Update()
    c1.SaveAs("perf-"+shname+".png")
