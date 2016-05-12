from Dataset import Dataset
import csv

def readPerformanceData(index, filename, datasets):
    """ Reads performance data from file <filename>, stores information
        into the non-empty <datasets> tuple.  
        The <index> will be used as the x-value in the plots.

        Example Usage: see scripts/plotNormalizedEvolution.py
    """
    ok = True
    try:
        file = open(filename, "rt")
    except:
        print "probs reading file?  file=<%s>" % filename
        ok=False
        pass

    if ok:
        try:
            reader = csv.reader(file, delimiter=' ')
            for dirtyRow in reader :
                row = filter(len, dirtyRow)   # remove null elements from list
                #print 'read: <%s>' % row
                if row[0]=="#": continue        # skip comment lines
                if row[1]=="-.------": continue # skip non-existent algorithms
                #print "row[0]=",row[0]
                #print "row[1]=",row[1]
                #print "row[2]=",row[2]
                #print "row[3]=",row[3]
                #print "row[4] = <%s>" % row[4]

                #.. parse input elements
                datatype = row[0]
                perf = float(row[1])
                error = float(row[2])

                #.. parse shape / algorithm / implementation
                istart = 0
                iend = istart + datatype[istart:].find("-")
                shape = datatype[istart:iend]

                istart = iend+1
                iend = istart + datatype[istart:].find("-")
                algo = datatype[istart:iend]

                istart = iend+1
                impl = datatype[istart:]
                key = shape+algo+impl
                try:
                    datasets[key].addPoint(index,perf,error)
                except KeyError:
                    #print "Creating key=<%s>, index=%i, perf=%f, error=%f" % (key, index, perf, error)
                    datasets[key] = Dataset( shape, algo, impl )
                    datasets[key].addPoint(index,perf,error)

        finally:
            file.close()
            print "Commit=<%s> - %i datasets read." % (filename[0:7], len(datasets))
